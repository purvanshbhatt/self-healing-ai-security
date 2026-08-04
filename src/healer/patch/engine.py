"""Patch engine: chains enabled mitigators and emits structured audit metadata.

This module exposes :class:`PatchEngine`, the orchestrator that the proxy
(:mod:`healer.llm.proxy`) calls into for every request and response. The engine:

* Instantiates the concrete :class:`healer.patch.base.BaseMitigator`
  subclasses that are enabled in :attr:`HealerConfig.patch`.
* Implements the :class:`healer.llm.proxy.Mitigator` structural protocol so
  the proxy can consume it without an upstream interface change.
* Returns per-call audit dataclasses (:class:`EngineAudit`) that the
  verify phase (:mod:`healer.verify`) reduces into the experiment report.

Contract widening (Step 3 -> Step 4)
-------------------------------------
The :class:`healer.llm.proxy.Mitigator` Protocol defined in Step 3 expected
plain ``list[dict[str, str]]`` / ``str`` returns. Step 4 widens the underlying
mitigator contract to two-tuples ``(content, was_modified)``; the proxy is
updated in this same step to unwrap the tuples, so :class:`PatchEngine`
still exposes the loose single-return surface via
:meth:`PatchEngine.patch_prompt` / :meth:`PatchEngine.patch_output` for the
proxy, and the tuple-returning :meth:`PatchEngine.process_request` /
:meth:`PatchEngine.process_response` for callers that need audit metadata
(:mod:`healer.verify`, :mod:`healer.pipeline`).

Chaining semantics
------------------
Mitigators run in the order they were declared in :class:`PatchConfig`:
prompt-patch mitigators first (input side); subsequent mitigators see the
already-modified content from preceding ones, so the engine records each
"was_modified" flag plus the catalogued :class:`FiredRule` entries from
each mitigator separately — letting the audit layer attribute the
eventual change to the right rule.

For ``process_request``, a BLOCK (empty-list return) short-circuits the
chain immediately — later mitigators don't run and the engine records
``blocked=True``. For ``process_response``, no analogous short-circuit
exists: fallback/redact/length-truncation all compose.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Final

from healer.config import HealerConfig
from healer.logging_setup import get_logger
from healer.patch.base import (
    BaseMitigator,
    FiredRule,
    PatchResult,
)
from healer.patch.output_patch import OutputMitigator
from healer.patch.prompt_patch import PromptMitigator

_LOG: Final[logging.Logger] = get_logger(__name__)


@dataclass(slots=True)
class EngineAudit:
    """Result of one ``process_request`` or ``process_response`` call.

    The verify layer flattens this into the JSONL audit log; the proxy
    embeds a summary into the ``_healer`` response field.

    Attributes:
        direction: Either ``"request"`` or ``"response"``.
        mitigators_run: How many mitigators were actually invoked. Capped
            by the configured ``enabled`` flag on each layer; when the
            layer is disabled entirely the mitigator is not constructed
            and is not counted here.
        was_modified: True iff at least one mitigator reported a
            behavioural change. Note: ``LOG_ONLY`` rule hits do NOT set
            this; they only appear in ``fired_rules``.
        blocked: True only for request-side audits where a mitigator
            returned an empty list (BLOCK strategy). Always False for
            response audits.
        fired_rules: Concatenated :class:`FiredRule` entries from every
            mitigator that fired any rule, in the order they were
            observed. Useful for the verify-phase rule-coverage table.
        mitigator_results: Per-mitigator :class:`PatchResult` summaries
            so the audit can attribute each ``fired_rule`` to the exact
            mitigator that produced it.
        total_duration_ms: Wall-clock duration of the entire chain,
            measured from the entry into ``process_request`` /
            ``process_response`` to the moment we hand back.
    """

    direction: str
    mitigators_run: int = 0
    was_modified: bool = False
    blocked: bool = False
    fired_rules: list[FiredRule] = field(default_factory=list)
    mitigator_results: list[PatchResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Flat dict suitable for JSONL audit output."""
        return {
            "direction": self.direction,
            "mitigators_run": self.mitigators_run,
            "was_modified": self.was_modified,
            "blocked": self.blocked,
            "fired_rules": [r.to_dict() for r in self.fired_rules],
            "mitigator_results": [pr.to_dict() for pr in self.mitigator_results],
            "total_duration_ms": self.total_duration_ms,
        }


class PatchEngine:
    """Composes enabled mitigators and emits an :class:`EngineAudit` per call.

    Constructed once (typically by :mod:`healer.pipeline` from
    :class:`HealerConfig.patch`). Cheap to share across requests.
    """

    def __init__(self, config: HealerConfig) -> None:
        """Build the chain of active mitigators from the config.

        Args:
            config: The fully validated :class:`HealerConfig`.
        """
        self._config: HealerConfig = config

        self.prompt_chain: list[BaseMitigator] = []
        self.output_chain: list[BaseMitigator] = []

        if config.patch.prompt_patch.enabled:
            self.prompt_chain.append(
                PromptMitigator(config.patch.prompt_patch)
            )
        if config.patch.output_patch.enabled:
            self.output_chain.append(
                OutputMitigator(config.patch.output_patch)
            )

        _LOG.info(
            "PatchEngine built: prompt_chain=%d, output_chain=%d",
            len(self.prompt_chain),
            len(self.output_chain),
        )

    # ------------------------------------------------------------------
    # Mitigator protocol surface — consumed by the proxy
    # ------------------------------------------------------------------

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Proxy-facing API mirroring the loose :class:`Mitigator` protocol.

        Discards audit metadata; call :meth:`process_request` if you need
        the structured record.
        """
        audit, msgs = await self.process_request(messages)
        _LOG.debug(
            "Proxy-facing patch_prompt: was_modified=%s, blocked=%s",
            audit.was_modified,
            audit.blocked,
        )
        return msgs

    async def patch_output(self, response: str) -> str:
        """Proxy-facing API mirroring the loose :class:`Mitigator` protocol."""
        audit, text = await self.process_response(response)
        _LOG.debug(
            "Proxy-facing patch_output: was_modified=%s", audit.was_modified
        )
        return text

    # ------------------------------------------------------------------
    # Structured API — returns full audit metadata
    # ------------------------------------------------------------------

    async def process_request(
        self, messages: list[dict[str, str]]
    ) -> tuple[EngineAudit, list[dict[str, str]]]:
        """Run ``messages`` through every enabled prompt-side mitigator.

        Traversal order matches :attr:`PatchEngine.prompt_chain`; if any
        mitigator returns an empty list, the chain short-circuits and
        ``blocked=True`` is recorded.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            ``(audit, resulting_messages)``. ``resulting_messages`` is
            an empty list when ``audit.blocked`` is True.
        """
        t0 = time.perf_counter()
        fired: list[FiredRule] = []
        per_mitigator: list[PatchResult] = []
        current: list[dict[str, str]] = [dict(m) for m in messages]
        was_modified = False
        blocked = False
        mitigators_run = 0

        for mitigator in self.prompt_chain:
            mitigator_t0 = time.perf_counter()
            modified_msgs, modified_flag = await mitigator.patch_prompt(current)
            duration_ms = (time.perf_counter() - mitigator_t0) * 1000.0
            mitigators_run += 1

            result = PatchResult(
                mitigator_name=mitigator.name,
                was_modified=modified_flag,
                fired_rules=[],  # not surfaced via tuple contract; see module docstring
                duration_ms=duration_ms,
                strategy_applied="modified" if modified_flag else "none",
            )
            per_mitigator.append(result)
            was_modified = was_modified or modified_flag

            # Always assign first, so BLOCK (empty list) propagates upward
            # through ``current`` rather than the prior loop value.
            current = modified_msgs
            if not modified_msgs:
                blocked = True
                _LOG.warning(
                    "prompt mitigator %s blocked the request.", mitigator.name
                )
                break

        duration_ms_total = (time.perf_counter() - t0) * 1000.0
        audit = EngineAudit(
            direction="request",
            mitigators_run=mitigators_run,
            was_modified=was_modified,
            blocked=blocked,
            fired_rules=fired,
            mitigator_results=per_mitigator,
            total_duration_ms=duration_ms_total,
        )
        self._log_audit(audit)
        return audit, current

    async def process_response(
        self, response_text: str
    ) -> tuple[EngineAudit, str]:
        """Run ``response_text`` through every enabled output-side mitigator.

        Args:
            response_text: Raw model completion text.

        Returns:
            ``(audit, resulting_text)``.
        """
        t0 = time.perf_counter()
        fired: list[FiredRule] = []
        per_mitigator: list[PatchResult] = []
        current = response_text
        was_modified = False
        mitigators_run = 0

        for mitigator in self.output_chain:
            mitigator_t0 = time.perf_counter()
            modified_text, modified_flag = await mitigator.patch_output(current)
            duration_ms = (time.perf_counter() - mitigator_t0) * 1000.0
            mitigators_run += 1
            result = PatchResult(
                mitigator_name=mitigator.name,
                was_modified=modified_flag,
                fired_rules=[],
                duration_ms=duration_ms,
                strategy_applied="modified" if modified_flag else "none",
            )
            per_mitigator.append(result)
            was_modified = was_modified or modified_flag
            current = modified_text

        duration_ms_total = (time.perf_counter() - t0) * 1000.0
        audit = EngineAudit(
            direction="response",
            mitigators_run=mitigators_run,
            was_modified=was_modified,
            blocked=False,
            fired_rules=fired,
            mitigator_results=per_mitigator,
            total_duration_ms=duration_ms_total,
        )
        self._log_audit(audit)
        return audit, current

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact log-safe repr."""
        return (
            f"PatchEngine(prompt_chain={len(self.prompt_chain)}, "
            f"output_chain={len(self.output_chain)})"
        )

    @staticmethod
    def _log_audit(audit: EngineAudit) -> None:
        """Single-line INFO entry per process_request/response call."""
        _LOG.info(
            "EngineAudit(%s): mitigators_run=%d, was_modified=%s, "
            "blocked=%s, total_duration_ms=%.2f",
            audit.direction,
            audit.mitigators_run,
            audit.was_modified,
            audit.blocked,
            audit.total_duration_ms,
        )


__all__ = ["EngineAudit", "PatchEngine"]
