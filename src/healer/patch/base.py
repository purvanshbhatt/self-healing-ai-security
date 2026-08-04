"""Abstract base for all mitigation strategies.

This module defines :class:`BaseMitigator`, the abstract contract every
prompt-level and output-level mitigation strategy must satisfy, along with
the per-call audit dataclasses :class:`FiredRule` and :class:`PatchResult`
that flow upward into :class:`healer.patch.engine.PatchEngine` for forensic
reporting in the verify phase.

The contract intentionally differs from the loose
:class:`healer.llm.proxy.Mitigator` :class:`typing.Protocol` defined in Step 3:
the protocol is structural (used by the proxy for duck-typed injection), while
:class:`BaseMitigator` is a behavioural conformance point — every concrete
mitigator in :mod:`healer.patch.prompt_patch` and
:mod:`healer.patch.output_patch` inherits from it.

Return-type contract (shared by both layer methods)
---------------------------------------------------
``tuple[<content>, bool]`` where:

* ``content`` — the resulting message-list or output-text after mitigation.
  Implementations MUST return a value of the same shape as the input even
  when no rule fires (so call sites can always assign back unconditionally).
* ``was_modified`` — ``True`` iff at least one rule fired AND the configured
  strategy modified the content (``BLOCK``, ``REWRITE``, ``FALLBACK``,
  ``REDACT``). For ``LOG_ONLY``, ``was_modified`` is ``False`` even when a
  rule fired — the canonical signal that the rule fired lives in the
  engine's per-call audit, not in the boolean.

This separation lets the verify layer distinguish three meaningful states
from a single mitigated response:

* no rule fired           -> behaviour unchanged
* rule fired, log_only    -> behaviour unchanged, audit captured
* rule fired, mute action -> behaviour changed AND audit captured
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Final


class BaseMitigator(abc.ABC):
    """Abstract contract for all mitigation strategies.

    Concrete subclasses are expected to be cheap to construct and own their
    configuration models. There is no shared state across instances beyond
    what each subclass's constructor captures; this keeps the mitigators
    trivially testable in isolation and lets :class:`PatchEngine` compose
    multiple mitigators without coordination concerns.
    """

    #: Short human-readable name used in audit records. Subclasses must
    #: set this; it ends up in the engine's per-call ``FiredRule.mitigator``
    #: field and in the structured log lines.
    name: str = "BaseMitigator"

    @abc.abstractmethod
    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        """Sanitise the inbound chat messages before the upstream call.

        Args:
            messages: OpenAI-format chat messages. Each message is a dict
                with at minimum ``role`` and ``content`` string values.
                Implementations MUST NOT mutate the input list in place —
                they must return a fresh list. (Returning a copy is cheap
                and keeps callers free of aliasing surprises.)

        Returns:
            A 2-tuple ``(modified_messages, was_modified)``. ``modified_messages``
            is the post-mitigation message list; ``was_modified`` is True
            iff the configured strategy actually altered the content.

        Raises:
            PatchPolicyError: Optional — the BLOCK strategy may raise this
                instead of returning an empty list, but the canonical
                contract is to return ``[]`` and let the engine / proxy
                interpret it. Implementations that raise will be caught
                by :class:`PatchEngine` and translated to a block audit.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def patch_output(self, response_text: str) -> tuple[str, bool]:
        """Filter or replace the model's output before it leaves the proxy.

        Args:
            response_text: Raw assistant completion text as returned by
                :class:`healer.llm.target.TargetLLMClient`.

        Returns:
            A 2-tuple ``(modified_text, was_modified)``.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FiredRule:
    """Audit record describing a single rule that fired during a mitigator run.

    Attributes:
        mitigator: Name of the :class:`BaseMitigator` subclass instance that
            performed the scan. Mirrors :attr:`BaseMitigator.name`.
        rule_name: Identifier of the matching rule. For prompt mitigators this
            is the ``name`` field of an entry in ``injection_patterns.yaml``;
            for output mitigators it is the matched category (e.g. ``"pii"``)
            plus the offending keyword.
        category: Free-form bucket (``"injection"`` for prompt, the policy
            category for output). Useful for grouping audit buckets post-hoc.
        strategy: The :class:`healer.config.OnMatch` enum value applied.
        matched_substring: The offending substring that triggered the rule,
            truncated to 80 chars. Useful for triage without re-running.
    """

    mitigator: str
    rule_name: str
    category: str
    strategy: str
    matched_substring: str = ""

    def to_dict(self) -> dict[str, str]:
        """Render as a flat dict suitable for JSONL audit logs."""
        return {
            "mitigator": self.mitigator,
            "rule_name": self.rule_name,
            "category": self.category,
            "strategy": self.strategy,
            "matched_substring": self.matched_substring,
        }


@dataclass(slots=True)
class PatchResult:
    """Complete per-mitigator call outcome used by :class:`PatchEngine`.

    The engine chains multiple mitigators and aggregates ``FiredRule`` audit
    entries across them. This dataclass is the unit of work in that chain.

    Attributes:
        mitigator_name: Name of the :class:`BaseMitigator` that ran.
        was_modified: ``True`` iff the mitigator actually altered content.
        fired_rules: List of :class:`FiredRule` audit entries produced
            during this call. Empty when no rule fired.
        duration_ms: Wall-clock duration of the mitigator call, milliseconds.
        strategy_applied: The highest-severity strategy actually executed
            across all fired rules (``"none"`` when nothing fired).
    """

    mitigator_name: str
    was_modified: bool = False
    fired_rules: list[FiredRule] = field(default_factory=list)
    duration_ms: float = 0.0
    strategy_applied: str = "none"

    def to_dict(self) -> dict[str, object]:
        """Render as a flat dict suitable for JSONL audit logs."""
        return {
            "mitigator_name": self.mitigator_name,
            "was_modified": self.was_modified,
            "fired_rules": [r.to_dict() for r in self.fired_rules],
            "duration_ms": self.duration_ms,
            "strategy_applied": self.strategy_applied,
        }


class PatchPolicyError(Exception):
    """Raised (optionally) by a BLOCK-strategy mitigator.

    The canonical BLOCK contract is to return an empty message list, but
    concrete mitigators wishing to short-circuit early (e.g., to skip
    subsequent mitigators in the chain) may raise this; the engine catches
    it and records a synthetic ``FiredRule`` with ``strategy="block"``.
    """

    #: Human-readable reason captured in the audit record.
    reason: str

    def __init__(self, reason: str = "policy-block") -> None:
        """Store the block reason for audit logging.

        Args:
            reason: Short machine-readable cause. Mirrors the
                ``reason`` parameter in :class:`FallbackResponse`.
        """
        super().__init__(reason)
        self.reason = reason


#: Sentinel strategy string recorded when no rule fired.
NO_STRATEGY: Final[str] = "none"

#: Sentinel category used for prompt-layer mitigators in audit records.
PROMPT_CATEGORY: Final[str] = "injection"

#: Sentinel category used for output-layer mitigators in audit records.
OUTPUT_CATEGORY: Final[str] = "policy"


__all__ = [
    "NO_STRATEGY",
    "OUTPUT_CATEGORY",
    "PROMPT_CATEGORY",
    "BaseMitigator",
    "FiredRule",
    "PatchPolicyError",
    "PatchResult",
]
