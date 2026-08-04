"""Prompt-level mitigation: detect and neutralise prompt-injection attempts.

This module implements :class:`PromptMitigator`, the input-side guard
responsible for sanitising user messages before they reach the target LLM.
Detection rules live in ``data/prompts/injection_patterns.yaml`` and are
loaded once at construction time (compiled to :class:`re.Pattern` for
sub-millisecond scanning).

Strategy matrix (matches :class:`healer.config.OnMatch`)
---------------------------------------------------------
* ``block``    -> return ``[]`` from :meth:`patch_prompt`. The proxy converts
  this to HTTP 400 ``blocked_by_prompt_patch`` and the upstream call never
  fires. No :class:`PatchPolicyError` is raised — the empty-list contract
  is canonical so chained mitigators can short-circuit cleanly.
* ``rewrite`` -> substitute the matched user message with the configured
  :attr:`PromptPatchConfig.strategies.detect_injection.rewrite_template`
  after substituting the ``{sanitized_input}`` placeholder with the original
  content minus the matched fragment.
* ``log_only`` -> leave the message unchanged but emit a :class:`FiredRule`
  so the verify-phase audit can still quantify detection coverage.

Threading & caching
-------------------
The patterns file is parsed once at construction; subsequent calls only run
regex scans. :class:`re.Pattern` objects are thread-safe for matching, so
the mitigator is safe to share across requests in the proxy.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from healer.config import OnMatch, PromptPatchConfig
from healer.logging_setup import get_logger
from healer.patch.base import (
    PROMPT_CATEGORY,
    BaseMitigator,
    FiredRule,
)

_LOG: Final[logging.Logger] = get_logger(__name__)

#: Placeholder string in ``rewrite_template`` substituted with the sanitized
#: version of the user's original content (matches existing config.yaml).
SANITIZED_INPUT_PLACEHOLDER: Final[str] = "{sanitized_input}"

#: Default mark used to redact matched injection fragments before the
#: template substitution. Kept explicit rather than importing from a
#: downstream module to avoid circular dependencies on the output-corner.
_REDACTION_MARK: Final[str] = "[redacted-injection]"

#: Maximum length preserved from the offending substring when capturing it
#: in the audit record. Keeps JSONL logs bounded.
_MAX_AUDIT_SUBSTRING_LEN: Final[int] = 80


@dataclass(frozen=True, slots=True)
class _CompiledPattern:
    """In-memory representation of one entry from injection_patterns.yaml.

    Attributes:
        name: Stable identifier from the YAML file. Used as ``rule_name``
            in :class:`FiredRule` so audits can re-identify the rule that
            fired without re-loading the YAML.
        pattern: Compiled regex. Case-insensitive (compiled with ``re.I``
            + ``re.M``) since attackers frequently vary capitalisation.
    """

    name: str
    pattern: re.Pattern[str]


class PromptMitigator(BaseMitigator):
    """Input-side mitigation: detect and neutralise prompt injection."""

    name: str = "PromptMitigator"

    def __init__(
        self,
        config: PromptPatchConfig,
        *,
        patterns_file: Path | None = None,
    ) -> None:
        """Initialise the mitigator and eagerly compile its pattern table.

        Args:
            config: The :class:`PromptPatchConfig` block from
                :class:`healer.config.HealerConfig.patch.prompt_patch`.
            patterns_file: Optional explicit path to the pattern YAML.
                Defaults to the path declared inside ``config``; useful for
                tests that want to point at a temp fixture.

        Raises:
            FileNotFoundError: The patterns file does not exist.
            ValueError: A regex in the YAML failed to compile.
        """
        self._config: PromptPatchConfig = config
        self._patterns: list[_CompiledPattern] = self._load_patterns(
            patterns_file or config.strategies.detect_injection.patterns_file
        )
        di = config.strategies.detect_injection
        if not di.enabled:
            _LOG.info("PromptMitigator initialised but detect_injection is disabled.")
        else:
            _LOG.info(
                "PromptMitigator loaded %d patterns (on_match=%s).",
                len(self._patterns),
                di.on_match.value,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        """Scan all user-role messages for injection signatures.

        Detection runs over a *copy* of the input list — the original is
        never mutated, even when ``was_modified`` ends up ``False``.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            ``(modified_messages, was_modified)``. ``modified_messages`` is
            an empty list when the configured strategy is ``block`` and at
            least one rule fires.
        """
        t0 = time.perf_counter()
        di = self._config.strategies.detect_injection

        # When the whole mechanism is disabled, we're a no-op pass-through.
        if not (self._config.enabled and di.enabled):
            _LOG.debug("PromptMitigator disabled -> passthrough.")
            return [dict(m) for m in messages], False

        modified: bool = False
        out: list[dict[str, str]] = []
        fired: list[FiredRule] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only user-role messages carry injection risk from the caller's
            # perspective; assistant/system messages are trusted upstream of
            # the proxy (they're constructed by us or by garak's harness, both
            # of which we control).
            if role != "user":
                out.append(dict(msg))
                continue

            match = self._first_match(content)
            if match is None:
                out.append(dict(msg))
                continue

            fired.append(self._build_fired_rule(match))
            _LOG.info(
                "Injection match: rule=%s, strategy=%s, content_preview=%r",
                match.pattern_name,
                di.on_match.value,
                content[:80],
            )

            if di.on_match == OnMatch.BLOCK:
                # Return immediately; the canonical block token is [].
                _LOG.warning(
                    "Blocking request — rule=%s matched in user prompt.",
                    match.pattern_name,
                )
                self._log_result(fired, t0, "block", True)
                return [], True
            elif di.on_match == OnMatch.REWRITE:
                sanitized = self._sanitized_content(content, match)
                new_content = di.rewrite_template.replace(
                    SANITIZED_INPUT_PLACEHOLDER, sanitized
                ).strip()
                out.append({**msg, "content": new_content})
                modified = True
            elif di.on_match == OnMatch.LOG_ONLY:
                # Audit-only; pass through unchanged.
                out.append(dict(msg))
            else:  # pragma: no cover - exhaustive enum guard
                _LOG.warning(
                    "Unknown OnMatch=%s; treating as LOG_ONLY.",
                    di.on_match,
                )
                out.append(dict(msg))

        self._log_result(fired, t0, di.on_match.value, modified)
        return out, modified

    async def patch_output(self, response_text: str) -> tuple[str, bool]:
        """Prompt mitigators never edit outputs — identity return.

        The contract still needs an async implementation, so this is a
        cheap ``await``-free passthrough.
        """
        return response_text, False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_patterns(path: Path) -> list[_CompiledPattern]:
        """Parse and pre-compile every pattern entry from the patterns YAML.

        Eager compilation at construction time turns each request-time scan
        into a cheap ``pattern.search(content)`` call instead of a parse +
        compile loop per request — important under probe-batch load.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Injection patterns file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Pattern YAML parse error in {path}: {exc}") from exc

        entries = raw.get("patterns") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise ValueError(  # noqa: TRY004 - isinstance-check on YAML structure
                f"Pattern YAML {path} missing top-level 'patterns' list."
            )

        compiled: list[_CompiledPattern] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(  # noqa: TRY004 - isinstance-check on YAML structure
                    f"Pattern entry [{idx}] in {path} is not a mapping: {entry!r}"
                )
            name = str(entry.get("name", f"rule_{idx}"))
            regex = str(entry.get("regex", ""))
            if not regex:
                raise ValueError(f"Pattern entry [{idx}] ({name}) missing regex.")
            try:
                pattern = re.compile(regex, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                raise ValueError(
                    f"Pattern entry [{idx}] ({name}) regex compile failed: {exc}"
                ) from exc
            compiled.append(_CompiledPattern(name=name, pattern=pattern))
        return compiled

    def _first_match(self, content: str) -> _PatternHit | None:
        """Return the first pattern that matches anywhere in ``content``."""
        for pat in self._patterns:
            m = pat.pattern.search(content)
            if m is not None:
                return _PatternHit(
                    pattern_name=pat.name,
                    matched_substring=m.group(0),
                    span=m.span(),
                )
        return None

    @staticmethod
    def _sanitized_content(content: str, hit: _PatternHit) -> str:
        """Replace the matched fragment with a redaction mark.

        The sanitized text is what gets substituted into the rewrite
        template via the ``{sanitized_input}`` placeholder, so the model
        sees the original intent minus the offensive instruction.
        """
        start, end = hit.span
        return (
            content[:start]
            + _REDACTION_MARK
            + content[end:]
        )

    @staticmethod
    def _build_fired_rule(hit: _PatternHit) -> FiredRule:
        """Construct a :class:`FiredRule` from a :class:`_PatternHit`."""
        return FiredRule(
            mitigator=PromptMitigator.name,
            rule_name=hit.pattern_name,
            category=PROMPT_CATEGORY,
            strategy="injection_detected",
            matched_substring=hit.matched_substring[:_MAX_AUDIT_SUBSTRING_LEN],
        )

    @staticmethod
    def _log_result(
        fired: list[FiredRule],
        t0: float,
        strategy: str,
        modified: bool,
    ) -> None:
        """Emit a single structured INFO line per mitigator call.

        Could just emit at DEBUG and rely on the engine to fan-out audits,
        but having the per-call summary here makes ad-hoc operator triage
        faster without needing the engine's report.
        """
        _LOG.info(
            "PromptMitigator call: fired=%d, strategy=%s, modified=%s, "
            "duration_ms=%.2f",
            len(fired),
            strategy,
            modified,
            (time.perf_counter() - t0) * 1000.0,
        )


@dataclass(frozen=True, slots=True)
class _PatternHit:
    """First-window match result returned by :meth:`PromptMitigator._first_match`.

    Attributes:
        pattern_name: Name of the matched compiled pattern.
        matched_substring: The literal substring that triggered the match.
        span: ``(start, end)`` offsets within the original content.
    """

    pattern_name: str
    matched_substring: str
    span: tuple[int, int]


__all__ = ["PromptMitigator"]
