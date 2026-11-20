r"""Output-level mitigation: scan completions and apply policy actions.

This module implements :class:`OutputMitigator`, the second half of the
healer patch pipeline. It runs *after* the target LLM returns: every
assistant completion is scanned against the keyword tables in
``data/prompts/policy_keywords.yaml`` and, if any configured category
matches, the configured action is applied. After policy filtering, an
independent ``max_length`` guard trims overly-long responses (defending
against token-budget exhaustion probes that aren't covered by content
rules).

Strategy matrix (matches :class:`healer.config.OnMatch`)
---------------------------------------------------------
* ``fallback`` -> replace the entire completion with the canned
  :attr:`PolicyFilterConfig.fallback_message`. The whole text is dropped
  on a single category hit — we never partially-redact AND partially-fallback,
  because partial fallbacks leak content taxonomy hints.
* ``redact``   -> mask each offending keyword with ``[redacted:category]``
  in place; remaining content is preserved.
* ``log_only`` -> don't alter, just emit a :class:`FiredRule` for audit.

Tokenisation note
------------------
Keyword matching is **case-insensitive, word-boundary aware** via
:class:`re.Pattern` compiled with ``re.IGNORECASE`` and ``\\b`` anchors
on each keyword (where the keyword itself starts/ends with a word
character). This means ``"credit card number"`` matches the policy keyword
inside ``"your credit card number is..."`` but not inside ``"discredit
cardinal number theory"``.

Threading & caching
-------------------
Keywords are compiled once at construction; the mitigator is safe to
share across requests.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

from healer.config import OnMatch, OutputPatchConfig
from healer.logging_setup import get_logger
from healer.patch.base import (
    OUTPUT_CATEGORY,
    BaseMitigator,
    FiredRule,
)

_LOG: Final[logging.Logger] = get_logger(__name__)

#: Placeholder written in place of an offending keyword when ``redact`` fires.
_REDACTION_TEMPLATE: Final[str] = "[redacted:{category}]"

#: Default cap on the length of audit-captured substrings.
_MAX_AUDIT_SUBSTRING_LEN: Final[int] = 80

#: Length cap fall-back text emitted when truncation alone fires (regardless
#: of policy). Kept short so it can ride in any content slot.
_TRUNCATION_NOTICE: Final[str] = "\n[truncated by healer max_length policy]"


@dataclass(frozen=True, slots=True)
class _CompiledKeyword:
    """One keyword from a policy category compiled for fast scanning.

    Attributes:
        category: The YAML key the keyword lives under (``"pii"``, etc.).
        keyword: The original (un-escaped) source keyword, used in audits.
        pattern: Compiled regex with case-insensitive word-boundary anchors.
    """

    category: str
    keyword: str
    pattern: re.Pattern[str]


@dataclass(slots=True)
class _ScanOutcome:
    """Per-call mutable accumulator used during scanning.

    Captures every keyword hit so we can replay its strategy choice per
    category (each category contributes its own ``OnMatch`` value when, in
    future, the config grows per-category overrides; for now we share the
    single configured ``on_match`` across all enabled categories).
    """

    fired: list[FiredRule] = field(default_factory=list)
    redactions: list[tuple[str, str, str]] = field(default_factory=list)
    blocked_categories: set[str] = field(default_factory=set)


class OutputMitigator(BaseMitigator):
    """Output-side mitigation: scan completions and apply policy actions."""

    name: str = "OutputMitigator"

    def __init__(
        self,
        config: OutputPatchConfig,
        *,
        keywords_file: Path | None = None,
    ) -> None:
        """Initialise the mitigator and pre-compile its keyword table.

        Args:
            config: The :class:`OutputPatchConfig` block from
                :class:`healer.config.HealerConfig.patch.output_patch`.
            keywords_file: Optional explicit path to the keywords YAML.
                Defaults to the path declared inside ``config``.

        Raises:
            FileNotFoundError: The keywords file does not exist.
            ValueError: The keywords YAML is malformed.
        """
        self._config: OutputPatchConfig = config
        self._keywords: list[_CompiledKeyword] = self._load_keywords(
            keywords_file or config.strategies.policy_filter.keywords_file,
            allowed_categories=set(config.strategies.policy_filter.categories),
        )

        pf = config.strategies.policy_filter
        ml = config.strategies.max_length
        _LOG.info(
            "OutputMitigator initialised: %d keywords across %d categories "
            "(policy=%s, max_length=%s).",
            len(self._keywords),
            len(pf.categories),
            "enabled" if pf.enabled else "disabled",
            "enabled" if ml.enabled else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        """Output mitigators never edit inputs — identity return."""
        return [dict(m) for m in messages], False

    async def patch_output(self, response_text: str) -> tuple[str, bool]:
        """Apply policy filtering then length truncation, in that order.

        Args:
            response_text: Raw assistant completion text.

        Returns:
            ``(modified_text, was_modified)``. ``was_modified`` is True when
            either the policy filter or the length guard altered the text.
        """
        t0 = time.perf_counter()
        text = response_text
        modified = False
        fired: list[FiredRule] = []

        # --- policy filter ---
        pf = self._config.strategies.policy_filter
        if self._config.enabled and pf.enabled:
            text, fired, policy_modified = self._apply_policy(text)
            modified = modified or policy_modified

        # --- max_length ---
        ml = self._config.strategies.max_length
        if self._config.enabled and ml.enabled:
            text, length_modified = self._apply_max_length(text)
            modified = modified or length_modified

        self._log_result(fired, t0, modified)
        return text, modified

    # ------------------------------------------------------------------
    # Policy-filter internals
    # ------------------------------------------------------------------

    def _apply_policy(
        self, text: str
    ) -> tuple[str, list[FiredRule], bool]:
        """Scan ``text`` against enabled keywords and apply ``on_match``.

        Returns:
            ``(new_text, fired_rules, was_modified)``.
        """
        pf = self._config.strategies.policy_filter
        outcome = _ScanOutcome()

        for kw in self._keywords:
            matches = list(kw.pattern.finditer(text))
            if not matches:
                continue
            for m in matches:
                fired_rule = FiredRule(
                    mitigator=OutputMitigator.name,
                    rule_name=f"{kw.category}:{kw.keyword}",
                    category=f"{OUTPUT_CATEGORY}:{kw.category}",
                    strategy=pf.on_match.value,
                    matched_substring=m.group(0)[:_MAX_AUDIT_SUBSTRING_LEN],
                )
                outcome.fired.append(fired_rule)
                outcome.redactions.append(
                    (kw.category, kw.keyword, m.group(0))
                )
                outcome.blocked_categories.add(kw.category)
                _LOG.info(
                    "Policy hit: category=%s, keyword=%r, strategy=%s",
                    kw.category,
                    kw.keyword,
                    pf.on_match.value,
                )

        if not outcome.fired:
            return text, [], False

        if pf.on_match == OnMatch.FALLBACK:
            _LOG.warning(
                "Applying fallback replacement — %d rule(s) fired across %d "
                "category(ies).",
                len(outcome.fired),
                len(outcome.blocked_categories),
            )
            return pf.fallback_message, outcome.fired, True

        if pf.on_match == OnMatch.REDACT:
            new_text = text
            # Apply redactions in reverse order so earlier spans stay valid.
            for category, _keyword, original in reversed(outcome.redactions):
                replacement = _REDACTION_TEMPLATE.format(category=category)
                new_text = new_text.replace(original, replacement, 1)
            _LOG.info(
                "Redacted %d hit(s) across %d category(ies).",
                len(outcome.redactions),
                len(outcome.blocked_categories),
            )
            return new_text, outcome.fired, True

        if pf.on_match == OnMatch.LOG_ONLY:
            return text, outcome.fired, False

        # pragma: no cover - exhaustive enum guard
        _LOG.warning("Unknown OnMatch=%s; treating as LOG_ONLY.", pf.on_match)
        return text, outcome.fired, False

    def _apply_max_length(self, text: str) -> tuple[str, bool]:
        """Trim the response if its character length exceeds the configured cap.

        We use **character length** rather than token count: the proxy does
        not maintain a tokenizer (the upstream client does), so an exact
        token count would require either shipping or heuristically
        approximating one. Character length is a sufficient proxy for the
        proof-of-concept and the audit record clearly distinguishes the
        two.

        Returns:
            ``(new_text, was_modified)``.
        """
        ml = self._config.strategies.max_length
        if len(text) <= ml.max_tokens:
            return text, False
        _LOG.info(
            "Length-truncating response: %d -> %d chars.",
            len(text), ml.max_tokens,
        )
        return text[: ml.max_tokens] + _TRUNCATION_NOTICE, True

    # ------------------------------------------------------------------
    # Keyword loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_keywords(
        path: Path,
        *,
        allowed_categories: set[str],
    ) -> list[_CompiledKeyword]:
        """Parse the policy_keywords YAML and compile keyword patterns.

        Args:
            path: Location of the keywords file.
            allowed_categories: Set of categories the engine is configured
                to enforce. Keywords outside this set are skipped so we don't
                pay scanning cost for unmatched categories.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: YAML is malformed or missing category entries.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Policy keywords file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Keyword YAML parse error in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(  # noqa: TRY004 - isinstance-check on YAML structure
                f"Keyword YAML {path} must be a mapping of category -> keyword list."
            )

        compiled: list[_CompiledKeyword] = []
        for category, entries in raw.items():
            if not isinstance(category, str):
                raise ValueError(f"Category key {category!r} must be a string.")  # noqa: TRY004 - isinstance-check on YAML structure
            if allowed_categories and category not in allowed_categories:
                continue
            if not isinstance(entries, list):
                raise ValueError(  # noqa: TRY004 - isinstance-check on YAML structure
                    f"Category {category!r} value must be a list of strings."
                )
            for kw in entries:
                if not isinstance(kw, str) or not kw.strip():
                    continue
                pattern = re.compile(
                    OutputMitigator._word_boundary_regex(kw),
                    re.IGNORECASE,
                )
                compiled.append(
                    _CompiledKeyword(
                        category=category,
                        keyword=kw,
                        pattern=pattern,
                    )
                )
        return compiled

    @staticmethod
    def _word_boundary_regex(keyword: str) -> str:
        r"""Build a case-insensitive pattern for a keyword entry.

        The policy_keywords.yaml format is dual-mode:

        * If the entry contains regex metacharacters (``|``, ``(``, ``*``,
          ``?``, ``+``, ``[``, ``\\``), it is treated as a **raw regex** and
          used verbatim, with optional ``\\b`` anchors when the first/last
          characters are plain literal alphanumerics.

        * Otherwise, it is treated as a **plain phrase**: escaped with
          :func:`re.escape` and wrapped with ``\\b`` word-boundary anchors
          on its first/last alnum chars, so that ``"credit card number"``
          matches the literal phrase but not ``"discredit cardinal number
          theory"``.

        The dual-mode parse keeps the YAML readable for policy analysts
        (phrase-mode for 95% of entries) while still allowing the rare
        regex entry where alternation or character classes are genuinely
        needed (e.g., ``"self[- ]harm"``).
        """
        # Regex metachar set forces regex-mode interpretation.
        contains_regex_meta = any(c in keyword for c in "|()[]?*+\\")
        keyword_stripped = keyword.strip()

        if contains_regex_meta:
            left = r"\b" if OutputMitigator._is_word_char(keyword_stripped[0]) else ""
            right = (
                r"\b"
                if OutputMitigator._is_word_char(keyword_stripped[-1])
                else ""
            )
            return f"{left}{keyword_stripped}{right}"

        escaped = re.escape(keyword_stripped)
        left = r"\b" if OutputMitigator._is_word_char(keyword_stripped[0]) else ""
        right = (
            r"\b"
            if OutputMitigator._is_word_char(keyword_stripped[-1])
            else ""
        )
        return f"{left}{escaped}{right}"

    @staticmethod
    def _is_word_char(c: str) -> bool:
        """Return True if ``c`` is a word character (alnum or underscore)."""
        return bool(c) and (c.isalnum() or c == "_")

    # ------------------------------------------------------------------
    # Audit & logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_result(
        fired: list[FiredRule],
        t0: float,
        modified: bool,
    ) -> None:
        """Emit a single structured INFO line per mitigator call."""
        _LOG.info(
            "OutputMitigator call: fired=%d, modified=%s, duration_ms=%.2f",
            len(fired),
            modified,
            (time.perf_counter() - t0) * 1000.0,
        )


__all__ = ["OutputMitigator"]
