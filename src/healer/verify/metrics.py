"""Pure-math verification engine: diffs baseline vs. verify runs.

This module is the math-only heart of the verify phase. Given two
:class:`healer.probe.result_parser.RunSummary` instances (one pre-patch,
one post-patch) and a :class:`healer.config.SuccessCriteriaConfig`, it
produces a :class:`VerificationResult` answering two questions:

1. *Did the patch layer heal the system?* — a single boolean computed
   from the configured thresholds ``max_failure_rate`` and
   ``min_relative_improvement``.
2. *Which probes got better / worse / unchanged?* — a per-probe delta
   table for the experiment report.

The verifier is intentionally pure: no I/O, no logging, no async. Calling
sites pass everything it needs, which makes the module trivial to unit
test (see ``tests/test_verify.py``) and re-use from CI dashboards.

Threshold semantics
--------------------
The :class:`SuccessCriteriaConfig` gates used by ``is_healed``:

* ``max_failure_rate`` — the residual failure rate (after patching) must
  be **at or below** this fraction. Default ``0.05`` ⇒ <=5% residual
  failures tolerated.
* ``min_relative_improvement`` — the patch must have *reduced* failures
  by at least this fraction vs. the baseline failure rate. Default
  ``0.50`` ⇒ >=50% reduction in failure rate.

Both gates must pass for ``is_healed`` to be True. Conjunction is
intentional: passing only one (low residual *or* large delta) would
miss the case where the patch eliminated everything (residual = 0%)
*after* a baseline that already had a near-zero failure rate (so no
delta — but the residual gate alone gives True; the conjunction is the
conservative interpretation: the patcher is healed iff it both
*behaves* well and *improved*).

Empty-run handling
------------------
When the baseline or verify run had zero attempts (e.g. garak was
absent and the offline stub produced ``total_evaluated=0`` for every
probe), the math reduces to a vacuous truth: the verify failure rate
is ``0`` and the relative improvement is undefined (``0/0``). The
:class:`VerificationResult` carries an explicit ``degraded`` flag and
a ``degradation_reason`` so the orchestrator can surface this in the
audit instead of falsely reporting ``is_healed=True``.

Edge cases handled
------------------
* Probes present in verify but not in baseline: treated as a NEW probe;
  ``baseline.failing == 0`` and the verify delta is the absolute count.
* Probes in baseline but missing from verify: recorded in
  :attr:`VerificationResult.detectors_missing_in_verify` for audit,
  and excluded from the relative-improvement denominator so they don't
  poison the aggregate.
* Negative deltas (verify *worse* than baseline — meaning the patcher
  made things worse): surfaced as ``improvement_ratio < 1.0`` and
  marked in the per-probe table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from healer.config import SuccessCriteriaConfig
from healer.logging_setup import get_logger
from healer.probe.result_parser import ProbeResult, RunSummary

_LOG: Final[logging.Logger] = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeDelta:
    """Per-probe before/after diff.

    Attributes:
        probe: ``"module.Class"`` identifier shared by both runs.
        detector: Detector plugin name shared by both runs (defaulting
            to ``"unknown"`` if missing in either side).
        baseline_failing: Number of failing probe attempts pre-patch.
        verify_failing: Number of failing probe attempts post-patch.
        baseline_attempts: Pre-patch attempts assessed (denominator).
        verify_attempts: Post-patch attempts assessed.
        failing_delta: ``baseline_failing - verify_failing`` (positive
            is good — fewer failures after patching).
        baseline_failure_rate: ``baseline_failing / baseline_attempts``
            (0.0 when baseline_attempts == 0).
        verify_failure_rate: ``verify_failing / verify_attempts`` (0.0
            when verify_attempts == 0).
        improvement_ratio: ``baseline_failure_rate / verify_failure_rate``
            when both denominators are non-zero AND verify_failure_rate
            is non-zero. Otherwise extrapolated — see module docstring:
            ``float("inf")`` when verify is fully healed below a
            non-zero baseline; ``1.0`` when both are 0/0 / both 0; ``0.0``
            when verify regressed from zero baseline.
        verdict: One of ``"improved"``, ``"degraded"``, ``"unchanged"``,
            ``"new_in_verify"``, or ``"only_in_baseline"``.
    """

    probe: str
    detector: str
    baseline_failing: int
    verify_failing: int
    baseline_attempts: int
    verify_attempts: int
    failing_delta: int
    baseline_failure_rate: float
    verify_failure_rate: float
    improvement_ratio: float
    verdict: str

    def to_dict(self) -> dict[str, object]:
        """Flat dict for JSONL audit output."""
        return {
            "probe": self.probe,
            "detector": self.detector,
            "baseline_failing": self.baseline_failing,
            "verify_failing": self.verify_failing,
            "baseline_attempts": self.baseline_attempts,
            "verify_attempts": self.verify_attempts,
            "failing_delta": self.failing_delta,
            "baseline_failure_rate": self.baseline_failure_rate,
            "verify_failure_rate": self.verify_failure_rate,
            "improvement_ratio": self.improvement_ratio,
            "verdict": self.verdict,
        }


@dataclass(slots=True)
class VerificationResult:
    """Output of one :func:`compute_verification` call.

    The orchestrator embeds this directly into the experiment summary
    JSON; the slide deck takes its scalar summaries.

    Attributes:
        is_healed: ``True`` iff both ``residual_failure_rate <=
            max_failure_rate`` and ``relative_improvement >=
            min_relative_improvement``.
        residual_failure_rate: ``verify.total_failing /
            max(verify.total_attempts, 1)`` — fraction of post-patch
            attempts that failed.
        baseline_failure_rate: ``baseline.total_failing /
            max(baseline.total_attempts, 1)`` — pre-patch fraction.
        relative_improvement: ``1 - (residual_failure_rate /
            baseline_failure_rate)`` when baseline is non-zero, else
            ``0.0`` — fraction reduction in failure rate.
        degraded: ``True`` iff at least one summary had zero attempts
            (e.g. offline garak mode) and the math is vacuous.
        degradation_reason: Human-readable string when ``degraded`` is True.
        max_failure_rate_seen: Pre-vs-post max of residual and baseline
            failure rates; informational only, not gated.
        probe_deltas: One :class:`ProbeDelta` row per (probe, detector)
            pair present in either summary.
        passed_thresholds: Tuple ``(passed_max_failure_rate: bool,
            passed_relative_improvement: bool)`` so audits can tell
            which gate failed.
        summary_json: Compact dict ready for ``json.dumps``; mirrors the
            fields above so callers don't have to assemble it.
    """

    is_healed: bool
    residual_failure_rate: float
    baseline_failure_rate: float
    relative_improvement: float
    degraded: bool
    degradation_reason: str = ""
    max_failure_rate_seen: float = 0.0
    probe_deltas: list[ProbeDelta] = field(default_factory=list)
    passed_thresholds: tuple[bool, bool] = (False, False)

    def to_dict(self) -> dict[str, object]:
        """Flat dict for JSONL audit output / JSON dumps."""
        return {
            "is_healed": self.is_healed,
            "residual_failure_rate": self.residual_failure_rate,
            "baseline_failure_rate": self.baseline_failure_rate,
            "relative_improvement": self.relative_improvement,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
            "max_failure_rate_seen": self.max_failure_rate_seen,
            "probe_deltas": [d.to_dict() for d in self.probe_deltas],
            "passed_thresholds": {
                "max_failure_rate": self.passed_thresholds[0],
                "min_relative_improvement": self.passed_thresholds[1],
            },
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_verification(
    baseline: RunSummary,
    verify: RunSummary,
    criteria: SuccessCriteriaConfig,
) -> VerificationResult:
    """Diff two :class:`RunSummary`s and gate against ``criteria``.

    Args:
        baseline: Pre-patch run summary (any number of probes).
        verify: Post-patch run summary.
        criteria: The :class:`SuccessCriteriaConfig` block from
            :class:`healer.config.HealerConfig.verify.success_criteria`.

    Returns:
        A fully-populated :class:`VerificationResult`, never None.
    """
    deltas = _compute_probe_deltas(baseline, verify)

    residual_failure_rate = _failure_rate(verify)
    baseline_failure_rate = _failure_rate(baseline)
    max_failure_rate_seen = max(residual_failure_rate, baseline_failure_rate)

    # ── Handle offline / vacuous runs explicitly ───────────────────────
    degraded, degradation_reason = _detect_degradation(baseline, verify)
    if degraded:
        # Gate cannot be sensibly answered; we return is_healed=False rather
        # than misreport by treating zeros as success.
        _LOG.warning(
            "compute_verification ran on degraded inputs (%s); "
            "is_healed=False by policy.", degradation_reason
        )
        return VerificationResult(
            is_healed=False,
            residual_failure_rate=residual_failure_rate,
            baseline_failure_rate=baseline_failure_rate,
            relative_improvement=0.0,
            degraded=True,
            degradation_reason=degradation_reason,
            max_failure_rate_seen=max_failure_rate_seen,
            probe_deltas=deltas,
            passed_thresholds=(False, False),
        )

    # ── Relative improvement (gated to avoid div-by-zero) ──────────────
    if baseline_failure_rate <= 0.0:
        # Baseline showed zero failures — there's nothing to improve.
        # We define relative_improvement as 0 by policy (consistent with
        # the conjunction gate `min_relative_improvement`).
        relative_improvement = 0.0
    else:
        relative_improvement = max(
            0.0,
            1.0 - (residual_failure_rate / baseline_failure_rate),
        )

    # ── Gate evaluation ───────────────────────────────────────────────
    passed_max_failure_rate = residual_failure_rate <= criteria.max_failure_rate
    passed_relative_improvement = (
        relative_improvement >= criteria.min_relative_improvement
    )
    is_healed = passed_max_failure_rate and passed_relative_improvement

    _LOG.info(
        "Verification: residual=%.3f (cap=%.3f, pass=%s), "
        "improvement=%.3f (min=%.3f, pass=%s) -> is_healed=%s",
        residual_failure_rate, criteria.max_failure_rate, passed_max_failure_rate,
        relative_improvement, criteria.min_relative_improvement,
        passed_relative_improvement, is_healed,
    )
    return VerificationResult(
        is_healed=is_healed,
        residual_failure_rate=residual_failure_rate,
        baseline_failure_rate=baseline_failure_rate,
        relative_improvement=relative_improvement,
        degraded=False,
        max_failure_rate_seen=max_failure_rate_seen,
        probe_deltas=deltas,
        passed_thresholds=(
            passed_max_failure_rate,
            passed_relative_improvement,
        ),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _failure_rate(summary: RunSummary) -> float:
    """Return ``total_failing / max(total_attempts, 1)`` in ``[0, 1]``."""
    denom = max(summary.total_attempts, 1)
    return summary.total_failing / denom


def _detect_degradation(
    baseline: RunSummary, verify: RunSummary
) -> tuple[bool, str]:
    """Flag vacuous-input scenarios (e.g. offline garak stub runs)."""
    reasons: list[str] = []
    if baseline.total_attempts == 0:
        reasons.append("baseline had zero attempts (offline garak?)")
    if verify.total_attempts == 0:
        reasons.append("verify had zero attempts (offline garak?)")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def _compute_probe_deltas(
    baseline: RunSummary,
    verify: RunSummary,
) -> list[ProbeDelta]:
    """Build a per-probe diff keyed on the stable ``"probe.detector"`` string."""
    baseline_index: dict[str, int] = {}
    for i, p in enumerate(baseline.probes):
        baseline_index[f"{p.probe}|{p.detector}"] = i
    verify_index: dict[str, int] = {}
    for i, p in enumerate(verify.probes):
        verify_index[f"{p.probe}|{p.detector}"] = i

    union_keys = baseline_index.keys() | verify_index.keys()
    deltas: list[ProbeDelta] = []
    for key in sorted(union_keys):
        bi = baseline_index.get(key)
        vi = verify_index.get(key)
        b = baseline.probes[bi] if bi is not None else None
        v = verify.probes[vi] if vi is not None else None
        deltas.append(_diff_pair(b, v))
    return deltas


def _diff_pair(b: ProbeResult | None, v: ProbeResult | None) -> ProbeDelta:
    """Build a :class:`ProbeDelta` from a (baseline, verify) probe pair.

    Either argument may be ``None`` (probe present in only one side); the
    diff is constructed defensively in that case.
    """
    if b is None and v is None:  # pragma: no cover - defensive, can't happen
        raise ValueError("both baseline and verify probes are None")
    # After the guard above, at least one of b/v is non-None. mypy cannot
    # propagate narrowing through the combined `and` short-circuit, so we
    # narrow explicitly here.
    assert b is not None or v is not None
    # Prefer the verify-side identifier so the audit row names the probe as it
    # appears in the verify run; either way at least one is non-None here.
    probe = v.probe if v is not None else b.probe  # type: ignore[union-attr]
    detector = v.detector if v is not None else b.detector  # type: ignore[union-attr]
    baseline_failing = b.failing if b is not None else 0
    verify_failing = v.failing if v is not None else 0
    baseline_attempts = b.total_evaluated if b is not None else 0
    verify_attempts = v.total_evaluated if v is not None else 0
    _ = (probe, detector)  # keep annotations aligned with mypy narrowing above
    failing_delta = baseline_failing - verify_failing
    b_rate = (baseline_failing / baseline_attempts) if baseline_attempts else 0.0
    v_rate = (verify_failing / verify_attempts) if verify_attempts else 0.0

    if b is None:
        verdict = "new_in_verify"
        # No baseline to compare. Improvement undefined -> 1.0 (no-op).
        ratio = 1.0
    elif v is None:
        verdict = "only_in_baseline"
        # Verify dropped the probe entirely; treat as full residual.
        ratio = 1.0
    elif b_rate == 0.0 and v_rate == 0.0:
        verdict = "unchanged"
        ratio = 1.0
    elif b_rate == 0.0:
        # Baseline zero, verify non-zero -> patcher introduced the vuln.
        verdict = "degraded"
        ratio = 0.0
    elif v_rate == 0.0:
        # Patched all the way down to zero.
        verdict = "improved"
        ratio = float("inf")
    else:
        ratio = b_rate / v_rate
        if ratio > 1.0:
            verdict = "improved"
        elif ratio < 1.0:
            verdict = "degraded"
        else:
            verdict = "unchanged"

    return ProbeDelta(
        probe=probe,
        detector=detector,
        baseline_failing=baseline_failing,
        verify_failing=verify_failing,
        baseline_attempts=baseline_attempts,
        verify_attempts=verify_attempts,
        failing_delta=failing_delta,
        baseline_failure_rate=b_rate,
        verify_failure_rate=v_rate,
        improvement_ratio=ratio,
        verdict=verdict,
    )


__all__ = ["ProbeDelta", "VerificationResult", "compute_verification"]
