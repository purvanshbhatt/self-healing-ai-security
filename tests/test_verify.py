"""Tests for the verify layer: :mod:`healer.verify.metrics`.

Coverage matrix (21 behaviour contracts from the module spec):
* :func:`compute_verification` — top-level gating
  - Happy path: both gates pass -> ``is_healed=True``.
  - Only residual gate passes (improvement too low) -> ``is_healed=False``.
  - Only improvement gate passes (residual too high) -> ``is_healed=False``.
  - Both gates fail -> ``is_healed=False``.
  - Degraded inputs (zero attempts in baseline or verify) -> ``degraded=True`` and
    is_healed=False (do not misreport zeros as success).
  - Empty summaries on both sides -> degraded, is_healed=False.
  - Baseline failure rate = 0 (nothing to improve) -> ``relative_improvement=0.0``.
  - ``residual_failure_rate`` uses ``max(total_attempts, 1)`` (no ZeroDivisionError).
* :class:`VerificationResult` math computations
  - ``residual_failure_rate`` global formula.
  - ``baseline_failure_rate`` global formula.
  - ``relative_improvement = 1 - residual/baseline`` (clamped at >=0).
  - ``relative_improvement=0`` when baseline_failure_rate=0.
  - ``max_failure_rate_seen`` = max of residual/baseline (the per-summary rates,
    not per-probe; informational field).
  - ``passed_thresholds`` reports which gate passed individually.
* :class:`ProbeDelta` per-probe diff
  - Union of (probe, detector) keys present in either summary.
  - Sorted output for deterministic audit artefacts.
  - ``new_in_verify`` (baseline missing) -> ratio=1.0.
  - ``only_in_baseline`` (verify missing) -> ratio=1.0.
  - Both rates zero -> ``unchanged``, ratio=1.0.
  - Baseline rate 0, verify >0 -> ``degraded``, ratio=0.0.
  - Verify rate 0 from non-zero baseline -> ``improved``, ratio=+inf.
  - ``b_rate/v_rate > 1.0`` -> ``improved`` with exact ratio.
  - ``b_rate/v_rate < 1.0`` -> ``degraded`` with exact ratio.
  - ``failing_delta = baseline_failing - verify_failing``.
* Serialisation
  - :meth:`VerificationResult.to_dict` is JSON-safe (json.dumps succeeds).
  - :meth:`ProbeDelta.to_dict` exposes the documented key set.
"""

from __future__ import annotations

import json
import math

import pytest

from healer.config import SuccessCriteriaConfig
from healer.probe.result_parser import ProbeResult, RunSummary
from healer.verify.metrics import (
    ProbeDelta,
    VerificationResult,
    compute_verification,
)

# ---------------------------------------------------------------------------
# Builders — construct RunSummary directly to keep tests I/O-free.
# ---------------------------------------------------------------------------


def _probe(name: str, passed: int, total: int, detector: str = "always.Pass") -> ProbeResult:
    """Build a ProbeResult the way :func:`parse_report` would."""
    pass_rate = (passed / total) if total else 0.0
    return ProbeResult(
        probe=name,
        detector=detector,
        passed=passed,
        total_evaluated=total,
        pass_rate=pass_rate,
        failing=total - passed,
    )


def _summary(*probes: ProbeResult) -> RunSummary:
    """Assemble an aggregated RunSummary without going through file I/O."""
    s = RunSummary()
    s.probes = list(probes)
    s.aggregate()
    return s


def _criteria(
    max_failure_rate: float = 0.05,
    min_relative_improvement: float = 0.50,
) -> SuccessCriteriaConfig:
    """Build a SuccessCriteriaConfig with explicit thresholds (bypasses pydantic
    model validator chain by using model_construct when needed).
    """
    return SuccessCriteriaConfig(
        max_failure_rate=max_failure_rate,
        min_relative_improvement=min_relative_improvement,
    )


# ===========================================================================
# compute_verification — gating logic
# ===========================================================================


def test_happy_path_both_gates_pass_is_healed_true() -> None:
    """Contract 1 & 2: clear improvement of >50% and residual <=5% -> healed."""
    baseline = _summary(_probe("dan.DanInTheWild", 0, 100))   # 100% fail
    verify = _summary(_probe("dan.DanInTheWild", 96, 100))   #  4% fail
    # relative_improvement = 1 - 0.04/1.0 = 0.96 (>=0.50)
    # residual = 0.04 (<=0.05)
    result = compute_verification(baseline, verify, _criteria())
    assert result.is_healed is True
    assert result.degraded is False
    assert result.passed_thresholds == (True, True)


def test_only_residual_gate_passes_yields_false() -> None:
    """Contract 3: residual passes, improvement too low -> is_healed=False."""
    baseline = _summary(_probe("p.A", 94, 100))  # 6% fail
    verify = _summary(_probe("p.A", 96, 100))    # 4% fail
    # relative_improvement = 1 - 0.04/0.06 = 0.333... (<0.50)
    result = compute_verification(baseline, verify, _criteria())
    assert result.residual_failure_rate == pytest.approx(0.04)
    assert result.relative_improvement == pytest.approx(1 - 0.04 / 0.06)
    assert result.passed_thresholds == (True, False)
    assert result.is_healed is False


def test_only_improvement_gate_passes_yields_false() -> None:
    """Contract 4: improvement passes, residual too high -> is_healed=False."""
    baseline = _summary(_probe("p.A", 0, 100))   # 100% fail
    verify = _summary(_probe("p.A", 80, 100))    #  20% fail
    # relative_improvement = 1 - 0.20/1.0 = 0.80 (>=0.50)
    # residual = 0.20 (>0.05)
    result = compute_verification(baseline, verify, _criteria())
    assert result.relative_improvement == pytest.approx(0.80)
    assert result.passed_thresholds == (False, True)
    assert result.is_healed is False


def test_both_gates_fail_yields_false() -> None:
    """Tight residual cap and high improvement bar both missing."""
    baseline = _summary(_probe("p.A", 0, 100))   # 100% fail
    verify = _summary(_probe("p.A", 40, 100))    #  60% fail
    # improvement = 1 - 0.6/1.0 = 0.4 (<0.50 fail); residual = 0.6 (>0.05 fail)
    result = compute_verification(baseline, verify, _criteria())
    assert result.passed_thresholds == (False, False)
    assert result.is_healed is False


# ---------------------------------------------------------------------------
# Degraded inputs — vacuous truth must NOT be reported as healed
# ---------------------------------------------------------------------------


def test_degraded_baseline_zero_attempts() -> None:
    """Contract 6: baseline had zero attempts -> degraded, is_healed=False."""
    baseline = _summary()  # empty
    verify = _summary(_probe("p.A", 10, 10))
    result = compute_verification(baseline, verify, _criteria())
    assert result.degraded is True
    assert result.is_healed is False
    assert "baseline" in result.degradation_reason


def test_degraded_verify_zero_attempts() -> None:
    """Contract 7: verify had zero attempts -> degraded, is_healed=False."""
    baseline = _summary(_probe("p.A", 10, 10))
    verify = _summary()  # empty
    result = compute_verification(baseline, verify, _criteria())
    assert result.degraded is True
    assert result.is_healed is False
    assert "verify" in result.degradation_reason


def test_degraded_both_zero_attempts() -> None:
    """Contracts 5: both summaries empty -> degraded, is_healed=False."""
    result = compute_verification(_summary(), _summary(), _criteria())
    assert result.degraded is True
    assert result.is_healed is False
    assert result.passed_thresholds == (False, False)


def test_degraded_returns_relative_improvement_zero() -> None:
    """When degraded, the result short-circuits with relative_improvement=0."""
    result = compute_verification(_summary(), _summary(), _criteria())
    assert result.relative_improvement == 0.0


def test_offline_stub_zero_per_probe_is_degraded() -> None:
    """Realistic offline garak stub: probes present but total_evaluated=0 each."""
    baseline = _summary(
        _probe("probes.promptinject", 0, 0),
        _probe("probes.encoding", 0, 0),
    )
    verify = _summary(_probe("probes.promptinject", 0, 0))
    result = compute_verification(baseline, verify, _criteria())
    assert result.degraded is True
    assert result.is_healed is False


# ===========================================================================
# Math contracts on global failure rates
# ===========================================================================


def test_residual_failure_rate_formula() -> None:
    """``residual = total_failing / max(total_attempts, 1)``."""
    verify = _summary(
        _probe("p.A", 96, 100),  # 4 fail
        _probe("p.B", 90, 100),  # 10 fail
    )  # total 14 fail / 200 attempts = 0.07
    baseline = _summary(_probe("p.A", 0, 1))  # any non-empty baseline to skip degraded
    result = compute_verification(baseline, verify, _criteria())
    assert result.residual_failure_rate == pytest.approx(14 / 200)


def test_baseline_failure_rate_formula() -> None:
    """``baseline_failure_rate = baseline.total_failing / max(total_attempts, 1)``."""
    baseline = _summary(_probe("p.A", 80, 100))  # 20 fail -> 0.20
    verify = _summary(_probe("p.A", 100, 100))   # 0 fail -> 0
    result = compute_verification(baseline, verify, _criteria())
    assert result.baseline_failure_rate == pytest.approx(0.20)


def test_relative_improvement_formula() -> None:
    """``relative_improvement = max(0, 1 - residual/baseline)``."""
    baseline = _summary(_probe("p.A", 50, 100))  # 0.50 fail rate
    verify = _summary(_probe("p.A", 90, 100))    # 0.10 fail rate
    # expected: 1 - 0.10/0.50 = 0.80
    result = compute_verification(baseline, verify, _criteria())
    assert result.relative_improvement == pytest.approx(0.80)


def test_relative_improvement_zero_when_baseline_failure_rate_is_zero() -> None:
    """Contract 8: nothing to improve when baseline had no failures."""
    baseline = _summary(_probe("p.A", 100, 100))  # 0 fail
    verify = _summary(_probe("p.A", 100, 100))    # 0 fail
    result = compute_verification(baseline, verify, _criteria())
    assert result.baseline_failure_rate == 0.0
    assert result.relative_improvement == 0.0


def test_relative_improvement_clamped_at_zero_when_regressed() -> None:
    """``max(0, ...)`` clamp prevents negative improvement."""
    baseline = _summary(_probe("p.A", 100, 100))  # 0% fail
    verify = _summary(_probe("p.A", 50, 100))     # 50% fail (regressed)
    # Not degraded (attempts>0 on both sides) so we don't take the early-return
    # branch; the residual/baseline division would be 0.50/0 -> handled by the
    # baseline==0 shortcut that returns 0.
    result = compute_verification(baseline, verify, _criteria())
    assert result.relative_improvement == 0.0


def test_residual_failure_rate_safe_when_verify_zero_attempts() -> None:
    """Contract 10: degraded result still computes residual via max(.,1) -> 0."""
    baseline = _summary(_probe("p.A", 50, 100))
    verify = _summary()  # zero attempts
    result = compute_verification(baseline, verify, _criteria())
    assert result.residual_failure_rate == 0.0
    assert result.degraded is True


def test_max_failure_rate_seen_is_max_of_summary_rates() -> None:
    """Contract 11: informational max-of-the-two-summary failure rates."""
    baseline = _summary(_probe("p.A", 80, 100))  # 0.20
    verify = _summary(_probe("p.A", 100, 100))   # 0.00
    result = compute_verification(baseline, verify, _criteria())
    assert result.max_failure_rate_seen == pytest.approx(0.20)
    # flipped: baseline cleaner than verify
    result2 = compute_verification(verify, baseline, _criteria())
    assert result2.max_failure_rate_seen == pytest.approx(0.20)


# ===========================================================================
# ProbeDelta — per-probe diff verdicts
# ===========================================================================


def _deltas_by_key(result: VerificationResult) -> dict[str, ProbeDelta]:
    return {f"{d.probe}|{d.detector}": d for d in result.probe_deltas}


def test_probe_delta_union_of_keys() -> None:
    """Deltas span the union of (probe, detector) keys in either summary."""
    baseline = _summary(_probe("a.A", 5, 10), _probe("b.B", 5, 10))
    verify = _summary(_probe("b.B", 9, 10), _probe("c.C", 5, 10))
    result = compute_verification(baseline, verify, _criteria())
    keys = set(_deltas_by_key(result))
    assert keys == {"a.A|always.Pass", "b.B|always.Pass", "c.C|always.Pass"}


def test_probe_delta_sorted_alphabetically() -> None:
    """Contract 21: deltas are sorted by their union key for stable audits."""
    baseline = _summary(_probe("Z.Z", 5, 10), _probe("A.A", 5, 10))
    verify = _summary(_probe("M.M", 5, 10), _probe("A.A", 9, 10))
    result = compute_verification(baseline, verify, _criteria())
    keys = [f"{d.probe}|{d.detector}" for d in result.probe_deltas]
    assert keys == ["A.A|always.Pass", "M.M|always.Pass", "Z.Z|always.Pass"]


def test_probe_delta_new_in_verify_ratio_one() -> None:
    """Contract 14: probe only in verify -> new_in_verify, ratio=1.0."""
    baseline = _summary(_probe("shared", 5, 10))
    verify = _summary(_probe("shared", 5, 10), _probe("newonly", 5, 10))
    result = compute_verification(baseline, verify, _criteria())
    by_key = _deltas_by_key(result)
    d = by_key["newonly|always.Pass"]
    assert d.verdict == "new_in_verify"
    assert d.improvement_ratio == 1.0
    assert d.baseline_failing == 0
    assert d.baseline_attempts == 0


def test_probe_delta_only_in_baseline_ratio_one() -> None:
    """Contract 15: probe only in baseline -> only_in_baseline, ratio=1.0."""
    baseline = _summary(_probe("shared", 5, 10), _probe("gonemissing", 5, 10))
    verify = _summary(_probe("shared", 5, 10))
    result = compute_verification(baseline, verify, _criteria())
    by_key = _deltas_by_key(result)
    d = by_key["gonemissing|always.Pass"]
    assert d.verdict == "only_in_baseline"
    assert d.improvement_ratio == 1.0
    assert d.verify_failing == 0
    assert d.verify_attempts == 0


def test_probe_delta_unchanged_when_both_zero() -> None:
    """Contract 16: both rates zero -> unchanged, ratio=1.0."""
    baseline = _summary(_probe("p.A", 10, 10))
    verify = _summary(_probe("p.A", 10, 10))
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.verdict == "unchanged"
    assert d.improvement_ratio == 1.0
    assert d.failing_delta == 0


def test_probe_delta_degraded_when_baseline_zero_verify_nonzero() -> None:
    """Contract 17: baseline 0 fail, verify >0 fail -> degraded, ratio=0.0."""
    baseline = _summary(_probe("p.A", 10, 10))  # 0 fail
    verify = _summary(_probe("p.A", 5, 10))    # 5 fail
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.verdict == "degraded"
    assert d.improvement_ratio == 0.0
    assert d.failing_delta == -5  # patcher made things worse (5 - 10)


def test_probe_delta_improved_when_verify_zero_from_nonzero_baseline() -> None:
    """Contract 18: verify 0 fail from non-zero baseline -> improved, ratio=+inf."""
    baseline = _summary(_probe("p.A", 5, 10))   # 5 fail
    verify = _summary(_probe("p.A", 10, 10))    # 0 fail
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.verdict == "improved"
    assert math.isinf(d.improvement_ratio)
    assert d.improvement_ratio > 0
    assert d.failing_delta == 5  # 5 - 0


def test_probe_delta_improved_with_exact_ratio_above_one() -> None:
    """Contract 19: verify strictly better -> improved, ratio = b_rate/v_rate."""
    # baseline 5/10 = 0.5 fail; verify 25/100 = 0.25 fail -> ratio 2.0
    baseline = _summary(_probe("p.A", 5, 10))
    verify = _summary(_probe("p.A", 75, 100))
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.improvement_ratio == pytest.approx(0.5 / 0.25)
    assert d.verdict == "improved"


def test_probe_delta_degraded_with_exact_ratio_below_one() -> None:
    """Contract 20: verify strictly worse -> degraded, ratio = b_rate/v_rate."""
    # baseline 2/10 = 0.2 fail; verify 4/10 = 0.4 fail -> ratio 0.5
    baseline = _summary(_probe("p.A", 8, 10))
    verify = _summary(_probe("p.A", 6, 10))
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.improvement_ratio == pytest.approx(0.2 / 0.4)
    assert d.verdict == "degraded"


def test_probe_delta_failing_delta_field() -> None:
    """``failing_delta = baseline_failing - verify_failing`` (positive is good)."""
    baseline = _summary(_probe("p.A", 0, 100))   # 100 fail
    verify = _summary(_probe("p.A", 90, 100))    #  10 fail
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0]
    assert d.failing_delta == 90
    assert d.baseline_failing == 100
    assert d.verify_failing == 10


def test_probe_delta_distinct_detectors_produce_distinct_rows() -> None:
    """The same probe name with different detectors stays separate rows."""
    baseline = _summary(
        _probe("p.A", 0, 10, detector="d1"),
        _probe("p.A", 0, 10, detector="d2"),
    )
    verify = _summary(
        _probe("p.A", 10, 10, detector="d1"),
        _probe("p.A", 10, 10, detector="d2"),
    )
    result = compute_verification(baseline, verify, _criteria())
    assert len(result.probe_deltas) == 2
    keys = {f"{d.probe}|{d.detector}" for d in result.probe_deltas}
    assert keys == {"p.A|d1", "p.A|d2"}


# ===========================================================================
# Serialisation contracts
# ===========================================================================


def test_verification_result_to_dict_is_json_safe() -> None:
    """Contract 12: to_dict() output round-trips through json.dumps."""
    baseline = _summary(_probe("p.A", 5, 10))
    verify = _summary(_probe("p.A", 10, 10))
    result = compute_verification(baseline, verify, _criteria())
    payload = result.to_dict()
    # Must not raise; +inf in improvement_ratio is emitted as `Infinity` by the
    # stdlib json encoder (valid per ES5; downstream reader can parse it).
    s = json.dumps(payload)
    decoded = json.loads(s)
    assert decoded["is_healed"] is True
    assert "probe_deltas" in decoded
    assert isinstance(decoded["probe_deltas"], list)
    assert "passed_thresholds" in decoded
    assert decoded["passed_thresholds"] == {
        "max_failure_rate": True,
        "min_relative_improvement": True,
    }


def test_verification_result_to_dict_degraded_reason_present() -> None:
    """to_dict() surfaces the degradation reason string for audits."""
    result = compute_verification(_summary(), _summary(), _criteria())
    payload = result.to_dict()
    assert payload["degraded"] is True
    assert payload["degradation_reason"]


def test_probe_delta_to_dict_shape() -> None:
    """Contract 13: ProbeDelta.to_dict exposes the documented key set."""
    baseline = _summary(_probe("p.A", 4, 10))   # 6 fail
    verify = _summary(_probe("p.A", 9, 10))     # 1 fail
    result = compute_verification(baseline, verify, _criteria())
    d = result.probe_deltas[0].to_dict()
    assert set(d.keys()) == {
        "probe",
        "detector",
        "baseline_failing",
        "verify_failing",
        "baseline_attempts",
        "verify_attempts",
        "failing_delta",
        "baseline_failure_rate",
        "verify_failure_rate",
        "improvement_ratio",
        "verdict",
    }
    assert d["verdict"] == "improved"


def test_inf_ratio_serialises_via_json_dumps() -> None:
    """``improvement_ratio = +inf`` does not crash json.dumps (stdlib allows it)."""
    baseline = _summary(_probe("p.A", 0, 10))   # 10 fail
    verify = _summary(_probe("p.A", 10, 10))    #  0 fail (full heal)
    result = compute_verification(baseline, verify, _criteria())
    payload = result.to_dict()
    # Standard json emits +Infinity (not raising) when allow_nan=True (default).
    s = json.dumps(payload)
    assert "Infinity" in s  # sanity check the serialised form


# ===========================================================================
# is_healed boundary conditions
# ===========================================================================


def test_is_healed_true_at_exact_threshold_residual() -> None:
    """``<=`` semantics: residual exactly equal to max_failure_rate passes."""
    baseline = _summary(_probe("p.A", 0, 100))   # 1.0
    verify = _summary(_probe("p.A", 95, 100))   # 0.05 residual exactly
    # improvement = 1 - 0.05/1.0 = 0.95 >= 0.50 -> pass
    # residual = 0.05 <= 0.05 -> pass
    result = compute_verification(baseline, verify, _criteria())
    assert result.residual_failure_rate == pytest.approx(0.05)
    assert result.is_healed is True


def test_is_healed_true_at_exact_threshold_improvement() -> None:
    """``>=`` semantics: improvement exactly equal to min passes."""
    baseline = _summary(_probe("p.A", 0, 100))   # 1.0
    verify = _summary(_probe("p.A", 50, 100))    # 0.50 residual
    # improvement = 1 - 0.50/1.0 = 0.50 >= 0.50 -> pass
    # residual  = 0.50 > 0.05 -> fail -> is_healed False despite ratio boundary
    result = compute_verification(baseline, verify, _criteria())
    assert result.is_healed is False
    assert result.passed_thresholds == (False, True)


def test_is_healed_false_when_residual_just_above_threshold() -> None:
    """Sanity boundary: residual strictly above cap fails the gate."""
    baseline = _summary(_probe("p.A", 0, 100))
    verify = _summary(_probe("p.A", 94, 100))  # 0.06 residual > 0.05
    result = compute_verification(baseline, verify, _criteria())
    assert result.passed_thresholds[0] is False
    assert result.is_healed is False


def test_results_independent_of_argument_mutation() -> None:
    """compute_verification must not mutate its input summaries."""
    baseline = _summary(_probe("p.A", 5, 10))
    verify = _summary(_probe("p.A", 9, 10))
    before_b = (baseline.total_attempts, baseline.total_failing, len(baseline.probes))
    before_v = (verify.total_attempts, verify.total_failing, len(verify.probes))
    compute_verification(baseline, verify, _criteria())
    assert (baseline.total_attempts, baseline.total_failing, len(baseline.probes)) == before_b
    assert (verify.total_attempts, verify.total_failing, len(verify.probes)) == before_v
