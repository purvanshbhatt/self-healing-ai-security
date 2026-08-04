"""Parser for garak ``.report.jsonl`` files.

This module reduces garak's per-evaluation JSONL records into a flat
:class:`ProbeResult` schema that the verify phase
(:mod:`healer.verify.metrics`) can diff between baseline and verify runs.

JSONL record shapes (validated against garak's :mod:`garak.report.Report`)
------------------------------------------------------------------------

Two kinds of records ride the JSONL stream:

1. ``entry_type == "start_run setup"`` — one-per-file metadata record
   carrying ``plugins.target_type`` and ``plugins.target_name`` plus the
   garak version. We capture these into :attr:`RunSummary.metadata`.

2. ``entry_type == "eval"`` — one per (probe x detector) pair. Fields we
   consume:
   - ``probe``: ``"module.Class"`` form (post-import normalization).
   - ``detector``: detector plugin name.
   - ``passed``: integer count of probe attempts that *resisted* the probe.
   - ``total_evaluated``: integer count of probe attempts actually assessed.

Pass-rate is computed as ``passed / total_evaluated``; an empty probe
(``total_evaluated == 0``) gets a ``pass_rate`` of ``0.0`` rather than
``NaN`` — the latter would propagate into JSON-unsafe sums downstream and
break the audit log. The convention is documented here in case a future
contributor wonders why a "no-op" run reports 0.0 instead of NaN.

Defensive parsing
-----------------
Malformed JSONL lines (non-JSON, missing required fields) are *not* fatal:
the parser collects them into :attr:`RunSummary.skipped_records` so the
pipeline surfaces garbage-rows in the audit without crashing a 30-minute
probe run on a single bad row.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from healer.logging_setup import get_logger

_LOG: Final[logging.Logger] = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One (probe x detector) row from the JSONL.

    ``total_evaluated == 0`` means the probe produced no assessable
    attempts (e.g. generator returned zero responses, or the offline
    stub). ``pass_rate`` is 0.0 in that case, not NaN — see module doc.

    Attributes:
        probe: ``"module.Class"`` form, e.g. ``"dan.DanInTheWild"``.
        detector: Detector plugin name, e.g. ``"always.Pass"``.
        passed: Number of attempts that resisted the probe.
        total_evaluated: Number of attempts assessed.
        pass_rate: ``passed / total_evaluated`` (0.0 when total is 0).
        failing: Convenience: ``total_evaluated - passed``.
    """

    probe: str
    detector: str
    passed: int
    total_evaluated: int
    pass_rate: float
    failing: int

    def to_dict(self) -> dict[str, object]:
        """Flat dict for JSONL audit output."""
        return {
            "probe": self.probe,
            "detector": self.detector,
            "passed": self.passed,
            "total_evaluated": self.total_evaluated,
            "pass_rate": self.pass_rate,
            "failing": self.failing,
        }


@dataclass(slots=True)
class RunSummary:
    """Aggregated metrics for one garak run (baseline OR verify).

    The verify phase diffs two :class:`RunSummary` instances to produce
    the experiment-level improvement table.

    Attributes:
        records_seen: Total JSONL lines parsed (any entry_type).
        skipped_records: Lines skipped due to malformed JSON / missing keys.
        probes: One :class:`ProbeResult` per (probe, detector) eval row.
        metadata: Parsed ``start_run setup`` row (may be empty dict).
        total_attempts: Sum of ``total_evaluated`` across all probes.
        total_failing: Sum of ``failing`` across all probes.
        mean_pass_rate: Unweighted mean of probe-level ``pass_rate``.
        overall_pass_rate: ``total_attempts - total_failing`` over
            ``total_attempts`` (weighted) — the single number the verify
            layer uses for success-criterion comparison.
    """

    records_seen: int = 0
    skipped_records: int = 0
    probes: list[ProbeResult] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    total_attempts: int = 0
    total_failing: int = 0
    mean_pass_rate: float = 0.0
    overall_pass_rate: float = 0.0

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def aggregate(self) -> None:
        """Recompute aggregate fields from :attr:`probes`.

        Idempotent; the verify layer calls it after any mutation to
        :attr:`probes`. Cheap (single pass).
        """
        if not self.probes:
            self.total_attempts = 0
            self.total_failing = 0
            self.mean_pass_rate = 0.0
            self.overall_pass_rate = 0.0
            return
        attempts = sum(p.total_evaluated for p in self.probes)
        failing = sum(p.failing for p in self.probes)
        passes = sum(p.passed for p in self.probes)
        self.total_attempts = attempts
        self.total_failing = failing
        self.mean_pass_rate = sum(p.pass_rate for p in self.probes) / len(self.probes)
        self.overall_pass_rate = (
            (passes / attempts) if attempts else 0.0
        )

    def to_dict(self) -> dict[str, object]:
        """Flat dict for JSONL audit output."""
        return {
            "records_seen": self.records_seen,
            "skipped_records": self.skipped_records,
            "probes": [p.to_dict() for p in self.probes],
            "metadata": dict(self.metadata),
            "total_attempts": self.total_attempts,
            "total_failing": self.total_failing,
            "mean_pass_rate": self.mean_pass_rate,
            "overall_pass_rate": self.overall_pass_rate,
        }


# ---------------------------------------------------------------------------
# Public parsing entry point
# ---------------------------------------------------------------------------


def parse_report(jsonl_path: Path | str) -> RunSummary:
    """Parse a garak ``.report.jsonl`` file into a :class:`RunSummary`.

    Args:
        jsonl_path: Path to the garak JSONL report.

    Returns:
        Fully aggregated :class:`RunSummary` (with row sums already
        populated via :meth:`RunSummary.aggregate`).

    Raises:
        FileNotFoundError: ``jsonl_path`` does not exist.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"garak report not found: {path}")

    summary = RunSummary()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            summary.records_seen += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                summary.skipped_records += 1
                _LOG.warning(
                    "%s:%d not JSON (%s); skipping", path, line_no, exc
                )
                continue
            if not isinstance(record, dict):
                summary.skipped_records += 1
                continue

            entry_type = record.get("entry_type")
            if entry_type == "start_run setup":
                # Capture the metadata row verbatim; callers downstream
                # decide which fields they want (model, garak version, etc.).
                summary.metadata = {
                    k: v
                    for k, v in record.items()
                    if k != "entry_type"
                }
                continue

            if entry_type != "eval":
                _LOG.debug(
                    "%s:%d unrecognised entry_type=%r; skipping",
                    path, line_no, entry_type,
                )
                summary.skipped_records += 1
                continue

            probe_result = _parse_eval_row(record, path, line_no)
            if probe_result is None:
                summary.skipped_records += 1
                continue
            summary.probes.append(probe_result)

    summary.aggregate()
    _LOG.info(
        "Parsed garak report %s: records=%d, probes=%d, skipped=%d, "
        "overall_pass_rate=%.3f, total_attempts=%d, total_failing=%d",
        path,
        summary.records_seen,
        len(summary.probes),
        summary.skipped_records,
        summary.overall_pass_rate,
        summary.total_attempts,
        summary.total_failing,
    )
    return summary


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_eval_row(
    record: dict[str, object],
    path: Path,
    line_no: int,
) -> ProbeResult | None:
    """Validate one ``entry_type=eval`` JSONL record and build a ProbeResult.

    Returns None when required fields are missing or of the wrong type;
    the caller increments ``skipped_records`` accordingly. We refuse to
    coerce silently — a misformed row is suspicious (likely an upstream
    garak version drift) and should be visible in the audit, not masked.
    """
    probe = record.get("probe")
    detector = record.get("detector")
    passed = record.get("passed")
    total = record.get("total_evaluated")
    if not isinstance(probe, str) or not probe:
        _LOG.warning(
            "%s:%d eval row missing/invalid 'probe' field; skipping",
            path, line_no,
        )
        return None
    if not isinstance(detector, str):
        # Some garak rows have no detector. Default to the legacy sentinel
        # so the (probe, detector) key tuple stays stable.
        detector = "unknown"
    if not isinstance(passed, int) or not isinstance(total, int):
        _LOG.warning(
            "%s:%d eval row passed/total not integers (%r, %r); skipping",
            path, line_no, passed, total,
        )
        return None
    if total < 0 or passed < 0 or passed > total:
        _LOG.warning(
            "%s:%d eval row inconsistent passed=%d total=%d; skipping",
            path, line_no, passed, total,
        )
        return None
    pass_rate = (passed / total) if total else 0.0
    return ProbeResult(
        probe=probe,
        detector=str(detector),
        passed=passed,
        total_evaluated=total,
        pass_rate=pass_rate,
        failing=total - passed,
    )


__all__ = ["ProbeResult", "RunSummary", "parse_report"]
