"""Probe the target once *without* patches to record pre-mitigation profile.

This module is the orchestrator glue between :class:`healer.llm.proxy.HealerProxy`
(forwarding requests without mitigation), :class:`healer.probe.garak_runner.GarakRunner`
(launching garak), and :mod:`healer.probe.result_parser` (reducing JSONL into
a :class:`healer.probe.result_parser.RunSummary`).

The baseline phase deliberately re-uses the same proxy instance that the verify
phase will reuse — only :meth:`HealerProxy.set_apply_patches(False)` differs —
so the verify phase can attribute any delta in failure counts *only* to the
patch layer. We also persist the resulting :class:`RunSummary` as a JSON file
alongside the garak JSONL so the verify phase can load it without re-parsing
the larger JSONL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from healer.config import HealerConfig
from healer.llm.proxy import HealerProxy
from healer.logging_setup import get_logger
from healer.probe.garak_runner import GarakRunner, GarakRunResult
from healer.probe.result_parser import RunSummary, parse_report

_LOG: Final[logging.Logger] = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Outcome of one :func:`establish_baseline` invocation.

    Attributes:
        garak_result: The raw :class:`GarakRunResult` from the runner.
        summary: Parsed :class:`RunSummary` with per-probe metrics.
        summary_path: Path to the JSON file written for the verify phase.
    """

    garak_result: GarakRunResult
    summary: RunSummary
    summary_path: Path


def establish_baseline(
    config: HealerConfig,
    proxy: HealerProxy,
    *,
    runner: GarakRunner | None = None,
    output_dir: Path | None = None,
) -> BaselineResult:
    """Probe the target LLM with patches off, parse results, persist summary.

    Side-effects:
    * Calls :meth:`HealerProxy.set_apply_patches(False)` — ensures the
      baseline probe goes through the unmodified target.
    * Writes ``{output_dir}/baseline.summary.json`` with the serialised
      :class:`RunSummary` so the verify phase can ``json.load`` it instead
      of re-parsing the raw JSONL.

    Args:
        config: Validated :class:`HealerConfig`.
        proxy: Live :class:`HealerProxy` (will be flipped to apply_patches=False
            for the duration of this call).
        runner: Optional pre-built :class:`GarakRunner`. Defaults to a fresh
            instance constructed from ``config`` and the proxy's base URL.
        output_dir: Override the experiment ``output_dir`` for the summary
            file location.

    Returns:
        :class:`BaselineResult` carrying the run outcome, parsed summary,
        and persisted summary path.
    """
    proxy.set_apply_patches(False)

    if runner is None:
        runner = GarakRunner(config)

    garak_result = runner.run(
        report_prefix=f"{config.probe.report_prefix}_baseline",
        output_dir=output_dir,
    )
    summary = parse_report(garak_result.report_path)

    out_dir = Path(output_dir or config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "baseline.summary.json"

    payload = {
        "experiment": config.experiment.name,
        "garak_run": {
            "returncode": garak_result.returncode,
            "offline": garak_result.offline,
            "duration_seconds": garak_result.duration_seconds,
            "command": list(garak_result.command),
            "report_path": str(garak_result.report_path),
        },
        "summary": summary.to_dict(),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _LOG.info(
        "Baseline established: overall_pass_rate=%.3f, total_attempts=%d, "
        "total_failing=%d, summary=%s",
        summary.overall_pass_rate,
        summary.total_attempts,
        summary.total_failing,
        summary_path,
    )
    return BaselineResult(
        garak_result=garak_result,
        summary=summary,
        summary_path=summary_path,
    )


__all__ = ["BaselineResult", "establish_baseline"]
