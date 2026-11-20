#!/usr/bin/env python
"""CLI entry point for the Self-Healing AI Security Pipeline.

Runs a complete probe -> patch -> verify cycle from a config file and prints a
human-readable summary of the verdict to the terminal. All machine-readable
artifacts (garak JSONL, baseline/verify summaries, experiment_summary.json)
are written to ``config.experiment.output_dir`` (default ``data/reports/``).

Usage
-----
.. code-block:: shell

    python scripts/run_experiment.py --config config.yaml
    python scripts/run_experiment.py               # uses config.yaml from cwd

Exit codes
----------
* ``0``  : pipeline ran to completion, regardless of the verification verdict.
           The ``is_healed`` flag is reported but doesn't fail the CLI —
           a "not healed" result is a successful *experiment*, not a tool
           failure.
* ``1``  : configuration did not load (file missing / validation error).
* ``2``  : pipeline raised at runtime (garak subprocess failed, proxy didn't
           bind the port, etc.). The traceback is written to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Final

import os
from dotenv import load_dotenv

# Make the ``healer`` package importable when this script is run directly
# (``python scripts/run_experiment.py``) — there is no install-on-the-fly
# here, the user is expected to run from a project with ``src/`` on path or
# with the venv active.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_SRC: Final[Path] = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load .env file variables into environment
load_dotenv(_REPO_ROOT / ".env")

# Ensure local venv executable directory is in PATH for garak execution
_VENV_SCRIPTS = _REPO_ROOT / ".venv" / "Scripts"
_VENV_BIN = _REPO_ROOT / ".venv" / "bin"
if _VENV_SCRIPTS.is_dir() and str(_VENV_SCRIPTS) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_SCRIPTS}{os.pathsep}{os.environ.get('PATH', '')}"
elif _VENV_BIN.is_dir() and str(_VENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

from healer.config import DEFAULT_CONFIG_PATH, HealerConfig  # noqa: E402
from healer.logging_setup import configure_logging  # noqa: E402
from healer.pipeline import HealerPipeline, PipelineResult  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse namespace for the CLI."""
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description=(
            "Run a Self-Healing AI Security experiment: probe -> patch -> "
            "verify against a target LLM."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override config.experiment.output_dir for this run. "
            "Useful for stamping experiment runs in their own folder."
        ),
    )
    return parser.parse_args(argv)


def _format_summary(result: PipelineResult) -> str:
    """Pretty-print the pipeline verdict.

    Returns a multi-line string suitable for stdout. Kept free of emojis and
    markup so it survives any TTY / pipe / CI log aggregator.
    """
    v = result.verification
    lines: list[str] = [
        "=" * 72,
        "SELF-HEALING AI SECURITY PIPELINE — EXPERIMENT SUMMARY",
        "=" * 72,
        f"  is_healed               : {v.is_healed}",
        f"  residual_failure_rate   : {v.residual_failure_rate:.4f}",
        f"  baseline_failure_rate   : {v.baseline_failure_rate:.4f}",
        f"  relative_improvement    : {v.relative_improvement:.4f}",
        f"  degraded                : {v.degraded}",
    ]
    if v.degraded:
        lines.append(f"  degradation_reason      : {v.degradation_reason}")
    lines.extend(
        [
            f"  max_failure_rate_seen    : {v.max_failure_rate_seen:.4f}",
            (
                "  passed_thresholds        : "
                f"max_failure_rate={v.passed_thresholds[0]}, "
                f"min_relative_improvement={v.passed_thresholds[1]}"
            ),
            "-" * 72,
            "  Probes diff:",
        ]
    )
    if not v.probe_deltas:
        lines.append("    (no probes assessed)")
    else:
        for d in v.probe_deltas:
            lines.append(
                f"    {d.probe:<32} {d.detector:<18} "
                f"baseline={d.baseline_failure_rate:.2f} -> "
                f"verify={d.verify_failure_rate:.2f} "
                f"[{d.verdict}] ratio={d.improvement_ratio}"
            )
    lines.extend(
        [
            "-" * 72,
            f"  baseline summary json : {result.baseline.summary_path}",
            f"  verify summary json    : {result.verify_summary_path}",
            f"  experiment summary     : {result.summary_path}",
            (
                f"  durations              : "
                f"baseline={result.durations.get('baseline', 0):.2f}s, "
                f"verify={result.durations.get('verify', 0):.2f}s, "
                f"total={result.durations.get('total', 0):.2f}s"
            ),
            "=" * 72,
        ]
    )
    return "\n".join(lines)


async def _run(config_path: Path, output_dir: Path | None) -> PipelineResult:
    """Load config, build the pipeline, await the verdict."""
    config = HealerConfig.load(config_path)
    configure_logging(config)
    pipeline = HealerPipeline(config, output_dir=output_dir)
    return await pipeline.run_end_to_end()


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point. Returns the exit code (see module docstring)."""
    args = _parse_args(argv)
    config_path: Path = args.config
    try:
        result = asyncio.run(_run(config_path, args.output_dir))
    except FileNotFoundError as exc:
        print(f"ERROR: config not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"ERROR: pipeline failed: {exc!r}", file=sys.stderr)
        return 2

    print(_format_summary(result))
    # Echo a compact JSON line for downstream CI parsers.
    print(
        "::set-output name=verification::"
        + json.dumps(result.verification.to_dict())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
