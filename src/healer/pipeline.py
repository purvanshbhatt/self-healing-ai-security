"""Master orchestrator: probe -> patch -> verify end-to-end pipeline.

This module wires the prior layers (:mod:`healer.config`, :mod:`healer.llm.target`,
:mod:`healer.llm.proxy`, :mod:`healer.patch`, :mod:`healer.probe`,
:mod:`healer.verify`) into a single reusable :class:`HealerPipeline` so the CLI
in ``scripts/run_experiment.py`` can launch a full self-healing cycle in one call.

Execution model
---------------
The pipeline's public entry point :meth:`HealerPipeline.run_end_to_end` is
``async`` so the CLI can drive it with ``asyncio.run`` and downstream
dashboards can compose it with other async workloads. The heavy lifting
inside, however, is consciously *synchronous*:

* :class:`~healer.llm.proxy.HealerProxy` is driven by
  :func:`~healer.llm.proxy.run_proxy_in_background` — a sync context manager
  that spawns the uvicorn server in a daemon thread. The sanctioned usage
  (see its module docstring) is to enter the ``with`` block on the
  orchestrator's thread.

* :class:`~healer.probe.garak_runner.GarakRunner.run` shells out to the
  ``garak`` CLI via :func:`subprocess.run` — a blocking call by design.

To honour both, :meth:`run_end_to_end` off-loads the entire synchronous
``baseline -> verify`` sequence to a worker thread via
:func:`asyncio.to_thread`, then ``await`` s its result. The proxy-thread
(daemon) and the garak-subprocess therefore both live on the same worker
thread, which is exactly the layout the proxy was written for.

Phase contract
--------------
1. **Baseline**: proxy started in *unpatched* mode. ``establish_baseline``
   from :mod:`healer.probe.baseline` flips ``apply_patches=False``, runs
   garak, parses the JSONL into a :class:`RunSummary`, and persists
   ``baseline.summary.json``.
2. **Verify**: same proxy instance, ``apply_patches=True``. Garak runs
   again, the new JSONL parses into a second :class:`RunSummary`, persisted
   as ``verify.summary.json``.
3. **Decision**: :func:`~healer.verify.metrics.compute_verification` diffs
   the two summaries against
   :attr:`HealerConfig.verify.success_criteria`, producing a
   :class:`~healer.verify.metrics.VerificationResult`. The result is
   serialised to ``experiment_summary.json`` alongside the two intermediate
   summaries so the audit trail is fully reconstructable from disk alone.

Liquid-failure semantics
-----------------------
If either garak run fails (non-zero returncode or missing JSONL), the
pipeline logs the failure at ``ERROR`` and re-raises the
:class:`RuntimeError`. The proxy teardown path is wrapped in a ``finally``
so the listening port is always released, even when a probe phase dies
mid-run. This is critical for test isolation — otherwise
``run_proxy_in_background`` would leak the port and the next test would
fail with ``OSError: [WinError 10048]``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from healer.config import HealerConfig
from healer.llm.proxy import HealerProxy, Mitigator, run_proxy_in_background
from healer.llm.target import TargetLLMClient
from healer.logging_setup import configure_logging, get_logger
from healer.patch.engine import PatchEngine
from healer.probe.baseline import BaselineResult, establish_baseline
from healer.probe.garak_runner import GarakRunner, GarakRunResult
from healer.probe.result_parser import RunSummary, parse_report
from healer.verify.metrics import VerificationResult, compute_verification

_LOG: Final[logging.Logger] = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Full outcome of one :meth:`HealerPipeline.run_end_to_end` invocation.

    Carries every intermediate the orchestrator produced so CI dashboards
    and the CLI pretty-printer can show as much (or as little) as they
    want without re-reading disk.

    Attributes:
        baseline: Pre-patch :class:`BaselineResult` (with :class:`RunSummary`
            and the ``baseline.summary.json`` path).
        verify_summary: Post-patch :class:`RunSummary`.
        verify_summary_path: Where ``verify.summary.json`` was written.
        verification: :class:`VerificationResult` from compute_verification.
        summary_path: Path to the experiment-level ``experiment_summary.json``
            that bundles baseline + verify + the verification verdict.
        durations: Per-phase wall-clock seconds. Keys: ``baseline``,
            ``verify``, ``total``.
    """

    baseline: BaselineResult
    verify_summary: RunSummary
    verify_summary_path: Path
    verification: VerificationResult
    summary_path: Path
    durations: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mitigator adapter
# ---------------------------------------------------------------------------


class _EngineMitigatorAdapter:
    """Adapter exposing :class:`PatchEngine` as the :class:`Mitigator` Protocol.

    The proxy (:class:`~healer.llm.proxy.HealerProxy`) consumes a
    :class:`~healer.llm.proxy.Mitigator` whose ``patch_prompt`` /
    ``patch_output`` methods return ``(content, was_modified)`` tuples.
    :class:`~healer.patch.engine.PatchEngine` itself exposes a *loose*
    single-return surface (returns only the new ``content``); its tuple-form
    API lives in :meth:`~healer.patch.engine.PatchEngine.process_request`
    / :meth:`~healer.patch.engine.PatchEngine.process_response`. This adapter
    bridges the two by delegating to the tuple-form methods and projecting
    out only the ``(content, was_modified)`` pair the proxy expects.

    The adapter is a private implementation detail of :mod:`healer.pipeline`
    so the rest of the codebase never has to import the tuple-form contract
    alongside the loose one.
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: PatchEngine) -> None:
        self._engine = engine

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        audit, msgs = await self._engine.process_request(messages)
        return msgs, audit.was_modified

    async def patch_output(self, response: str) -> tuple[str, bool]:
        audit, text = await self._engine.process_response(response)
        return text, audit.was_modified


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class HealerPipeline:
    """Drives the probe -> patch -> verify cycle end-to-end."""

    def __init__(
        self,
        config: HealerConfig,
        *,
        target: TargetLLMClient | None = None,
        mitigator: Mitigator | None = None,
        runner: GarakRunner | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: A fully validated :class:`HealerConfig`. Source of truth
                for every downstream knob the pipeline uses.
            target: Optional pre-built :class:`TargetLLMClient`. When ``None``,
                a fresh client is built from ``config.target``. Useful for
                tests that want to inject a mock-mode target.
            mitigator: Optional :class:`Mitigator` satisfying the structural
                protocol. When ``None`` a :class:`PatchEngine` is built from
                ``config.patch``. Passing ``None`` does NOT disable patching —
                pass a no-op mitigator if that is the intent.
            runner: Optional pre-built :class:`GarakRunner`. When ``None`` a
                fresh one is constructed from ``config``. Tests inject a
                stub runner to avoid the real subprocess.
            output_dir: Override the experiment-level report output directory
                (``config.experiment.output_dir``). Useful for tests so they
                can point at ``tmp_path`` instead of polluting ``data/reports/``.
        """
        self.config: HealerConfig = config
        self.target: TargetLLMClient = target if target is not None else TargetLLMClient(
            config.target
        )
        # PatchEngine is built eagerly so a misconfigured patch layer fails
        # fast at construction rather than mid-experiment. The proxy
        # consumes the Mitigator Protocol (tuple-returning); the adapter in
        # front of PatchEngine exposes the tuple-form surface the proxy
        # expects. If the caller already injected a Mitigator, use it as-is.
        if mitigator is not None:
            self.mitigator: Mitigator | None = mitigator
        else:
            self.mitigator = _EngineMitigatorAdapter(PatchEngine(config))
        self.runner: GarakRunner = runner if runner is not None else GarakRunner(config)
        self.output_dir: Path = Path(output_dir or config.experiment.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_end_to_end(self) -> PipelineResult:
        """Run baseline -> verify -> decide, end-to-end.

        Off-loads the synchronous proxy+garak orchestration to a worker
        thread so the public surface stays awaitable from async call sites
        (CLI, fastapi background tasks, pytest-asyncio).

        Returns:
            :class:`PipelineResult` carrying every intermediate and audit
            artifact path. Never ``None``.

        Raises:
            RuntimeError: Re-raised from a failed baseline or verify phase
                (e.g. garak exited non-zero, JSONL missing). The proxy is
                always torn down before the raise propagates.
        """
        _LOG.info("Pipeline starting (experiment=%s).", self.config.experiment.name)
        # configure_logging is idempotent — safe to call from the async entry
        # point in case the CLI forgot to.
        configure_logging(self.config)
        result = await asyncio.to_thread(self._run_sync)
        _LOG.info(
            "Pipeline complete (is_healed=%s, residual=%.3f, improvement=%.3f).",
            result.verification.is_healed,
            result.verification.residual_failure_rate,
            result.verification.relative_improvement,
        )
        return result

    # ------------------------------------------------------------------
    # Synchronous orchestration — runs on the to_thread worker.
    # ------------------------------------------------------------------

    def _run_sync(self) -> PipelineResult:
        """Drive proxy + runner + parser + verifier synchronously.

        The proxy is brought up once in unpatched mode; ``establish_baseline``
        keeps the same proxy instance but flips ``apply_patches=False`` so the
        baseline probes hit the raw target. We then flip ``apply_patches=True``
        in-place for the verify phase — the proxy's state object is mutated
        live, and the next wave of garak requests observes the patched path.
        """
        proxy = HealerProxy(self.config, self.target, mitigator=self.mitigator)
        # Use lower ports/shorter timeouts in tests via config overrides.
        host = self.config.proxy.host
        port = self.config.proxy.port

        t_pipeline_start = time.perf_counter()

        with run_proxy_in_background(
            proxy, host=host, port=port, ready_timeout=10.0
        ) as base_url:
            _LOG.info("Proxy ready at %s.", base_url)

            # ---------- Baseline phase ----------
            t0 = time.perf_counter()
            baseline = establish_baseline(
                self.config, proxy, runner=self.runner, output_dir=self.output_dir
            )
            if baseline.garak_result.returncode not in (0, -1):
                # -1 is offline-stub mode (acceptable); anything else is a hard
                # failure and we abort before the verify phase wastes tokens.
                raise RuntimeError(
                    f"Baseline garak run failed (returncode="
                    f"{baseline.garak_result.returncode}): "
                    f"{baseline.garak_result.stderr_tail[:300]}"
                )
            baseline_duration = time.perf_counter() - t0
            _LOG.info(
                "Baseline phase complete: overall_pass_rate=%.3f, attempts=%d, "
                "failing=%d, duration=%.2fs.",
                baseline.summary.overall_pass_rate,
                baseline.summary.total_attempts,
                baseline.summary.total_failing,
                baseline_duration,
            )

            # ---------- Verify phase ----------
            t0 = time.perf_counter()
            verify_summary, verify_summary_path, verify_run = self._run_verify_phase(
                proxy
            )
            if verify_run.returncode not in (0, -1):
                raise RuntimeError(
                    f"Verify garak run failed (returncode="
                    f"{verify_run.returncode}): "
                    f"{verify_run.stderr_tail[:300]}"
                )
            verify_duration = time.perf_counter() - t0
            _LOG.info(
                "Verify phase complete: overall_pass_rate=%.3f, attempts=%d, "
                "failing=%d, duration=%.2fs.",
                verify_summary.overall_pass_rate,
                verify_summary.total_attempts,
                verify_summary.total_failing,
                verify_duration,
            )

        # Proxy has been torn down by the `with` exit; compute the verdict
        # outside the proxy context so we don't hold the port during the
        # (cheap) math + serialisation.
        criteria = self.config.verify.success_criteria
        verification = compute_verification(
            baseline.summary, verify_summary, criteria
        )

        summary_path = self.output_dir / "experiment_summary.json"
        durations = {
            "baseline": baseline_duration,
            "verify": verify_duration,
            "total": time.perf_counter() - t_pipeline_start,
        }
        self._write_summary_json(
            summary_path=summary_path,
            experiment_name=self.config.experiment.name,
            baseline=baseline,
            verify_summary=verify_summary,
            verify_summary_path=verify_summary_path,
            verify_run=verify_run,
            verification=verification,
            durations=durations,
        )

        return PipelineResult(
            baseline=baseline,
            verify_summary=verify_summary,
            verify_summary_path=verify_summary_path,
            verification=verification,
            summary_path=summary_path,
            durations=durations,
        )

    # ------------------------------------------------------------------
    # Verify phase helper
    # ------------------------------------------------------------------

    def _run_verify_phase(
        self, proxy: HealerProxy
    ) -> tuple[RunSummary, Path, GarakRunResult]:
        """Run garak with patches ON, parse, persist ``verify.summary.json``.

        Returns:
            ``(verify_summary, summary_path, garak_run_result)``.
        """
        proxy.set_apply_patches(True)
        # The verify config exposes its own report_prefix (defaults to
        # "run_verify"); we honour it directly so users can disambiguate
        # baseline/verify JSONL files in a shared output_dir.
        prefix = self.config.verify.report_prefix
        garak_result = self.runner.run(
            report_prefix=prefix,
            output_dir=self.output_dir,
        )
        summary = parse_report(garak_result.report_path)
        summary_path = self.output_dir / "verify.summary.json"
        payload = {
            "experiment": self.config.experiment.name,
            "phase": "verify",
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
        return summary, summary_path, garak_result

    # ------------------------------------------------------------------
    # Audit serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _write_summary_json(
        *,
        summary_path: Path,
        experiment_name: str,
        baseline: BaselineResult,
        verify_summary: RunSummary,
        verify_summary_path: Path,
        verify_run: GarakRunResult,
        verification: VerificationResult,
        durations: dict[str, float],
    ) -> None:
        """Emit ``experiment_summary.json`` bundling the full audit trail."""
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": experiment_name,
            "phases": {
                "baseline": {
                    "summary_path": str(baseline.summary_path),
                    "report_path": str(baseline.garak_result.report_path),
                    "offline": baseline.garak_result.offline,
                    "returncode": baseline.garak_result.returncode,
                    "duration_seconds": baseline.garak_result.duration_seconds,
                    "summary": baseline.summary.to_dict(),
                },
                "verify": {
                    "summary_path": str(verify_summary_path),
                    "report_path": str(verify_run.report_path),
                    "offline": verify_run.offline,
                    "returncode": verify_run.returncode,
                    "duration_seconds": verify_run.duration_seconds,
                    "summary": verify_summary.to_dict(),
                },
            },
            "verification": verification.to_dict(),
            "durations": durations,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        _LOG.info("Wrote experiment summary to %s.", summary_path)


__all__ = ["HealerPipeline", "PipelineResult"]
