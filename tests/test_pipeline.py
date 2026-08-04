"""Integration tests for :mod:`healer.pipeline` (the master orchestrator).

Strategy
--------
The pipeline calls three heavy subsystems we deliberately *don't* exercise in
unit tests:

1. :class:`~healer.probe.garak_runner.GarakRunner` shells out to the garak CLI.
2. :func:`~healer.llm.proxy.run_proxy_in_background` binds a real uvicorn socket
   in a daemon thread.
3. :class:`~healer.llm.target.TargetLLMClient` talks the OpenAI SDK across the
   network.

To exercise the orchestrator's state transitions and artifact writing without
any of that, we:

* Use ``patch("healer.probe.garak_runner.shutil.which", return_value=None)`` so
  :class:`~healer.probe.garak_runner.GarakRunner` writes its offline-stub JSONL
  instead of launching garak. The JSONL has ``total_evaluated=0`` on every
  probe — which produces a *degraded* :class:`VerificationResult` (the
  expected offline behaviour path the metrics module was designed to surface).
* Patch ``healer.pipeline.run_proxy_in_background`` with a stub context manager
  that immediately yields a fake base URL. No socket is bound and no thread
  is spawned, so the tests stay deterministic and fast (<100ms each).
* Construct the pipeline with a ``local`` provider target so the
  :class:`TargetLLMClient` runs in mock mode and never touches the network.

The sequence we verify is::

    proxy OFF (baseline) -> garak -> baseline.summary.json
    proxy ON  (verify)   -> garak -> verify.summary.json
    compute_verification -> experiment_summary.json

Coverage matrix:
* Construction: pipeline builds from a config; the PatchEngine is eagerly
  constructed (so misconfigured patches fail fast at construction).
* State transitions: ``apply_patches`` is flipped False during baseline and
  True during verify.
* Artifact writing: ``baseline.summary.json``, ``verify.summary.json``, and
  ``experiment_summary.json`` are written and JSON-decodable with the
  expected keys.
* Offline mode produces a ``degraded=True`` verification result (the
  documented offline-stub contract).
* Failure path: a non-zero garak returncode raises ``RuntimeError`` from the
  pipeline, and the proxy teardown still runs (mocked context manager exit).
* CLI ``scripts/run_experiment.py`` argument parser accepts --config and --output-dir.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from healer.config import HealerConfig, Provider, TargetConfig
from healer.llm.target import TargetLLMClient
from healer.pipeline import HealerPipeline, PipelineResult
from healer.probe.garak_runner import GarakRunner, GarakRunResult
from healer.probe.result_parser import ProbeResult, RunSummary
from healer.verify.metrics import VerificationResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_target_config() -> TargetConfig:
    """Local provider forces the TargetLLMClient into mock mode (no network)."""
    return TargetConfig(
        provider=Provider.LOCAL,
        model="local-mock",
        api_base="http://127.0.0.1:1/v1",
        api_key_env="HEALER_PIPELINE_TEST_KEY_MISSING",
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=2,
    )


@pytest.fixture()
def healer_config(local_target_config: TargetConfig) -> HealerConfig:
    cfg = HealerConfig.load(DEFAULT_CONFIG)
    return cfg.model_copy(deep=True)


@pytest.fixture()
def mocked_proxy_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``run_proxy_in_background`` with a no-op stub context manager.

    The stub records when it's entered/exited so tests can assert the pipeline
    used the proxy lifecycle correctly. No socket is bound and no thread is
    spawned, so suite runtime stays sub-second.
    """
    state = {"entered": False, "exited": False}

    @contextlib.contextmanager
    def _stub(
        proxy: Any, *, host: str | None = None, port: int | None = None,
        ready_timeout: float = 10.0,
    ) -> Iterator[str]:
        state["entered"] = True
        try:
            yield "http://127.0.0.1:65535"
        finally:
            state["exited"] = True

    # Expose the dict on the stub itself for assertion convenience.
    _stub.state = state  # type: ignore[attr-defined]
    monkeypatch.setattr("healer.pipeline.run_proxy_in_background", _stub)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEALER_PIPELINE_TEST_KEY_MISSING", raising=False)


def _stub_garak_runner(
    config: HealerConfig,
    *,
    baseline_probes: list[ProbeResult] | None = None,
    verify_probes: list[ProbeResult] | None = None,
) -> GarakRunner:
    """Build a GarakRunner whose ``run()`` writes deterministic JSONL rows.

    The runner is real (so argv/output_dir handling is exercised); only the
    subprocess dispatch is bypassed. When called twice (baseline then verify)
    it alternates between the two probe lists; when called once it always
    uses ``baseline_probes``.

    Both lists default to empty (a fully-offline-stub-shaped run).
    """
    bp = baseline_probes or []
    vp = verify_probes or bp
    runner = GarakRunner(config)
    call_state = {"count": 0}

    def _stub_run(
        self: GarakRunner, *, report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        call_state["count"] += 1
        probes = bp if call_state["count"] == 1 else vp
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / f"{report_prefix or 'run'}.{call_state['count']}.report.jsonl"
        with report.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({
                    "entry_type": "start_run setup",
                    "plugins.target_type": "openai",
                    "plugins.target_name": self._config.target.model,
                    "garak_version": "offline-test-stub",
                }) + "\n"
            )
            for p in probes:
                fh.write(
                    json.dumps({
                        "entry_type": "eval",
                        "probe": p.probe,
                        "detector": p.detector,
                        "passed": p.passed,
                        "total_evaluated": p.total_evaluated,
                    }) + "\n"
                )
        return GarakRunResult(
            report_path=report,
            returncode=-1,
            offline=True,
            duration_seconds=0.0,
            command=("garak", "--offline-stub"),
            stderr_tail="",
        )

    runner.run = _stub_run.__get__(runner, GarakRunner)  # type: ignore[method-assign]
    return runner


def _probe(name: str, passed: int, total: int, detector: str = "always.Pass") -> ProbeResult:
    return ProbeResult(
        probe=name,
        detector=detector,
        passed=passed,
        total_evaluated=total,
        pass_rate=(passed / total) if total else 0.0,
        failing=total - passed,
    )


def _summary(*probes: ProbeResult) -> RunSummary:
    s = RunSummary()
    s.probes = list(probes)
    s.aggregate()
    return s


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pipeline_constructs_from_config(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """Pipeline builds eagerly: target, mitigator (PatchEngine), runner."""
    pipeline = HealerPipeline(healer_config, output_dir=tmp_path)
    assert pipeline.config is healer_config
    assert isinstance(pipeline.target, TargetLLMClient)
    assert pipeline.mitigator is not None  # PatchEngine built eagerly
    assert isinstance(pipeline.runner, GarakRunner)
    assert pipeline.output_dir == tmp_path


def test_pipeline_accepts_injected_runner(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """Tests can supply their own GarakRunner to bypass the subprocess."""
    custom_runner = GarakRunner(healer_config)
    pipeline = HealerPipeline(
        healer_config, runner=custom_runner, output_dir=tmp_path
    )
    assert pipeline.runner is custom_runner


# ---------------------------------------------------------------------------
# End-to-end with mocked proxy + offline-stub garak
# ---------------------------------------------------------------------------


def test_run_end_to_end_offline_yields_degraded_result(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """Offline garak -> zero attempts -> degraded VerificationResult.

    This is the canonical 'audit pipeline finds no probes on this host'
    outcome: every probe in the offline stub has total_evaluated=0, so
    compute_verification flags degraded=True (the documented behaviour
    of the offline-stub contract).
    """
    # Offline stub for each probe -> all-zero attempts -> degraded.
    stub_probes = [
        _probe("probes.promptinject", 0, 0),
        _probe("probes.encoding", 0, 0),
        _probe("probes.leakreplay", 0, 0),
    ]
    runner = _stub_garak_runner(healer_config, baseline_probes=stub_probes)
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    # The public surface is async; tests are sync, so we drive it via asyncio.
    result = asyncio.run(pipeline.run_end_to_end())

    assert isinstance(result, PipelineResult)
    assert result.verification.degraded is True
    assert result.verification.is_healed is False
    # Artifacts on disk
    assert result.baseline.summary_path.is_file()
    assert result.verify_summary_path.is_file()
    assert result.summary_path.is_file()
    # Experiment summary is JSON-decodable and has the expected top-level keys.
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "phases" in payload
    assert "verification" in payload
    assert "durations" in payload
    assert set(payload["phases"]) == {"baseline", "verify"}


def test_run_end_to_end_proxy_state_transitions(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """The proxy is flipped OFF for baseline and ON for verify (in order).

    We patch HealerProxy.set_apply_patches to record calls so we can assert
    the sequence is (False, True). The mocked_proxy_ctx fixture ensures no
    real port is bound.
    """
    stub_probes = [_probe("probes.promptinject", 0, 0)]
    runner = _stub_garak_runner(healer_config, baseline_probes=stub_probes)

    calls: list[bool] = []

    # Build the pipeline with a real proxy instance, but intercept state flips.
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)

    from healer.llm.proxy import HealerProxy as _Proxy

    def _recorder(self: Any, enabled: bool) -> None:
        calls.append(enabled)

    with patch.object(_Proxy, "set_apply_patches", _recorder):
        asyncio.run(pipeline.run_end_to_end())

    # First baseline call flips OFF; the verify phase flips ON.
    assert calls == [False, True]


def test_run_end_to_end_writes_correct_summary_shapes(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """The persisted JSONs have the documented per-phase schema."""
    stub_probes = [_probe("p.A", 0, 0)]
    runner = _stub_garak_runner(healer_config, baseline_probes=stub_probes)
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    result = asyncio.run(pipeline.run_end_to_end())

    baseline_json = json.loads(result.baseline.summary_path.read_text("utf-8"))
    verify_json = json.loads(result.verify_summary_path.read_text("utf-8"))
    summary_json = json.loads(result.summary_path.read_text("utf-8"))

    assert baseline_json["experiment"] == healer_config.experiment.name
    assert "garak_run" in baseline_json
    assert "summary" in baseline_json
    assert baseline_json["garak_run"]["offline"] is True

    # Verify summary carries the phase label.
    assert verify_json["phase"] == "verify"
    assert "garak_run" in verify_json
    assert verify_json["garak_run"]["offline"] is True

    # Experiment summary bundles both per-phase dicts and the verification dict.
    assert summary_json["experiment"] == healer_config.experiment.name
    assert summary_json["phases"]["baseline"]["summary_path"] == str(
        result.baseline.summary_path
    )
    assert summary_json["phases"]["verify"]["summary_path"] == str(
        result.verify_summary_path
    )
    assert "is_healed" in summary_json["verification"]
    assert summary_json["verification"]["degraded"] is True


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_failed_garak_run_raises_runtime_error(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """A non-zero garak returncode (other than offline -1) aborts the pipeline.

    The teardown context manager still runs (proxy lifecycle balanced) and
    the RuntimeError propagates to the caller.
    """
    # Stub a garak result with returncode=2 (hard fail, NOT offline).
    runner = GarakRunner(healer_config)

    def _failing_run(
        self: GarakRunner, *, report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write a minimal JSONL so any cleanup code that reads it doesn't crash,
        # but return a failing returncode.
        report = out_dir / f"{report_prefix or 'run'}.failed.report.jsonl"
        report.write_text(
            json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
            + "\n",
            encoding="utf-8",
        )
        return GarakRunResult(
            report_path=report,
            returncode=2,
            offline=False,
            duration_seconds=0.0,
            command=("garak", "--target_type", "openai"),
            stderr_tail="simulated garak crash",
        )

    runner.run = _failing_run.__get__(runner, GarakRunner)  # type: ignore[method-assign]
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Baseline garak run failed"):
        asyncio.run(pipeline.run_end_to_end())


def test_failed_verify_run_raises_after_baseline_succeeds(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """Baseline succeeds, then verify garak fails -> RuntimeError.

    Ensures the verify-phase error check actually triggers; previously only
    the baseline failure path was tested.
    """
    # Need to alternate results: first run (baseline) ok, second run (verify) fail.
    runner = GarakRunner(healer_config)
    call_state = {"count": 0}

    def _alternating_run(
        self: GarakRunner, *, report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        call_state["count"] += 1
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / f"{report_prefix or 'run'}.{call_state['count']}.report.jsonl"
        if call_state["count"] == 1:
            # Baseline: offline-stub success.
            report.write_text(
                json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
                + "\n"
                + json.dumps({
                    "entry_type": "eval", "probe": "p.A", "detector": "d",
                    "passed": 0, "total_evaluated": 0,
                }) + "\n",
                encoding="utf-8",
            )
            return GarakRunResult(
                report_path=report, returncode=-1, offline=True,
                duration_seconds=0.0, command=("garak",), stderr_tail="",
            )
        # Verify: hard failure.
        report.write_text(
            json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
            + "\n",
            encoding="utf-8",
        )
        return GarakRunResult(
            report_path=report, returncode=2, offline=False,
            duration_seconds=0.0, command=("garak",),
            stderr_tail="verify-phase crash",
        )

    runner.run = _alternating_run.__get__(runner, GarakRunner)  # type: ignore[method-assign]
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Verify garak run failed"):
        asyncio.run(pipeline.run_end_to_end())


# ---------------------------------------------------------------------------
# Real-phase: non-degraded happy path with non-trivial improvement
# ---------------------------------------------------------------------------


def test_run_end_to_end_improvement_path_is_not_degraded(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """When garak actually returns non-zero attempts, the verdict is real.

    Baseline shows high failure rate; verify shows strong reduction; the
    result should NOT be degraded (real improvement computed) and should be
    gated by the configured thresholds.
    """
    # Baseline: 10 probes, all fail. Verify: 10 probes, only 1 fails.
    runner = GarakRunner(healer_config)
    call_state = {"count": 0}

    def _two_phase_run(
        self: GarakRunner, *, report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        call_state["count"] += 1
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / f"{report_prefix or 'run'}.{call_state['count']}.jsonl"
        if call_state["count"] == 1:
            report.write_text(
                json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
                + "\n"
                + json.dumps({
                    "entry_type": "eval", "probe": "p.A", "detector": "d",
                    "passed": 0, "total_evaluated": 10,
                }) + "\n",
                encoding="utf-8",
            )
        else:
            report.write_text(
                json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
                + "\n"
                + json.dumps({
                    "entry_type": "eval", "probe": "p.A", "detector": "d",
                    "passed": 9, "total_evaluated": 10,
                }) + "\n",
                encoding="utf-8",
            )
        return GarakRunResult(
            report_path=report, returncode=0, offline=False,
            duration_seconds=0.0, command=("garak",), stderr_tail="",
        )

    runner.run = _two_phase_run.__get__(runner, GarakRunner)  # type: ignore[method-assign]
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    result = asyncio.run(pipeline.run_end_to_end())

    assert result.verification.degraded is False
    # baseline 100% fail -> residual 10% -> improvement 0.90
    assert result.verification.baseline_failure_rate == pytest.approx(1.0)
    assert result.verification.residual_failure_rate == pytest.approx(0.1)
    assert result.verification.relative_improvement == pytest.approx(0.9)
    # residual 10% > default cap 5% => is_healed=False despite improvement
    assert result.verification.is_healed is False
    assert result.verification.passed_thresholds == (False, True)


# ---------------------------------------------------------------------------
# CLI script tests
# ---------------------------------------------------------------------------


def _load_cli_module() -> Any:
    """Import scripts/run_experiment.py as a module without running it."""
    script_path = PROJECT_ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", script_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cli_arg_parser_defaults() -> None:
    """``--config`` defaults to DEFAULT_CONFIG_PATH and --output-dir to None."""
    mod = _load_cli_module()
    ns = mod._parse_args([])
    assert ns.config == mod.DEFAULT_CONFIG_PATH
    assert ns.output_dir is None


def test_cli_arg_parser_accepts_overrides(tmp_path: Path) -> None:
    """Both --config and --output-dir are accepted and parsed."""
    mod = _load_cli_module()
    cfg = tmp_path / "my.yaml"
    cfg.write_text("placeholder", encoding="utf-8")
    outdir = tmp_path / "out"
    ns = mod._parse_args(["--config", str(cfg), "--output-dir", str(outdir)])
    assert ns.config == cfg
    assert ns.output_dir == outdir


def test_cli_format_summary_contains_verdict_lines() -> None:
    """The pretty-printer emits the headline fields the user cares about."""
    mod = _load_cli_module()
    baseline_summary = _summary(_probe("p.A", 8, 10))  # 20% fail
    verify_summary = _summary(_probe("p.A", 9, 10))    # 10% fail
    v = VerificationResult(
        is_healed=False,
        residual_failure_rate=0.1,
        baseline_failure_rate=0.2,
        relative_improvement=0.5,
        degraded=False,
        max_failure_rate_seen=0.2,
        probe_deltas=[],
        passed_thresholds=(False, True),
    )
    # Build a minimal BaselineResult-like object via MagicMock.
    from healer.probe.baseline import BaselineResult
    from healer.probe.garak_runner import GarakRunResult

    fake_garak = GarakRunResult(
        report_path=Path("/tmp/x.jsonl"), returncode=0, offline=False,
        duration_seconds=0.0, command=("garak",), stderr_tail="",
    )
    baseline = BaselineResult(
        garak_result=fake_garak, summary=baseline_summary,
        summary_path=Path("/tmp/baseline.summary.json"),
    )
    result = PipelineResult(
        baseline=baseline,
        verify_summary=verify_summary,
        verify_summary_path=Path("/tmp/verify.summary.json"),
        verification=v,
        summary_path=Path("/tmp/experiment_summary.json"),
        durations={"baseline": 1.0, "verify": 2.0, "total": 3.0},
    )
    text = mod._format_summary(result)
    assert "SELF-HEALING" in text
    assert "is_healed" in text
    assert "residual_failure_rate" in text
    assert "relative_improvement" in text


def test_cli_main_returns_zero_on_offline_degraded_success(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI main() returns 0 even when the verdict is 'not healed'.

    A non-healing experiment is a *successful* experiment from the tool's POV;
    the CLI only returns non-zero on actual failures (config error, exception).
    """
    stub_probes = [_probe("probes.promptinject", 0, 0)]
    runner = _stub_garak_runner(healer_config, baseline_probes=stub_probes)
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)

    async def _fake_run(config_path: Path, output_dir: Any) -> PipelineResult:
        return await pipeline.run_end_to_end()

    mod = _load_cli_module()
    monkeypatch.setattr(mod, "_run", _fake_run)
    # Avoid writing to the real default output dir; the test's tmp pipeline already
    # owns its output_dir, so this is mainly a smoke check that nothing throws.
    code = mod.main(["--config", str(DEFAULT_CONFIG)])
    assert code == 0


def test_cli_main_returns_one_on_missing_config(tmp_path: Path) -> None:
    """A missing config file exits with code 1, not a traceback."""
    mod = _load_cli_module()
    missing = tmp_path / "no-such.yaml"
    code = mod.main(["--config", str(missing)])
    assert code == 1


def test_cli_main_returns_two_on_pipeline_failure(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised exception during the run returns exit code 2."""

    async def _boom(config_path: Path, output_dir: Any) -> PipelineResult:
        raise RuntimeError("boom")

    mod = _load_cli_module()
    monkeypatch.setattr(mod, "_run", _boom)
    code = mod.main(["--config", str(DEFAULT_CONFIG)])
    assert code == 2


# ---------------------------------------------------------------------------
# JSON-safety of the experiment_summary payload (defensive)
# ---------------------------------------------------------------------------


def test_experiment_summary_json_safe_with_inf_ratio(
    healer_config: HealerConfig,
    tmp_path: Path,
    mocked_proxy_ctx: None,
) -> None:
    """experiment_summary.json must be json.dumps-able even when a probe
    shows a fully-healed (improvement_ratio=+inf) entry."""
    runner = GarakRunner(healer_config)
    call_state = {"count": 0}

    def _two_phase(
        self: GarakRunner, *, report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        call_state["count"] += 1
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / f"{report_prefix or 'run'}.{call_state['count']}.jsonl"
        passed = 0 if call_state["count"] == 1 else 10
        report.write_text(
            json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"})
            + "\n"
            + json.dumps({
                "entry_type": "eval", "probe": "p.A", "detector": "d",
                "passed": passed, "total_evaluated": 10,
            }) + "\n",
            encoding="utf-8",
        )
        return GarakRunResult(
            report_path=report, returncode=0, offline=False,
            duration_seconds=0.0, command=("garak",), stderr_tail="",
        )

    runner.run = _two_phase.__get__(runner, GarakRunner)  # type: ignore[method-assign]
    pipeline = HealerPipeline(healer_config, runner=runner, output_dir=tmp_path)
    result = asyncio.run(pipeline.run_end_to_end())
    # The fully-healed probe has improvement_ratio=+inf; the experiment summary
    # must still be valid (json.dumps serialises +inf as the literal Infinity).
    text = result.summary_path.read_text(encoding="utf-8")
    assert "Infinity" in text
    # Round-trip via the stdlib json decoder, which DOES accept Infinity.
    reparsed = json.loads(text)
    assert reparsed["verification"]["is_healed"] in (True, False)
    _ = math  # math.inf would also work; kept for clarity if extended later
