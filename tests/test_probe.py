"""Tests for the probe layer: garak_runner, result_parser, baseline.

Coverage matrix:
* :class:`GarakRunner`
  - Offline mode (garak NOT on PATH) writes a stub JSONL with one
    ``start_run setup`` record + one ``eval`` per configured probe.
  - The ``--spec`` flag is built correctly from the config probe_spec.
  - The proxy base URL is forwarded via ``OPENAI_BASE_URL`` env var.
  - Calling ``run()`` offline returns ``offline=True``.
  - Calling with garak present delegates to a mocked subprocess.
  - Timeout is handled and reported in the result.
* :class:`parse_report`
  - Parses a representative garak JSONL with multiple ``eval`` rows.
  - Aggregates ``total_attempts`` / ``total_failing`` / ``mean_pass_rate``
    / ``overall_pass_rate`` correctly.
  - Captures the ``start_run setup`` metadata row.
  - Skips malformed lines without raising.
  - Skips rows with negative / inconsistent passed/total.
  - Empty probe (total=0) yields pass_rate=0.0 (NOT NaN).
  - Missing file raises FileNotFoundError.
* :func:`establish_baseline`
  - End-to-end: proxy + offline garak + parser + summary JSON write.
  - :meth:`HealerProxy.set_apply_patches(False)` is called.
  - summary JSON is JSON-decodable and contains expected keys.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from healer.config import HealerConfig, Provider, TargetConfig
from healer.llm.proxy import HealerProxy
from healer.llm.target import TargetLLMClient
from healer.probe.baseline import BaselineResult, establish_baseline
from healer.probe.garak_runner import GarakRunner
from healer.probe.result_parser import ProbeResult, RunSummary, parse_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config/proxy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_target_config() -> TargetConfig:
    """Mirror of the proxy test config — local provider forces mock-mode target."""
    return TargetConfig(
        provider=Provider.LOCAL,
        model="local-mock",
        api_base="http://127.0.0.1:1/v1",
        api_key_env="HEALER_PROBE_TEST_KEY_MISSING",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=2,
    )


@pytest.fixture()
def healer_config(local_target_config: TargetConfig) -> HealerConfig:
    cfg = HealerConfig.load(DEFAULT_CONFIG)
    return cfg.model_copy(deep=True)


@pytest.fixture()
def target_client(local_target_config: TargetConfig) -> TargetLLMClient:
    return TargetLLMClient(local_target_config)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEALER_PROBE_TEST_KEY_MISSING", raising=False)


# ===========================================================================
# GarakRunner — offline mode
# ===========================================================================


def test_runner_offline_writes_stub_report(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """When garak is not on PATH the runner writes a synthetic JSONL."""
    runner = GarakRunner(healer_config, proxy_base_url="http://127.0.0.1:8899/v1")
    # Force offline by patching shutil.which to return None.
    with patch("healer.probe.garak_runner.shutil.which", return_value=None):
        result = runner.run(
            report_prefix="test_baseline",
            output_dir=tmp_path,
        )
    assert result.offline is True
    assert result.returncode == -1
    assert result.report_path.is_file()
    # Each line is valid JSON; first line is start_run setup.
    lines = result.report_path.read_text(encoding="utf-8").splitlines()
    assert lines
    rec0 = json.loads(lines[0])
    assert rec0["entry_type"] == "start_run setup"
    assert rec0["plugins.target_type"] == "openai"
    # Each subsequent line is an eval row matching the configured probe_spec.
    expected_probes = healer_config.probe.garak.probe_spec
    assert len(lines) == 1 + len(expected_probes)
    for line, expected in zip(lines[1:], expected_probes, strict=True):
        rec = json.loads(line)
        assert rec["entry_type"] == "eval"
        assert rec["passed"] == 0
        assert rec["total_evaluated"] == 0
        # `expected` is included for future per-probe expectation expansions.
        _ = expected


def test_runner_offline_command_is_captured_in_audit(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """The argv is recorded even in offline mode for audit trail purposes."""
    runner = GarakRunner(healer_config)
    with patch("healer.probe.garak_runner.shutil.which", return_value=None):
        result = runner.run(output_dir=tmp_path)
    assert "garak" in result.command[0]
    assert "--spec" in result.command
    spec_idx = result.command.index("--spec")
    spec_value = result.command[spec_idx + 1]
    assert spec_value.startswith("probes.promptinject")
    assert "probes.encoding" in spec_value
    assert "probes.leakreplay" in spec_value
    assert "--target_type" in result.command
    target_type_idx = result.command.index("--target_type")
    assert result.command[target_type_idx + 1] == "openai"


def test_runner_offline_env_carries_proxy_base_url(
    healer_config: HealerConfig,
) -> None:
    """The proxy base URL is injected via OPENAI_BASE_URL env var."""
    runner = GarakRunner(healer_config, proxy_base_url="http://localhost:9999/v1")
    env = runner._build_env()
    assert env["OPENAI_BASE_URL"] == "http://localhost:9999/v1"
    assert env["OPENAI_API_KEY"]


def test_runner_offline_empty_probe_spec_writes_only_setup_row(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """An empty probe_spec produces only the ``start_run setup`` row."""
    cfg = healer_config.model_copy(deep=True)
    cfg.probe.garak.probe_spec = []
    runner = GarakRunner(cfg)
    with patch("healer.probe.garak_runner.shutil.which", return_value=None):
        result = runner.run(output_dir=tmp_path)
    lines = result.report_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["entry_type"] == "start_run setup"


# ===========================================================================
# GarakRunner — real subprocess path (mocked)
# ===========================================================================


def _fake_completed_process(returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["garak"], returncode=returncode, stdout="garak stdout", stderr=""
    )


def test_runner_subprocess_happy_path(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """When garak IS on PATH the runner shells out and returns its result."""
    runner = GarakRunner(healer_config, proxy_base_url="http://127.0.0.1:9/v1")
    # Pre-create the JSONL file so _find_report succeeds.
    cfg = healer_config
    out_dir = tmp_path
    full_prefix = "test_baseline.deadbeef"

    def _patched_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Simulate garak writing its report.
        report_path = out_dir / f"{full_prefix}.report.jsonl"
        report_path.write_text(
            json.dumps({
                "entry_type": "start_run setup",
                "plugins.target_type": "openai",
                "plugins.target_name": cfg.target.model,
            })
            + "\n"
            + json.dumps({
                "entry_type": "eval",
                "probe": "dan.DanInTheWild",
                "detector": "always.Pass",
                "passed": 5,
                "total_evaluated": 10,
            })
            + "\n",
            encoding="utf-8",
        )
        return _fake_completed_process(returncode=0)

    with (
        patch("healer.probe.garak_runner.shutil.which", return_value="/usr/bin/garak"),
        patch("healer.probe.garak_runner.subprocess.run", side_effect=_patched_run),
        patch("healer.probe.garak_runner.uuid.uuid4") as mock_uuid,
    ):
        mock_uuid.return_value.hex = "deadbeefdeadbeef"
        result = runner.run(report_prefix="test_baseline", output_dir=out_dir)
    assert result.offline is False
    assert result.returncode == 0
    assert result.report_path.is_file()
    assert result.duration_seconds >= 0.0


def test_runner_subprocess_timeout_is_reported(
    healer_config: HealerConfig, tmp_path: Path
) -> None:
    """A garak timeout is captured and surfaced (returncode=-1, not raised)."""
    healer_config.probe.garak.timeout_seconds = 1
    runner = GarakRunner(healer_config)

    def _raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(
            cmd="garak", timeout=1, output=b"o", stderr=b"e"
        )

    with (
        patch("healer.probe.garak_runner.shutil.which", return_value="/usr/bin/garak"),
        patch("healer.probe.garak_runner.subprocess.run", side_effect=_raise_timeout),
        patch("healer.probe.garak_runner.uuid.uuid4") as mock_uuid,
    ):
        mock_uuid.return_value.hex = "deadbeefdeadbeef"
        result = runner.run(output_dir=tmp_path)
    assert result.returncode == -1
    assert result.offline is False
    # stderr_tail carries the captured error text.
    assert isinstance(result.stderr_tail, str)


# ===========================================================================
# result_parser
# ===========================================================================


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    """Helper: dump records as one-per-line JSON."""
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_parser_happy_path(tmp_path: Path) -> None:
    """Parse a representative garak JSONL with the spec's record shapes."""
    jsonl = _write_jsonl(
        tmp_path / "garak.run.report.jsonl",
        [
            {
                "entry_type": "start_run setup",
                "plugins.target_type": "openai",
                "plugins.target_name": "gpt-4o-mini",
                "garak_version": "1.0.0",
            },
            {
                "entry_type": "eval",
                "probe": "dan.DanInTheWild",
                "detector": "always.Pass",
                "passed": 8,
                "total_evaluated": 10,
            },
            {
                "entry_type": "eval",
                "probe": "encoding.DecodeMatch",
                "detector": "always.Pass",
                "passed": 9,
                "total_evaluated": 10,
            },
        ],
    )
    summary = parse_report(jsonl)
    assert summary.records_seen == 3
    assert summary.skipped_records == 0
    assert len(summary.probes) == 2
    assert summary.total_attempts == 20
    assert summary.total_failing == 3
    assert summary.overall_pass_rate == pytest.approx(17 / 20)
    assert summary.mean_pass_rate == pytest.approx((0.8 + 0.9) / 2)


def test_parser_captures_metadata_row(tmp_path: Path) -> None:
    """The ``start_run setup`` row's non-``entry_type`` fields land in metadata."""
    jsonl = _write_jsonl(
        tmp_path / "r.jsonl",
        [
            {
                "entry_type": "start_run setup",
                "plugins.target_type": "openai",
                "plugins.target_name": "gpt-4o-mini",
                "garak_version": "1.0.0",
            },
        ],
    )
    summary = parse_report(jsonl)
    assert "garak_version" in summary.metadata
    assert summary.metadata["plugins.target_name"] == "gpt-4o-mini"


def test_parser_empty_total_evaluated_yields_zero_pass_rate(tmp_path: Path) -> None:
    """``total_evaluated == 0`` produces ``pass_rate = 0.0``, not NaN."""
    jsonl = _write_jsonl(
        tmp_path / "empty.jsonl",
        [
            {"entry_type": "start_run setup", "plugins.target_type": "openai"},
            {
                "entry_type": "eval",
                "probe": "dan.DanInTheWild",
                "detector": "always.Pass",
                "passed": 0,
                "total_evaluated": 0,
            },
        ],
    )
    summary = parse_report(jsonl)
    assert summary.probes[0].pass_rate == 0.0
    import math
    assert not math.isnan(summary.probes[0].pass_rate)  # not NaN
    assert summary.total_attempts == 0
    assert summary.total_failing == 0
    assert summary.overall_pass_rate == 0.0


def test_parser_skips_malformed_lines(tmp_path: Path) -> None:
    """Non-JSON and missing-key rows are skipped, not raised."""
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(
        "\n".join([
            json.dumps({"entry_type": "start_run setup", "plugins.target_type": "openai"}),
            "this is not json",
            json.dumps({"entry_type": "eval", "detector": "always.Pass"}),
            json.dumps({"entry_type": "unknown_type"}),
            json.dumps({
                "entry_type": "eval",
                "probe": "dan.DanInTheWild",
                "detector": "always.Pass",
                "passed": 10,
                "total_evaluated": 10,
            }),
        ]),
        encoding="utf-8",
    )
    summary = parse_report(jsonl)
    assert summary.records_seen == 5
    assert summary.skipped_records == 3
    assert len(summary.probes) == 1
    assert summary.overall_pass_rate == 1.0


def test_parser_negatives_rejected(tmp_path: Path) -> None:
    """Negative or inverted passed/total values are dropped to the skip list."""
    jsonl = _write_jsonl(
        tmp_path / "neg.jsonl",
        [
            {"entry_type": "start_run setup", "plugins.target_type": "openai"},
            {
                "entry_type": "eval", "probe": "a.A", "detector": "d",
                "passed": -1, "total_evaluated": 10,
            },
            {
                "entry_type": "eval", "probe": "b.B", "detector": "d",
                "passed": 11, "total_evaluated": 10,
            },
            {
                "entry_type": "eval", "probe": "c.C", "detector": "d",
                "passed": 5, "total_evaluated": 10,
            },
        ],
    )
    summary = parse_report(jsonl)
    assert len(summary.probes) == 1
    assert summary.probes[0].probe == "c.C"
    assert summary.skipped_records == 2


def test_parser_non_string_detector_defaulted(tmp_path: Path) -> None:
    """A missing/non-string ``detector`` field is replaced with 'unknown'."""
    jsonl = _write_jsonl(
        tmp_path / "no-det.jsonl",
        [
            {"entry_type": "start_run setup", "plugins.target_type": "openai"},
            {
                "entry_type": "eval",
                "probe": "dan.Dan",
                # detector explicitly missing
                "passed": 5, "total_evaluated": 10,
            },
        ],
    )
    summary = parse_report(jsonl)
    assert summary.probes[0].detector == "unknown"


def test_parser_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_report(tmp_path / "does-not-exist.jsonl")


def test_parser_empty_file_yields_empty_summary(tmp_path: Path) -> None:
    """A zero-byte JSONL parses cleanly to an empty summary."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    summary = parse_report(empty)
    assert summary.records_seen == 0
    assert summary.probes == []
    assert summary.overall_pass_rate == 0.0


def test_run_summary_aggregate_idempotent() -> None:
    """Calling aggregate() twice yields the same numbers."""
    summary = RunSummary()
    summary.probes = [
        ProbeResult("a.A", "d", 9, 10, 0.9, 1),
        ProbeResult("b.B", "d", 1, 10, 0.1, 9),
    ]
    summary.aggregate()
    first_total = summary.total_attempts
    first_overall = summary.overall_pass_rate
    summary.aggregate()
    assert summary.total_attempts == first_total
    assert summary.overall_pass_rate == first_overall


def test_run_summary_to_dict_is_json_safe() -> None:
    summary = RunSummary()
    summary.probes = [ProbeResult("a.A", "d", 5, 10, 0.5, 5)]
    summary.aggregate()
    summary.metadata = {"plugins.target_name": "x", "garak_version": "1.0"}
    d = summary.to_dict()
    json.dumps(d)  # raises if not JSON-safe


def test_probe_result_to_dict_shape() -> None:
    p = ProbeResult("p.q", "d", 5, 10, 0.5, 5)
    d = p.to_dict()
    assert set(d.keys()) == {
        "probe", "detector", "passed", "total_evaluated", "pass_rate", "failing"
    }


# ===========================================================================
# establish_baseline — end-to-end with offline garak
# ===========================================================================


def test_establish_baseline_writes_summary(
    healer_config: HealerConfig,
    target_client: TargetLLMClient,
    tmp_path: Path,
) -> None:
    """Paper trail: baseline writes a JSON summary and flips apply_patches off."""
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(True)  # ensure we start ON

    runner = GarakRunner(healer_config)
    with patch("healer.probe.garak_runner.shutil.which", return_value=None):
        result = establish_baseline(
            healer_config, proxy, runner=runner, output_dir=tmp_path
        )

    assert isinstance(result, BaselineResult)
    # Baseline phase MUST disable patches while probing.
    assert proxy.state.apply_patches is False
    assert result.summary_path.is_file()
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "garak_run" in payload
    assert payload["garak_run"]["offline"] is True
    assert payload["garak_run"]["command"]
    # Summary has the per-probe slots the verify phase will diff.
    assert isinstance(payload["summary"]["probes"], list)


def test_establish_baseline_offline_summary_probes_match_config(
    healer_config: HealerConfig,
    target_client: TargetLLMClient,
    tmp_path: Path,
) -> None:
    """The number of summary.probes equals the configured probe_spec length."""
    proxy = HealerProxy(healer_config, target_client)
    runner = GarakRunner(healer_config)
    with patch("healer.probe.garak_runner.shutil.which", return_value=None):
        result = establish_baseline(
            healer_config, proxy, runner=runner, output_dir=tmp_path
        )
    expected = len(healer_config.probe.garak.probe_spec)
    assert len(result.summary.probes) == expected
    for p in result.summary.probes:
        assert p.total_evaluated == 0
        assert p.passed == 0
        assert p.pass_rate == 0.0
