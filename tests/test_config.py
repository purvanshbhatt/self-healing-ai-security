"""Smoke test: config.yaml loads and validates cleanly."""

from __future__ import annotations

from pathlib import Path

import pytest

from healer.config import (
    HealerConfig,
    LogLevel,
    OnMatch,
    OutputPatchStrategies,
    PromptPatchStrategies,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_loads_from_default() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    assert isinstance(cfg, HealerConfig)


def test_experiment_metadata_parsed() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    assert cfg.experiment.name == "self_healing_poc_v1"
    assert cfg.experiment.log_level == LogLevel.INFO
    assert cfg.experiment.seed == 42


def test_target_provider_and_model() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    assert cfg.target.model == "gpt-4o-mini"
    assert cfg.target.api_key_env == "OPENAI_API_KEY"
    assert cfg.target.temperature == 0.0


def test_proxy_apply_patches_default_true() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    assert cfg.proxy.apply_patches is True


def test_garak_probe_spec_parsed() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    assert "promptinject" in cfg.probe.garak.probe_spec
    assert cfg.probe.garak.generations == 1


def test_strategies_nested_correctly() -> None:
    """Verify the YAML `strategies:` nesting round-trips through pydantic."""
    cfg = HealerConfig.load(CONFIG_PATH)
    assert isinstance(cfg.patch.prompt_patch.strategies, PromptPatchStrategies)
    assert isinstance(cfg.patch.output_patch.strategies, OutputPatchStrategies)
    # on_match values parse into enums, not raw strings.
    assert (
        cfg.patch.prompt_patch.strategies.detect_injection.on_match
        == OnMatch.REWRITE
    )
    assert (
        cfg.patch.output_patch.strategies.policy_filter.on_match == OnMatch.FALLBACK
    )


def test_with_patches_toggles_without_mutating_original() -> None:
    cfg = HealerConfig.load(CONFIG_PATH)
    baseline = cfg.with_patches(enabled=False)
    assert baseline.proxy.apply_patches is False
    # Original untouched (important for orchestrator in-place reuse).
    assert cfg.proxy.apply_patches is True


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        HealerConfig.load("/nonexistent/config.yaml")


def test_extra_keys_rejected() -> None:
    """`extra=forbid` should make unknown keys fail validation."""
    import yaml
    from pydantic import ValidationError

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["experiment"]["rogue_field"] = True
    with pytest.raises(ValidationError):
        HealerConfig.model_validate(raw)
