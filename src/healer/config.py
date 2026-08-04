"""Configuration loading and validation for the self-healing pipeline.

This module is the single entry point for reading ``config.yaml``. It parses the
raw YAML into a hierarchy of pydantic models (``HealerConfig``) so that every
other module in the package receives a typed, validated configuration object
instead of ad-hoc dictionaries. Invalid configurations fail fast with
actionable errors before any probe is launched.

Design notes
------------
* The API key never lives in YAML. ``target.api_key_env`` only names *which*
  environment variable to read at runtime; the value is resolved lazily via
  :mod:`healer.llm.target`.
* Enums (``OnMatch``, ``LogLevel``, ``Provider``) make every switchable
  behaviour self-documenting and prevent typos such as ``"ReWrite"``.
* ``HealerConfig.load()`` is the only public surface; submodules import the
  frozen dataclass-like model tree and access fields via attribute syntax.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Module-level logger so validation-time warnings use an own logger rather
# than the root logger (ruff LOG015).
_LOG = logging.getLogger(__name__)

# Resolve the project root once: this file lives at
# ``src/healer/config.py``, so the repo root is three levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Enums — switchable, self-documenting behaviour
# ---------------------------------------------------------------------------


class LogLevel(str, Enum):
    """Standard logging levels, exposed as strings for YAML readability."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Provider(str, Enum):
    """Supported backend providers for the target LLM."""

    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


class OnMatch(str, Enum):
    """Action taken by a mitigator when a signature / keyword matches.

    Attributes:
        BLOCK: Reject the request outright with a hard error.
        REWRITE: Substitute the offending input with a safe template.
        FALLBACK: Replace the unsafe model output with a canned message.
        REDACT: Mask the offending substrings in the model output.
        LOG_ONLY: Record the match but let traffic pass unchanged (audit mode).
    """

    BLOCK = "block"
    REWRITE = "rewrite"
    FALLBACK = "fallback"
    REDACT = "redact"
    LOG_ONLY = "log_only"


# ---------------------------------------------------------------------------
# Model tree — mirrors the schema of config.yaml one-to-one.
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata and global knobs."""

    name: str
    description: str = ""
    seed: int = 42
    output_dir: Path = Path("data/reports")
    log_level: LogLevel = LogLevel.INFO

    model_config = {"extra": "forbid"}


class TargetConfig(BaseModel):
    """Connection parameters for the system-under-test LLM.

    ``api_key_env`` stores the *name* of an environment variable rather than
    the key itself, keeping secrets out of the repo.
    """

    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    timeout_seconds: int = Field(default=30, gt=0)

    model_config = {"extra": "forbid"}


class ProxyConfig(BaseModel):
    """Local OpenAI-compatible proxy that wraps the target with mitigations."""

    host: str = "127.0.0.1"
    port: int = Field(default=8899, gt=0, lt=65536)
    apply_patches: bool = True

    model_config = {"extra": "forbid"}


class GarakConfig(BaseModel):
    """Parameters forwarded to the garak CLI subprocess."""

    version: str = "0.9.0.13"
    executable: str = "garak"
    probe_spec: list[str] = Field(default_factory=list)
    generations: int = Field(default=1, gt=0)
    seed: int = 42
    timeout_seconds: int = Field(default=1800, gt=0)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("probe_spec")
    @classmethod
    def _non_empty_when_used(cls, v: list[str]) -> list[str]:
        # Empty probe_spec is allowed only if the user overrides it in verify;
        # nevertheless we warn loudly at validation time if completely empty.
        if not v:
            _LOG.warning(
                "probe.garak.probe_spec is empty — no probes will run. "
                "Populate it (or verify.probe_spec_override) before executing "
                "the pipeline."
            )
        return v

    model_config = {"extra": "forbid"}


class ProbeConfig(BaseModel):
    """Probe-layer settings consumed by :mod:`healer.probe`."""

    garak: GarakConfig = Field(default_factory=GarakConfig)
    report_prefix: str = "run"

    model_config = {"extra": "forbid"}


class DetectInjectionConfig(BaseModel):
    """Input-side injection detector parameters."""

    enabled: bool = True
    on_match: OnMatch = OnMatch.REWRITE
    patterns_file: Path = Path("data/prompts/injection_patterns.yaml")
    rewrite_template: str = Field(default="")

    model_config = {"extra": "forbid"}


class RateLimitConfig(BaseModel):
    """Optional request-rate limiter for the prompt patch layer."""

    enabled: bool = False
    max_requests_per_minute: int = Field(default=60, gt=0)

    model_config = {"extra": "forbid"}


class PromptPatchStrategies(BaseModel):
    """Container for prompt-side mitigations — matches YAML ``strategies`` key.

    The strategies dict separates mitigation *toggle* (``prompt_patch.enabled``)
    from the per-mechanism settings so policy analysts can disable individual
    guards without touching the layer flag.
    """

    detect_injection: DetectInjectionConfig = Field(
        default_factory=DetectInjectionConfig
    )
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    model_config = {"extra": "forbid"}


class PromptPatchConfig(BaseModel):
    """Aggregated prompt-level mitigation configuration.

    The ``strategies`` sub-key matches the YAML schema exactly: every
    mitigation mechanism sits under ``patch.prompt_patch.strategies.*``.
    """

    enabled: bool = True
    strategies: PromptPatchStrategies = Field(
        default_factory=PromptPatchStrategies
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_template_when_rewrite(self) -> PromptPatchConfig:
        """Force a non-empty rewrite template when ``on_match == rewrite``."""
        di = self.strategies.detect_injection
        if (
            di.enabled
            and di.on_match == OnMatch.REWRITE
            and not di.rewrite_template.strip()
        ):
            raise ValueError(
                "patch.prompt_patch.strategies.detect_injection.rewrite_template "
                "must be non-empty when on_match is 'rewrite'."
            )
        return self


class PolicyFilterConfig(BaseModel):
    """Output-side keyword policy filter."""

    enabled: bool = True
    keywords_file: Path = Path("data/prompts/policy_keywords.yaml")
    categories: list[str] = Field(default_factory=list)
    on_match: OnMatch = OnMatch.FALLBACK
    fallback_message: str = Field(default="")

    model_config = {"extra": "forbid"}


class MaxLengthConfig(BaseModel):
    """Cap on response length, enforced after policy filtering."""

    enabled: bool = True
    max_tokens: int = Field(default=500, gt=0)

    model_config = {"extra": "forbid"}


class OutputPatchStrategies(BaseModel):
    """Container for output-side mitigations — matches YAML ``strategies`` key."""

    policy_filter: PolicyFilterConfig = Field(default_factory=PolicyFilterConfig)
    max_length: MaxLengthConfig = Field(default_factory=MaxLengthConfig)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_fallback_message(self) -> OutputPatchStrategies:
        """Force a non-empty fallback message when ``on_match == fallback``."""
        pf = self.policy_filter
        if (
            pf.enabled
            and pf.on_match == OnMatch.FALLBACK
            and not pf.fallback_message.strip()
        ):
            raise ValueError(
                "patch.output_patch.strategies.policy_filter.fallback_message "
                "must be non-empty when on_match is 'fallback'."
            )
        return self


class OutputPatchConfig(BaseModel):
    """Aggregated output-level mitigation configuration."""

    enabled: bool = True
    strategies: OutputPatchStrategies = Field(
        default_factory=OutputPatchStrategies
    )

    model_config = {"extra": "forbid"}


class PatchConfig(BaseModel):
    """Root patch-layer configuration."""

    prompt_patch: PromptPatchConfig = Field(default_factory=PromptPatchConfig)
    output_patch: OutputPatchConfig = Field(default_factory=OutputPatchConfig)

    model_config = {"extra": "forbid"}


class SuccessCriteriaConfig(BaseModel):
    """Gates that declare the pipeline 'healed' after verification."""

    max_failure_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    min_relative_improvement: float = Field(default=0.50, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class VerifyConfig(BaseModel):
    """Verify-layer configuration."""

    enabled: bool = True
    reuse_probe_spec: bool = True
    probe_spec_override: list[str] = Field(default_factory=list)
    generations: int = Field(default=1, gt=0)
    success_criteria: SuccessCriteriaConfig = Field(
        default_factory=SuccessCriteriaConfig
    )
    report_prefix: str = "run_verify"

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_override_when_not_reused(self) -> VerifyConfig:
        """Require an explicit override list when reuse_probe_spec is False."""
        if not self.reuse_probe_spec and not self.probe_spec_override:
            raise ValueError(
                "verify.probe_spec_override must be non-empty when "
                "reuse_probe_spec is false."
            )
        return self


class ReportsConfig(BaseModel):
    """Which report artifacts to emit for the audit trail."""

    emit_jsonl: bool = True
    emit_summary_json: bool = True
    emit_markdown: bool = True
    markdown_path: Path = Path("data/reports/experiment_summary.md")

    model_config = {"extra": "forbid"}


class HealerConfig(BaseModel):
    """Top-level configuration tree, mirroring ``config.yaml``."""

    experiment: ExperimentConfig
    target: TargetConfig = Field(default_factory=TargetConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> HealerConfig:
        """Load and validate a configuration file.

        Args:
            path: Optional path to a YAML config file. When ``None`` the
                default ``config.yaml`` at the project root is used.

        Returns:
            A fully validated :class:`HealerConfig` instance.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            pydantic.ValidationError: The YAML is structurally or semantically
                invalid. The full list of failing fields is included.
        """
        cfg_path: Path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {cfg_path}"
            )

        with cfg_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        return cls.model_validate(raw)

    def with_patches(self, enabled: bool) -> HealerConfig:
        """Return a shallow copy with ``proxy.apply_patches`` toggled.

        Used by the orchestrator to flip between the baseline run
        (``enabled=False``) and the verification run (``enabled=True``)
        without re-reading the YAML file.

        Args:
            enabled: New value for ``proxy.apply_patches``.

        Returns:
            A new :class:`HealerConfig` instance; the original is untouched.
        """
        new = self.model_copy(deep=True)
        new.proxy.apply_patches = enabled
        return new


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "DetectInjectionConfig",
    "ExperimentConfig",
    "GarakConfig",
    "HealerConfig",
    "LogLevel",
    "MaxLengthConfig",
    "OnMatch",
    "OutputPatchConfig",
    "OutputPatchStrategies",
    "PatchConfig",
    "PolicyFilterConfig",
    "ProbeConfig",
    "PromptPatchConfig",
    "PromptPatchStrategies",
    "Provider",
    "ProxyConfig",
    "RateLimitConfig",
    "ReportsConfig",
    "SuccessCriteriaConfig",
    "TargetConfig",
    "VerifyConfig",
]
