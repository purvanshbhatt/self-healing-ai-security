# Self-Healing LLM Security Pipeline (Prototype)

A proof-of-concept research framework that implements an autonomous
`Probe → Patch → Verify` feedback loop for Large Language Models. Built as a
collaborative technical exercise for the Cyber Science Lab (CSL).

## Overview

This pipeline evaluates an LLM's security posture using
[garak](https://github.com/leondz/garak) (an LLM vulnerability scanner),
dynamically deploys input/output mitigations via a transparent local proxy,
and verifies the effectiveness of those patches by re-running the probes and
calculating improvement metrics.

### Key Features

- **Transparent Proxy Architecture** — Intercepts `garak` traffic via a local
  OpenAI-compatible FastAPI server. `garak` is unmodified; the target LLM SDK
  is unmodified. The proxy is the only insertion point.
- **Layered Mitigations** — Pluggable `PromptMitigator` (input sanitization /
  rewriting / blocking) and `OutputMitigator` (policy violation fallback /
  redaction / length caps), both driven by an abstract `BaseMitigator`
  protocol so new classifiers can be plugged in without touching the proxy.
- **Deterministic Verification Math** — Computes Residual Failure Rates and
  Relative Improvement Ratios against configurable thresholds
  (`max_failure_rate`, `min_relative_improvement`) to programmatically decide
  whether the system is "healed." Both gates must pass.
- **Graceful Offline Mode** — Runs in CI/CD without active API keys or a
  local `garak` install: the target LLM falls back to a deterministic
  hash-based mock generator, and `GarakRunner` writes a synthetic JSONL with
  zero-attempt rows. The verification math is exercised end-to-end.
- **Strict Engineering** — 172 unit/integration tests, `mypy --strict`
  compliance across `src tests scripts`, and `ruff` lint clean.

## Architecture

```
         garak probe wave                                    target LLM
               │                                                ▲
               ▼                                                │
   ┌──────────────────────┐    apply_patches=False/True   ┌────┴───────┐
   │  local FastAPI proxy  │ ──────────────────────────► │   TargetLLM  │
   │  /v1/chat/completions │ ◄─────────────────────────  │    Client     │
   └──────────┬───────────┘                                └──────────────┘
              │                       ▲
              ▼                       │
   ┌──────────────────────┐   PatchEngine.process_request / process_response
   │   Mitigation chain    │ ──► PromptMitigator → OutputMitigator
   │  (BaseMitigator impls)│
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │  GarakRunner + Parser│ ──► RunSummary (baseline) ──┐
   │                      │     RunSummary (verify)   ──┤
   └──────────────────────┘                            ▼
                                              ┌──────────────────┐
                                              │ compute_verification│
                                              │  → VerificationResult│
                                              └──────────────────┘
```

The orchestrator (`healer.pipeline.HealerPipeline`) drives four phases:

1. **Baseline Probe** — proxy started with `apply_patches=False`; `garak`
   runs against the raw target; JSONL parsed into a `RunSummary`; persisted
   to `baseline.summary.json`.
2. **Patch Activation** — the proxy's `ProxyState.apply_patches` is flipped
   to `True` in-place; the same proxy instance keeps serving.
3. **Verification Probe** — `garak` runs the *same* probe spec against the
   patched path; JSONL parsed into a second `RunSummary`; persisted to
   `verify.summary.json`.
4. **Metrics Validation** — `compute_verification` diffs the two summaries
   against the configured `success_criteria`, emitting a `VerificationResult`
   with `is_healed`, `residual_failure_rate`, `relative_improvement`, and a
   per-probe `ProbeDelta` table. Persisted to `experiment_summary.json`.

## Installation & Setup

Requires **Python 3.10+**.

```powershell
# Clone the repository
git clone https://github.com/purvanshbhatt/self-healing-ai-security.git
cd self-healing-ai-security

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # Linux / macOS

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

(Optional) Install `garak` globally if you wish to run live adversarial
probes. If the `garak` executable is not on `PATH`, the pipeline falls back
to a synthetic offline-stub JSONL so the verify math stays testable.

## Configuration

All pipeline behaviors, target LLM configurations, and mitigation thresholds
are controlled by a single `config.yaml` validated through `pydantic`
(unknown keys are rejected; invalid values fail fast with an actionable
`ValidationError`).

- Inject custom API keys by defining them in a `.env` file (see
  `.env.example`). The YAML only names *which* env var to read at runtime;
  secret values never live in the repo.
- Modify mitigation rules in `data/prompts/injection_patterns.yaml`
  (input-side signature database) and `data/prompts/policy_keywords.yaml`
  (output-side policy buckets).
- Tune the verification thresholds in `config.yaml`'s `verify.success_criteria`
  block: `max_failure_rate` (default `0.05`) and
  `min_relative_improvement` (default `0.50`).

## Running the Experiment

To execute the full end-to-end `Probe → Patch → Verify` loop:

```powershell
python scripts/run_experiment.py --config config.yaml
python scripts/run_experiment.py --config config.yaml --output-dir data/reports/experiment_01
```

### Output Artifacts

All generated artifacts land in `config.experiment.output_dir`
(default `data/reports/`):

- `run_baseline.<uuid>.report.jsonl` — raw garak JSONL from the baseline run.
- `run_verify.<uuid>.report.jsonl` — raw garak JSONL from the verify run.
- `baseline.summary.json` — aggregated `RunSummary` for the baseline.
- `verify.summary.json` — aggregated `RunSummary` for the verify run.
- `experiment_summary.json` — top-level bundle with both per-phase summaries,
  the `VerificationResult`, and per-phase wall-clock durations. This is the
  single artifact to read for the experiment verdict.

### CLI Exit Codes

- `0` — pipeline ran to completion (regardless of the `is_healed` verdict;
  a "not healed" experiment is still a successful experiment).
- `1` — configuration did not load (file missing or `ValidationError`).
- `2` — pipeline raised at runtime (garak subprocess failed, proxy didn't
  bind its port, etc.). Traceback is written to stderr.

## Testing & Quality Gates

This project enforces strict engineering standards: 172 unit/integration
tests, strict type-checking, and static analysis. Reproducible from the
project root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src tests scripts
```

### Test Coverage Breakdown

| Module | Tests |
|---|---|
| `healer.config` | 9 |
| `healer.llm.target` | 22 |
| `healer.llm.proxy` | 25 |
| `healer.patch.*` | 47 |
| `healer.probe.*` | 19 |
| `healer.verify.*` | 35 |
| `healer.pipeline` (+ CLI) | 15 |
| **Total** | **172** |

## Project Layout

```
self-healing-ai-security/
├── config.yaml                  # Single source of truth, validated by pydantic
├── pyproject.toml               # Package metadata, ruff/mypy/pytest config
├── requirements.txt
├── .env.example                 # OPENAI_API_KEY wiring for live runs
├── data/
│   └── prompts/                 # Injection signatures + policy keyword buckets
├── docs/
│   └── TECHNICAL_BRIEF.md       # 2-page architecture brief for Dr. Ali
├── scripts/
│   └── run_experiment.py        # CLI entry point (--config / --output-dir)
├── src/healer/
│   ├── config.py                # Pydantic model tree + HealerConfig.load()
│   ├── logging_setup.py         # Idempotent root logger configuration
│   ├── llm/
│   │   ├── target.py            # Async OpenAI-compatible client + mock fallback
│   │   └── proxy.py             # FastAPI proxy + Mitigator Protocol
│   ├── patch/
│   │   ├── base.py              # BaseMitigator + PatchResult + FiredRule
│   │   ├── prompt_patch.py     # Input-side mitigator (BLOCK/REWRITE/LOG_ONLY)
│   │   ├── output_patch.py     # Output-side mitigator (FALLBACK/REDACT/MAX_LEN)
│   │   └── engine.py           # Chained mitigator composition + EngineAudit
│   ├── probe/
│   │   ├── garak_runner.py     # Subprocess wrapper + offline-stub JSONL
│   │   ├── result_parser.py    # JSONL → RunSummary / ProbeResult
│   │   └── baseline.py         # Baseline phase: patches-off + parse + persist
│   ├── verify/
│   │   └── metrics.py          # compute_verification + ProbeDelta + RunSummary diff
│   └── pipeline.py             # HealerPipeline.run_end_to_end() orchestrator
└── tests/                       # 172 tests across 6 modules
    ├── conftest.py
    ├── test_config.py
    ├── test_target_llm.py
    ├── test_proxy.py
    ├── test_patch.py
    ├── test_probe.py
    ├── test_verify.py
    └── test_pipeline.py
```

## Author

**Purvansh Bhatt** — Applied AI Security Researcher
Portfolio: https://purvanshbhatt.github.io  ·  LinkedIn: https://linkedin.com/in/purvanshbhatt

## Acknowledgements

Prepared as a collaborative technical exercise for **Dr. Ali
Dehghantanha**, Cyber Science Lab (CSL). The pipeline's design philosophy
extends prior work on runtime telemetry validation and supply-chain semantic
provenance (ResilAI, OriginTrace) to the runtime behaviour security domain
of Large Language Models.
