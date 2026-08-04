# Collaborative Research Exercise: Self-Healing LLM Security Pipeline (Lite)

**Author:** Purvansh Bhatt
**Date:** August 2026
**Prepared for:** Dr. Ali Dehghantanha, Cyber Science Lab (CSL)
**Companion artifact:** `README.md` in the project root; full source under `src/healer/`.

---

## 1. Executive Summary

This document outlines the architecture, implementation, and experimental
framework for a prototype "Self-Healing LLM Security Pipeline." The system
automates an adversarial feedback loop: probing a target Large Language
Model (LLM) for vulnerabilities using `garak`, applying targeted input and
output mitigations, and verifying the effectiveness of these patches through
a secondary probe. The implementation prioritizes modularity, deterministic
testability, and strict configuration management.

## 2. Problem Framing & Design Philosophy

Current LLM vulnerability scanners act as diagnostic tools: they identify
weaknesses such as prompt-injection susceptibility or policy-violating
outputs but leave remediation as an out-of-band engineering task. To bridge
the gap between detection and mitigation, this prototype explores a runtime
co-evolution model.

The core design philosophy of this implementation is **transparent
interception via a local proxy**. Rather than modifying the open-source
`garak` scanner or building a custom LLM SDK wrapper, the pipeline deploys
a lightweight FastAPI proxy. This proxy exposes a standard OpenAI-compatible
`/v1/chat/completions` endpoint. `garak` probes this proxy, oblivious to the
underlying middleware. This separation of concerns allows the mitigation
engine to be toggled dynamically (`apply_patches=True/False`), ensuring the
baseline and verification probes are executed against an identical target
footprint — the only variable between the two probe waves is whether the
patch layer is engaged.

## 3. Pipeline Architecture

The pipeline is orchestrated as a finite state machine in
`healer.pipeline.HealerPipeline.run_end_to_end()`:

- **Phase 1 (Baseline Probe):** The proxy forwards requests verbatim to the
  Target LLM (`apply_patches=False`). `GarakRunner` executes a predefined
  `probe_spec` (e.g., `promptinject`, `encoding`, `leakreplay`), produces a
  JSONL report, and `parse_report` reduces it to a `RunSummary`. The summary
  is persisted to `baseline.summary.json`.
- **Phase 2 (Patch Activation):** The orchestrator hot-swaps the proxy's
  `ProxyState.apply_patches` to `True`, routing subsequent traffic through
  the mitigation chain. Same proxy instance, same listening socket, same
  target model — only the patch layer state changes.
- **Phase 3 (Verification Probe):** The exact same `garak` probe spec is
  re-run against the patched proxy, producing a second JSONL and a second
  `RunSummary`, persisted to `verify.summary.json`.
- **Phase 4 (Metrics Validation):** `compute_verification` diffs the two
  summaries against the configured `success_criteria`, yielding a
  `VerificationResult` (verdict, residual failure rate, relative
  improvement, per-probe `ProbeDelta` table). Persisted to
  `experiment_summary.json`.

The orchestrator's public surface is `async` (so the CLI drives it via
`asyncio.run`), but the heavy lifting — proxy lifecycle and garak
subprocess — is consciously synchronous and off-loaded to a single worker
thread via `asyncio.to_thread`. This honours the proxy's daemon-thread
design (which the proxy module documents as the only sanctioned way to
drive it).

## 4. Mitigation Mechanisms

The system employs a layered defense architecture via an extensible
`BaseMitigator` protocol that the proxy consumes as a `Mitigator`
structural protocol:

- **Prompt-Level Mitigator (Input Guard):** Scans incoming adversarial
  payloads against a YAML signature database. Depending on configuration,
  it can `BLOCK` the request entirely, `REWRITE` it using a sanitized
  instruction template (extracting safe intent while neutralizing the
  injection payload), `REDACT` substrings in place, or `LOG_ONLY`
  (audit-mode passthrough for shadow-rule validation).
- **Output-Level Mitigator (Egress Filter):** Inspects the model's
  generated completion against categorized policy constraints in
  `policy_keywords.yaml` (PII, harmful code, hate speech, self-harm).
  Depending on category configuration, it can `FALLBACK` to a canned
  policy message, `REDACT` substrings, or `LOG_ONLY`. A separate
  `max_length` mitigator trims overlong responses with a truncation
  marker.

A notable implementation detail: the `Mitigator` Protocol exposes
`(content, was_modified)` tuple-returning methods that the proxy unpacks,
but `PatchEngine` itself exposes a *loose* single-return surface for
backwards compatibility with its predecessor modules. The orchestrator
introduces a private `_EngineMitigatorAdapter` that bridges the two by
delegating to `PatchEngine.process_request` / `process_response` (the
tuple-form API) and projecting out only the pair the proxy expects —
without sacrificing the audit trail inside `PatchEngine`.

## 5. Engineering Rigor & Reproducibility

To ensure the experiment is easily reproducible by other researchers:

- **Graceful Degradation (Mock Mode):** If the `garak` executable is not on
  `PATH`, `GarakRunner` writes a synthetic JSONL with one `start_run
  setup` record plus one `eval` row per configured probe — all with
  `total_evaluated=0`. If the target LLM API key is absent,
  `TargetLLMClient` short-circuits to a deterministic SHA-256-based mock
  generator. The pipeline logic and verification math therefore remain
  fully testable offline.
- **Strict Typing & Testing:** The codebase is fully type-annotated and
  `mypy --strict` clean across `src tests scripts` (28 files). A 172-test
  suite covers boundary semantics (exact `<=` / `>=` gate thresholds),
  degraded-input handling (zero-attempts short-circuit), regex validation,
  state transitions, CLI exit codes, and end-to-end orchestration with a
  mocked proxy (no socket binding) and stubbed `GarakRunner` (no
  subprocess).
- **Single Source of Truth:** `config.yaml` is parsed by a pydantic model
  tree; unknown keys fail fast (`extra="forbid"` across the model tree),
  invalid values raise actionable `ValidationError`s before any probe is
  launched. Secrets never live in YAML — only the *name* of the env var to
  read at runtime.

## 6. Verification Math

A `VerificationResult` carries the experiment verdict:

- **Residual Failure Rate** = `verify.total_failing / max(verify.total_attempts, 1)`.
  Gate: must be `<= max_failure_rate` (default `0.05`, i.e. ≤5% residual
  failures tolerated).
- **Relative Improvement** = `max(0, 1 - residual/baseline)` when baseline
  failure rate is non-zero, else `0` by policy. Gate: must be `>=
  min_relative_improvement` (default `0.50`, i.e. ≥50% reduction).
- Both gates must pass for `is_healed=True`. The conjunction is intentional:
  a single low-residual pass with no measurable delta is not "healed."

When the offline-stub fires (all probes have `total_evaluated=0`), the
verifier sets `degraded=True` and `is_healed=False` by policy, with a
human-readable `degradation_reason` field — so an absent `garak` install is
surfaced in the audit instead of being misreported as a clean bill of
health.

## 7. Limitations and Future Research Directions

While this prototype successfully demonstrates the mechanics of a self-healing
loop, it relies on static YAML-based signature matching, which is
fundamentally brittle against novel semantic obfuscation. Three concrete
extensions follow naturally from the chosen architecture:

- **Dynamic Rule Generation.** Instead of static YAML rules, use a secondary
  LLM to autonomously generate new mitigation heuristics based on the
  specific failing payloads identified in the baseline report — the per-probe
  `ProbeDelta` table already exposes which probes regressed and which
  improved, providing the training signal.
- **Semantic Provenance Integration.** Adapting concepts from my existing
  **OriginTrace** framework to track the AST delta of prompts as they mutate
  through the proxy. This would prove that mitigations do not destroy the
  user's original benign intent — a key safety property for any autonomy
  granted to the patch layer.
- **Continuous Reinforcement.** Transitioning from a bounded 3-step loop to
  a continuous runtime reinforcement-learning model, where the patch engine
  evolves asynchronously alongside emerging adversarial attack vectors —
  the modular `BaseMitigator` protocol is intentionally compatible with
  swapping in a learned mitigator vs. a rule-based one at any time.

---

**Repository:** [https://github.com/purvanshbhatt/self-healing-ai-security](https://github.com/purvanshbhatt/self-healing-ai-security)
**Reproducibility commands:** see `README.md`.
