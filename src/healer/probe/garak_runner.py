"""Subprocess wrapper around the ``garak`` LLM vulnerability scanner CLI.

This module exposes :class:`GarakRunner`, the single entry point the pipeline
orchestrator uses to launch garak probe batches. It hides CLI-flag translation
(``--probes`` is deprecated, ``--spec`` is the modern flag) and produces a
deterministic JSONL output path that :mod:`healer.probe.result_parser` consumes.

Offline mode
------------
If the ``garak`` executable declared in :attr:`ProbeConfig.garak.executable`
isn't on ``$PATH`` (or install is still in progress, e.g. on a fresh CI
worker), :class:`GarakRunner` degrades to **offline mode**: it writes a
synthetic JSONL report with all-zero attempts and zero passes. This mirrors
the philosophy of :class:`healer.llm.target.TargetLLMClient.mock` — the
pipeline can run end-to-end in a CI box that doesn't have garak installed,
exercising every code path (proxy, patches, parser, verifier, report
generator) without the ~3-5 minute garak install step. The subnet result is
that the *baseline* shows zero vulnerability data and the *verify* delta
also reads as zero — which is exactly the*boring*, safe default for an
audit pipeline that finds no probes to enumerate when garak is absent.

CLI flag translation
--------------------
The user's :attr:`probe_spec` is a list of bare probe names like
``["promptinject", "encoding", "leakreplay"]``. They get prefixed with the
``probes.`` namespace and joined into the modern ``--spec`` flag value
(``"probes.promptinject,probes.encoding,probes.leakreplay"``). We never
emit the deprecated ``--probes`` flag.

OpenAI-compatible target via env var
------------------------------------
We don't ship a per-run garak YAML config — the parse cost is non-trivial
and our proxy is already OpenAI-compatible. Instead we set the
``OPENAI_BASE_URL`` env-var to point at our local proxy and use
``--target_type openai --target_name <model>``. The SDK reads the key from
``OPENAI_API_KEY`` (we set a sentinel value, because the proxy doesn't
auth-check; the upstream real-key is honoured by :class:`TargetLLMClient`,
not garak).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from healer.config import HealerConfig
from healer.logging_setup import get_logger

_LOG: Final[logging.Logger] = get_logger(__name__)

# Sentinel API key written into garak's environment. The local proxy doesn't
# verify it; we just need `garak`'s OpenAI generator to send *something* so
# it doesn't bail with "OPENAI_API_KEY unset".
_PROXY_SENTINEL_KEY: Final[str] = "healer-proxy-no-auth"

# Per-probe namespace prefix used in modern --spec values.
_PROBE_NS: Final[str] = "probes."


@dataclass(frozen=True, slots=True)
class GarakRunResult:
    """Outcome of one :meth:`GarakRunner.run` call.

    Attributes:
        report_path: Filesystem path to the JSONL report produced by garak
            (or the synthetic offline stub). Always exists when
            ``returncode == 0`` or when ``offline`` is True.
        returncode: Garak subprocess exit code. ``-1`` in offline mode.
        offline: True iff the offline stub was written because garak is
            unavailable on this host.
        duration_seconds: Wall-clock duration of the subprocess call.
        command: Full argv list that was (or would have been) executed,
            recorded for the audit log even in offline mode.
        stderr_tail: Last ~2 KiB of stderr output, useful when garak
            errors mid-run.
    """

    report_path: Path
    returncode: int
    offline: bool
    duration_seconds: float
    command: tuple[str, ...]
    stderr_tail: str = ""


class GarakRunner:
    """Drives the garak CLI as a subprocess and routes output to JSONL."""

    def __init__(
        self,
        config: HealerConfig,
        *,
        proxy_base_url: str | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            config: The fully validated :class:`HealerConfig`. The
                :attr:`probe.garak` block drives all CLI flags.
            proxy_base_url: Optional explicit base URL to the local
                OpenAI-compatible proxy. Defaults to
                ``http://{config.proxy.host}:{config.proxy.port}/v1``. Passed
                through to garak via the ``OPENAI_BASE_URL`` env var.
        """
        self._config: HealerConfig = config
        cfg = config.probe.garak
        if not cfg.probe_spec:
            _LOG.warning(
                "GarakRunner initialised with empty probe_spec — no probes "
                "will run."
            )
        if proxy_base_url is None:
            proxy_base_url = (
                f"http://{config.proxy.host}:{config.proxy.port}/v1"
            )
        self._proxy_base_url: str = proxy_base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        report_prefix: str | None = None,
        output_dir: Path | None = None,
    ) -> GarakRunResult:
        """Launch a single garak probe session.

        Args:
            report_prefix: Override the configured ``report_prefix`` so the
                orchestrator can write ``baseline`` vs ``verify`` to the same
                directory without collision.
            output_dir: Override the experiment ``output_dir``.

        Returns:
            :class:`GarakRunResult` describing where the JSONL landed, the
            process exit code, and a few audit odds-and-ends.
        """
        prefix = report_prefix or self._config.probe.report_prefix
        out_dir = Path(output_dir or self._config.experiment.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # garak appends ``.<uuid>.report.jsonl`` to the prefix it gets, so
        # we feed it a deterministic-ish prefix and then find the actual
        # JSONL on disk by globbing.
        full_prefix = out_dir / f"{prefix}.{uuid.uuid4().hex[:8]}"
        jsonl_path = full_prefix.with_suffix(".report.jsonl")

        argv = self._build_argv(prefix=str(full_prefix))
        env = self._build_env()

        executable = self._config.probe.garak.executable
        if not shutil.which(executable):
            _LOG.warning(
                "garak executable %r not found on PATH — writing an offline "
                "stub report instead. Install garak to enable real probing.",
                executable,
            )
            return self._offline_stub(argv, jsonl_path)

        t0 = time.perf_counter()
        _LOG.info("Launching garak: %s", " ".join(argv))
        try:
            proc = subprocess.run(
                list(argv),
                env=env,
                capture_output=True,
                text=True,
                timeout=self._config.probe.garak.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _LOG.error(
                "garak timed out after %ds. stdout tail=%r, stderr tail=%r",
                self._config.probe.garak.timeout_seconds,
                (exc.stdout or b"")[-2048:],
                (exc.stderr or b"")[-2048:],
            )
            return GarakRunResult(
                report_path=jsonl_path,
                returncode=-1,
                offline=False,
                duration_seconds=self._config.probe.garak.timeout_seconds,
                command=tuple(argv),
                stderr_tail=(exc.stderr or b"").decode("utf-8", "replace")[-2048:]
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")[-2048:],
            )

        duration = time.perf_counter() - t0
        actual = self._find_report(out_dir, prefix=full_prefix.name)
        return GarakRunResult(
            report_path=actual,
            returncode=proc.returncode,
            offline=False,
            duration_seconds=duration,
            command=tuple(argv),
            stderr_tail=(proc.stderr or "")[-2048:],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_argv(self, *, prefix: str) -> tuple[str, ...]:
        """Translate :class:`GarakConfig` into the modern garak CLI argv.

        Returns a tuple (not a list) for hashability in logs and tests.
        """
        cfg = self._config.probe.garak
        target_model = self._config.target.model

        spec_value = ",".join(
            f"{_PROBE_NS}{p}" for p in cfg.probe_spec
        )
        argv: list[str] = [
            cfg.executable,
            "--target_type", "openai",
            "--target_name", target_model,
            "--spec", spec_value,
            "--generations", str(cfg.generations),
            "--seed", str(cfg.seed),
            "--report_prefix", prefix,
        ]
        argv.extend(cfg.extra_args)
        return tuple(argv)

    def _build_env(self) -> dict[str, str]:
        """Env for the garak subprocess: carries its own OPENAI_BASE_URL."""
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = self._proxy_base_url
        env["OPENAI_API_KEY"] = _PROXY_SENTINEL_KEY
        return env

    def _find_report(self, out_dir: Path, *, prefix: str) -> Path:
        """Glob the output dir for the JSONL garak actually wrote.

        garak appends ``.<run_uuid>.report.jsonl`` to the supplied prefix
        when it doesn't end in ``.jsonl``. We honour either layout so the
        runner is robust to CLI flag changes in future garak versions.
        """
        target = Path(str(prefix) + ".report.jsonl")
        if target.is_file():
            return target
        candidates = sorted(out_dir.glob(f"{prefix}*.report.jsonl"))
        if not candidates:
            _LOG.warning("No JSONL report found for prefix %s", prefix)
            return target  # may not exist; callers must handle FileNotFoundError
        return candidates[0]

    def _offline_stub(
        self,
        argv: tuple[str, ...],
        jsonl_path: Path,
    ) -> GarakRunResult:
        """Write a synthetic zero-attempt JSONL when garak isn't installed.

        The records use the same shape as real garak reports so
        :mod:`healer.probe.result_parser` parses them identically; the
        ``total_evaluated == 0`` values are the audit signal that nothing
        was sampled. The verify layer is responsible for surfacing this
        to the user, not the parser.
        """
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = [{
            "entry_type": "start_run setup",
            "plugins.target_type": "openai",
            "plugins.target_name": self._config.target.model,
            "garak_version": "offline-stub",
        }]
        for probe in self._config.probe.garak.probe_spec:
            # Split dotted "module.Class" gracefully; fall back to itself.
            module_name = probe.split(".")[0] if "." in probe else probe
            plugin_class = probe.split(".", 1)[1] if "." in probe else "Probe"
            records.append({
                "entry_type": "eval",
                "probe": f"{module_name}.{plugin_class}",
                "detector": "offline-stub-detector",
                "passed": 0,
                "total_evaluated": 0,
            })
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        _LOG.info(
            "Wrote offline garak stub at %s (%d probe records).",
            jsonl_path, len(records) - 1,
        )
        return GarakRunResult(
            report_path=jsonl_path,
            returncode=-1,
            offline=True,
            duration_seconds=0.0,
            command=argv,
            stderr_tail="",
        )


__all__ = ["GarakRunResult", "GarakRunner"]
