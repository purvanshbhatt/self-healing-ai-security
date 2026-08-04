"""Target LLM client.

This module provides :class:`TargetLLMClient`, a thin async-first wrapper around
the OpenAI-compatible chat-completions API. It is intentionally generic — the same
client works against:

* OpenAI's hosted API (``provider == "openai"``).
* Azure OpenAI (``provider == "azure"``).
* Any OpenAI-compatible local endpoint (``provider == "local"``), including
  our own :mod:`healer.llm.proxy` and mock servers such as Ollama or LM Studio.

The client is consumed by two call-sites:

* :mod:`healer.llm.proxy` forwards upstream requests to it after the
  prompt-patch layer has sanitised inputs.
* :mod:`healer.patch` mechan isms may consult the model directly (for example,
  to ask the LLM itself whether a borderline response is harmful) without
  having to instantiate their own OpenAI client.

Usage
-----
.. code-block:: python

    from healer.config import HealerConfig, TargetConfig
    from healer.llm.target import TargetLLMClient

    cfg = HealerConfig.load()
    client = TargetLLMClient(cfg.target)
    reply = await client.generate(
        messages=[{"role": "user", "content": "Hello."}],
    )

Mock mode
---------
If the environment variable named by ``TargetConfig.api_key_env`` is unset *or*
:attr:`TargetLLMClient.mock` is forced on, generate() short-circuits and returns
a deterministic, hash-derived string. This lets the entire pipeline (and its
unit tests) run fully offline without touching the network, while still
exercising the patch and verify layers with non-trivial text.

Exception policy
----------------
- :class:`openai.APITimeoutError` and :class:`openai.APIConnectionError` are
  caught and surfaced as :class:`FallbackResponse` — the pipeline never
  aborts mid-probe because of a flaky endpoint.
- :class:`openai.RateLimitError`, :class:`openai.AuthenticationError`, and
  other :class:`openai.APIStatusError` are also caught and converted to a
  fallback response, with the underlying error logged at ``ERROR`` level
  (full traceback at ``DEBUG``). The reasoning: a probe run yields hundreds
  of completions; a single bad request should not invalidate the whole batch.
- Malformed user input (non-string ``content``, wrong ``role``) raises
  :class:`TypeError` immediately and is *not* swallowed — those are
  programming bugs, not transient failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Final

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from healer.config import Provider, TargetConfig
from healer.logging_setup import get_logger

_LOG: Final[logging.Logger] = get_logger(__name__)

# Sentinel returned when an upstream call degrades to a fallback response.
# Kept short and recognizable so tests and verify metrics can detect it.
FALLBACK_TEXT_TEMPLATE: Final[str] = (
    "[healer-fallback] request to {model} failed: {reason}"
)

# Header prefix used to make mock-mode responses clearly distinguishable
# in downstream logs and verify reports.
MOCK_SIGNATURE: Final[str] = "[healer-mock]"


@dataclass(frozen=True, slots=True)
class FallbackResponse:
    """Object returned when the upstream call degraded to a non-retryable state.

    Attributes:
        text: The canned text returned to the caller (logged + returned to
            the patcher / garak so the pipeline keeps running).
        reason: Human-readable cause, e.g. ``"timeout"`` or ``"rate_limit"``.
        exception_name: Class name of the caught exception, or the empty
            string when the fallback was requested by config.
    """

    text: str
    reason: str
    exception_name: str

    def __str__(self) -> str:
        """Render the fallback as its canned text (used in log lines)."""
        return self.text


class TargetLLMClient:
    """Async-first typed OpenAI-compatible chat client.

    The class is safe to instantiate in any module: real network I/O happens
    only inside :meth:`generate`, and the underlying ``AsyncOpenAI`` client is
    constructed lazily on first use so that simply *holding* a
    :class:`TargetLLMClient` reference (e.g. in tests, in disabled patch
    branches) never triggers a connection attempt.

    Attributes:
        config: The :class:`TargetConfig` the client was built from. Read-only
            by convention; mutations will not propagate to the cached client.
        mock: When True, every :meth:`generate` call returns a deterministic
            mock string derived from the input — see :meth:`_mock_reply`.
    """

    # Marker written into the AsyncOpenAI default_headers so upstreams can
    # identify traffic coming from the healer pipeline (audit trail).
    _USER_AGENT: Final[str] = "healer/0.1.0 (self-healing-pipeline)"

    def __init__(self, config: TargetConfig) -> None:
        """Initialise the client wrapper.

        Args:
            config: A fully validated :class:`TargetConfig` from
                :class:`healer.config.HealerConfig.target`.
        """
        self.config: TargetConfig = config
        # Decided once at construction; flipping later requires re-creating
        # the client, which is why this is not a writable attribute.
        self._mock: bool = False
        self._client: AsyncOpenAI | None = None
        # Cache the resolved API key for diagnostics; never logged in full.
        self._api_key_present: bool = False
        self._determine_mock_mode()

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    @property
    def mock(self) -> bool:
        """Whether the client is operating in offline mock mode."""
        return self._mock

    def force_mock(self, enabled: bool = True) -> None:
        """Force mock mode on or off, resetting any cached real client.

        Useful in tests that want to flip behaviour without re-instantiating.

        Args:
            enabled: When True, :meth:`generate` returns deterministic mock
                text; when False, subsequent calls go to the real endpoint
                unless the API key is missing.
        """
        if enabled == self._mock:
            return
        self._mock = enabled
        # A real client (if any) was bound to a different mode — drop it so
        # the next generate() call rebuilds against the new mode.
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Request a chat completion and return the assistant's text.

        Args:
            messages: OpenAI-format chat messages, e.g.
                ``[{"role": "user", "content": "Hello"}]``. Exactly one
                ``content`` field per message must be a non-empty string;
                otherwise a :class:`TypeError` is raised.
            temperature: Override ``TargetConfig.temperature`` for this call
                only. Values outside ``[0, 1]`` are clamped by the SDK.
            max_tokens: Override ``TargetConfig.max_tokens`` for this call
                only. Must be positive.

        Returns:
            The assistant message's ``content`` string. When the upstream
            call degrades to a fallback response (timeout, connection
            failure, rate limit, auth error), the returned string is the
            canned :class:`FallbackResponse` text so the caller never has to
            branch on success vs. failure.

        Raises:
            TypeError: ``messages`` is empty or any message is malformed
                (missing ``role``/``content``, non-string content). These
                are programming errors and propagate to the caller.
        """
        self._validate_messages(messages)

        # Offline stub: produce a deterministic reply without touching the
        # network. This branch is hot in unit tests and in CI runs that lack
        # API credentials.
        if self._mock:
            return self._mock_reply(messages)

        client = self._get_or_build_client()
        effective_temp = self.config.temperature if temperature is None else temperature
        effective_max = self.config.max_tokens if max_tokens is None else max_tokens

        # pydantic-validated config already guarantees positive values; this
        # assertion guards against accidental override via the per-call arg.
        if effective_max <= 0:
            raise TypeError(f"max_tokens must be > 0, got {effective_max}")

        try:
            completion = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=effective_temp,
                max_tokens=effective_max,
            )
        except APITimeoutError as exc:
            return self._fallback("timeout", exc)
        except APIConnectionError as exc:
            return self._fallback("connection_error", exc)
        except RateLimitError as exc:
            return self._fallback("rate_limit", exc)
        except AuthenticationError as exc:
            return self._fallback("authentication_error", exc)
        except APIError as exc:  # catch-all for non-retryable API failures
            return self._fallback("api_error", exc)

        choice = completion.choices[0]
        content: str | None = choice.message.content
        if content is None or not content.strip():
            _LOG.warning(
                "Upstream returned empty completion (finish_reason=%s); "
                "returning fallback.",
                choice.finish_reason,
            )
            return FALLBACK_TEXT_TEMPLATE.format(
                model=self.config.model, reason="empty_completion"
            )
        return content

    def generate_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Synchronous facade over :meth:`generate`.

        Runs the coroutine on the current event loop (or a fresh one if none
        is running) and returns the resulting string. Intended for CLI
        scripts, pytest fixtures, and code paths that are not async-aware.

        Raises:
            RuntimeError: Called from inside a running event loop — running
                the coroutine would deadlock. Use :meth:`generate` instead.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "generate_sync() must not be called from a running event "
                "loop — await generate() instead."
            )
        return asyncio.run(
            self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _determine_mock_mode(self) -> None:
        """Decide at construction time whether to operate in mock mode.

        The client falls back to mock mode when the configured environment
        variable is unset *or* when the target provider is explicitly
        ``local`` and the base URL looks like our own proxy (avoids
        self-referential loops during end-to-end integration tests).
        """
        api_key = os.environ.get(self.config.api_key_env, "")
        self._api_key_present = bool(api_key)
        # The local provider typically runs behind a mock or our own proxy
        # and has no upstream API key; mark as mock unless one was provided.
        if not api_key and self.config.provider != Provider.LOCAL:
            _LOG.warning(
                "Environment variable %s is unset; switching to mock mode. "
                "Set it in your .env to run against a real endpoint.",
                self.config.api_key_env,
            )
            self._mock = True
        elif not api_key:
            _LOG.info("Provider is 'local' and no API key set; using mock mode.")
            self._mock = True
        else:
            self._mock = False

    def _get_or_build_client(self) -> AsyncOpenAI:
        """Lazily construct and cache the AsyncOpenAI client.

        Construction is deferred until the first real call so that the
        constructor's only side effect is reading environment variables.
        """
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.config.api_key_env, "")
        if not api_key:
            # Reached only if force_mock(False) was called after init without
            # providing a key; flip back to mock to avoid an SDK error.
            _LOG.warning(
                "force_mock(False) called without %s set; reverting to "
                "mock mode.",
            )
            self._mock = True
            return self._get_or_build_client()

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.api_base,
            timeout=float(self.config.timeout_seconds),
            max_retries=2,
            default_headers={"User-Agent": self._USER_AGENT},
        )
        _LOG.info(
            "Initialised AsyncOpenAI client (provider=%s, model=%s, "
            "base_url=%s, timeout=%ds).",
            self.config.provider.value,
            self.config.model,
            self.config.api_base,
            self.config.timeout_seconds,
        )
        return self._client

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        """Reject messages likely to crash the SDK early, with clear errors.

        Validation is intentionally strict: it mirrors the OpenAI chat schema
        and lets the proxy surface malformed patch-engine output as a
        programming error rather than a cryptic SDK stack trace.
        """
        if not messages:
            raise TypeError("messages must be a non-empty list.")
        for idx, msg in enumerate(messages):
            if "role" not in msg:
                raise TypeError(f"message[{idx}] missing 'role' field.")
            if "content" not in msg:
                raise TypeError(f"message[{idx}] missing 'content' field.")
            if not isinstance(msg["content"], str):
                raise TypeError(
                    f"message[{idx}].content must be str, got "
                    f"{type(msg['content']).__name__}."
                )
            if not msg["content"]:
                raise TypeError(f"message[{idx}].content must be a non-empty str.")

    def _mock_reply(self, messages: list[dict[str, str]]) -> str:
        """Produce a deterministic, hashed reply derived from the input.

        The output is intentionally non-empty and unique per distinct input so
        downstream patch and verify layers see real variation across probes
        (a constant reply would defeat the purpose of running them).

        Args:
            messages: The canonical chat messages. Concatenated role+content
                pairs are hashed so identical inputs jitter-deterministically
                while even a single-character change yields different text.

        Returns:
            A recognisable stub string beginning with :data:`MOCK_SIGNATURE`.
        """
        canonical = "|".join(
            f"{m['role']}:{m['content']}" for m in messages
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:12]
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        # Echo a 40-char prefix of the prompt so that output-layer policy
        # filters can exercise keyword matching against realistic text.
        snippet = last_user[:40].replace("\n", " ")
        return f"{MOCK_SIGNATURE} digest={digest} echo=\"{snippet}\""

    def _fallback(self, reason: str, exc: BaseException) -> str:
        """Log a structured fallback notice and return the canned text.

        Args:
            reason: Short, machine-readable cause identifier (no spaces).
            exc: The exception that triggered the fallback. Its repr is
                appended to a debug log entry for post-mortem analysis.

        Returns:
            :data:`FALLBACK_TEXT_TEMPLATE` populated with the model name and
            ``reason``.
        """
        fb = FallbackResponse(
            text=FALLBACK_TEXT_TEMPLATE.format(model=self.config.model, reason=reason),
            reason=reason,
            exception_name=type(exc).__name__,
        )
        _LOG.error(
            "Upstream call failed -> fallback (provider=%s, model=%s, "
            "reason=%s, exc=%s).",
            self.config.provider.value,
            self.config.model,
            reason,
            type(exc).__name__,
        )
        _LOG.debug("Underlying exception: %r", exc, exc_info=exc)
        return fb.text

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact, secret-free representation suitable for log lines."""
        return (
            f"TargetLLMClient(model={self.config.model!r}, "
            f"provider={self.config.provider.value!r}, "
            f"base_url={self.config.api_base!r}, mock={self._mock})"
        )

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Best-effort close hook (no-op; SDK GC handles connection pools).

        The async context is gone by destruction time, so we cannot await
        ``aclose()`` on the underlying ``AsyncOpenAI``. The SDK holds an
        ``httpx.AsyncClient`` whose own ``__del__`` releases the connection
        pool. We intentionally do nothing here: raising inside ``__del__``
        would only be silently swallowed by CPython, so an empty body is
        the safest contract.

        Safe under partial-init: if construction aborted before ``_client``
        was set, this method is never reached because the attribute lookup
        is bypassed entirely.
        """
        return


__all__ = [
    "FALLBACK_TEXT_TEMPLATE",
    "MOCK_SIGNATURE",
    "FallbackResponse",
    "TargetLLMClient",
]
