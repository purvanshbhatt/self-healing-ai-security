"""Local OpenAI-compatible proxy that wires ``garak`` to the patch pipeline.

This module sits between :mod:`healer.llm.target` (the upstream LLM client)
and any OpenAI SDK-compatible probe (typically ``garak``). It exposes:

* ``POST /v1/chat/completions`` — OpenAI-shaped chat completions endpoint
* ``GET  /health``                — liveness probe returning patch state
* ``GET  /v1/models``             — minimal models listing for garak's
  generator auto-discovery (returns only the configured target model)

Request lifecycle
-----------------
For each inbound ``/v1/chat/completions`` request:

1. The body is validated against :class:`ChatCompletionRequest` (pydantic).
2. If ``app.state.proxy_state.apply_patches`` is True, every message's
   ``content`` is run through :meth:`Mitigator.patch_prompt` *before* the
   upstream call.
3. The patched (or verbatim) messages are passed to
   :meth:`TargetLLMClient.generate`.
4. The assistant text is run through :meth:`Mitigator.patch_output` (again
   only if patches are enabled) before being wrapped in the standard
   OpenAI response envelope.
5. Any failure degrades to an OpenAI-style error JSON so SDK clients on the
   other side receive structured error payloads rather than HTML/500s.

PatchEngine contract
--------------------
The proxy is decoupled from the concrete mitigation engine by the
:class:`Mitigator` :mod:`typing.Protocol` defined below. Step 4's
``patch/base.py`` must declare::

    class BaseMitigator:
        async def patch_prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]: ...
        async def patch_output(self, response: str) -> str: ...

Structural typing means ``patch/base.py`` does not need to import the
Protocol — it just needs matching method signatures.

Background execution
--------------------
:meth:`run_proxy_in_background` spins the proxy up inside a daemon thread
driven by :class:`uvicorn.Server`, polls ``/health`` until green, and on
context-exit signals shutdown and joins the thread. This is the only
sanctioned way to drive the proxy from pytest suites and the pipeline
orchestrator (in :mod:`healer.pipeline`) — direct ``uvicorn.run`` calls
block the calling thread and leak the listening socket on exceptions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Final, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from healer.config import HealerConfig
from healer.llm.target import TargetLLMClient
from healer.logging_setup import get_logger

_LOG: Final[logging.Logger] = get_logger(__name__)


# ---------------------------------------------------------------------------
# Patch engine contract
# ---------------------------------------------------------------------------


class Mitigator(Protocol):
    """Structural contract every patch engine must satisfy.

    The two methods are async because mitigation strategies may need to call
    auxiliary models (e.g., an LLM-based classifier for borderline inputs)
    without blocking the proxy's event loop.

    Implementations are free to mutate the inbound lists/strings in place —
    but for sub-millisecond overhead they should prefer pure return values.
    """

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        """Sanitise the inbound chat messages before they reach the model.

        Implementations may rewrite, redact, or inject safety scaffolding
        around individual messages. Returning an empty list is interpreted
        as 'block entirely' and surfaces as a 400 to the caller. The second
        tuple element is ``True`` iff the mitigator actually altered content,
        surfaced into the response ``_healer`` audit block.
        """
        ...

    async def patch_output(self, response: str) -> tuple[str, bool]:
        """Filter the model's output before returning to garak.

        Returns ``(modified_text, was_modified)``; ``was_modified`` lets the
        proxy distinguish fallback-vs-passthrough in the audit block.
        """
        ...


# ---------------------------------------------------------------------------
# Request/response models (OpenAI subset)
# ---------------------------------------------------------------------------


class _ChatMessage(BaseModel):
    """Single chat message; OpenAI schema requires role + content."""

    role: str
    content: str

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """Inbound request body for ``/v1/chat/completions``.

    Mirrors the subset of the OpenAI chat-completions schema used by
    ``garak``'s OpenAI generator. Unknown fields ride along via
    ``extra='allow'`` so probes that send extra fields don't fail validation.
    """

    model: str
    messages: list[_ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None

    model_config = {"extra": "allow"}


class _ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class _Choice(BaseModel):
    index: int = 0
    message: _ChoiceMessage
    finish_reason: str = "stop"


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-shaped success response.

    The ``usage`` block carries placeholder zeros because the proxy does not
    count tokens upstream; we leave accurate accounting to ``garak``'s own
    instrumentation hook. The ``_healer`` field is a non-standard extension
    that records whether the request was patched — useful for verify-phase
    forensics and is ignored by the SDK.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[_Choice]
    usage: _Usage = Field(default_factory=_Usage)
    _healer: dict[str, Any]


# ---------------------------------------------------------------------------
# Proxy state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProxyState:
    """Mutable runtime state held in ``app.state.proxy_state``.

    Kept as a dataclass rather than a dict so typed access patterns can be
    statically checked at the proxy code layer.

    Attributes:
        apply_patches: Whether the Mitigator runs on every request.
        requests_total: Incremented once per inbound completion request.
        patches_applied_total: Incremented once per request where patches ran.
        upstream_errors_total: Incremented on every degraded-fallback path.
    """

    apply_patches: bool
    requests_total: int = 0
    patches_applied_total: int = 0
    upstream_errors_total: int = 0


# ---------------------------------------------------------------------------
# Proxy app
# ---------------------------------------------------------------------------


class HealerProxy:
    """FastAPI application factory plus lifecycle helpers.

    A new :class:`HealerProxy` builds and holds one :class:`FastAPI` instance;
    the assembled app is accessible at :attr:`.app` for ``TestClient`` runs or
    for being handed to :class:`uvicorn.Server`. Multiple proxies may coexist
    (different ports) so concurrent pipeline runs don't collide.
    """

    def __init__(
        self,
        config: HealerConfig,
        target: TargetLLMClient,
        mitigator: Mitigator | None = None,
    ) -> None:
        """Initialise the proxy application.

        Args:
            config: Fully validated :class:`HealerConfig`.
            target: The upstream LLM client (might itself be in mock mode).
            mitigator: Optional patch engine satisfying :class:`Mitigator`.
                When ``None`` or when ``apply_patches`` is False, requests are
                forwarded verbatim — used by the baseline probe phase.
        """
        self.config: HealerConfig = config
        self.target: TargetLLMClient = target
        self.mitigator: Mitigator | None = mitigator
        self.state: ProxyState = ProxyState(
            apply_patches=config.proxy.apply_patches
        )
        self.app: FastAPI = self._build_app()

    # ------------------------------------------------------------------
    # Public state management
    # ------------------------------------------------------------------

    def set_apply_patches(self, enabled: bool) -> None:
        """Toggle whether requests are patched, at runtime.

        Used by :mod:`healer.pipeline` to flip from baseline phase
        (``enabled=False``) to verify phase (``enabled=True``) without
        rebuilding the proxy.

        Args:
            enabled: New patch application state.
        """
        self.state.apply_patches = enabled
        _LOG.info("apply_patches is now %s.", "ON" if enabled else "OFF")

    # ------------------------------------------------------------------
    # App assembly
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        """Assemble the FastAPI app with all routes and middleware."""
        app = FastAPI(
            title="healer-proxy",
            version="0.1.0",
            description="Local OpenAI-compatible proxy with mitigation layer.",
        )
        app.state.proxy_state = self.state
        app.state.proxy_owner = self

        app.middleware("http")(self._timing_middleware)

        app.add_api_route("/health", self._health, methods=["GET"])
        app.add_api_route("/v1/models", self._models, methods=["GET"])
        app.add_api_route(
            "/v1/chat/completions",
            self._chat_completions,
            methods=["POST"],
        )
        return app

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    async def _timing_middleware(self, request: Request, call_next: Any) -> Any:
        """Log method/path/status/duration_ms/patched for every request."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _LOG.info(
            "%s %s -> %d (%.1f ms, patched=%s)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            self.state.apply_patches,
        )
        return response

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    async def _health(self) -> dict[str, Any]:
        """Health probe. Returns 200 + current patch state."""
        return {
            "status": "ok",
            "apply_patches": self.state.apply_patches,
            "model": self.config.target.model,
            "requests_total": self.state.requests_total,
            "patches_applied_total": self.state.patches_applied_total,
            "upstream_errors_total": self.state.upstream_errors_total,
        }

    async def _models(self) -> dict[str, Any]:
        """Minimal OpenAI-style models listing (just our target model)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.target.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "healer",
                }
            ],
        }

    async def _chat_completions(
        self, request: Request
    ) -> JSONResponse:
        """Main entry point: ``POST /v1/chat/completions``.

        The endpoint intentionally accepts the raw :class:`Request` so the
        malformed-payload path stays in our control (rather than letting
        FastAPI return its own HTML error page) and we can guarantee the
        OpenAI-shaped error JSON the SDK on the other side expects.

        Returns:
            :class:`fastapi.responses.JSONResponse` shaped as a standard
            OpenAI chat completion on success, or an OpenAI-shaped error
            envelope on failure with the appropriate HTTP status code.
        """
        self.state.requests_total += 1
        try:
            raw = await request.json()
        except Exception as exc:  # noqa: BLE001 - JSON parse errors are heterogeneous
            return self._error_response(
                status=400,
                message=f"Invalid JSON body: {exc}",
                err_type="invalid_request_error",
                code="bad_json",
            )

        try:
            body = ChatCompletionRequest.model_validate(raw)
        except ValidationError as exc:
            return self._error_response(
                status=422,
                message=str(exc.errors(include_url=False)),
                err_type="invalid_request_error",
                code="validation_error",
            )

        # Convert pydantic models to plain dicts for the target/mitigator.
        messages: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in body.messages
        ]

        patched_flag = self.state.apply_patches and self.mitigator is not None
        prompt_modified = False
        output_modified = False
        if patched_flag and self.mitigator is not None:
            try:
                messages, prompt_modified = await self.mitigator.patch_prompt(messages)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception("patch_prompt raised; returning 502.")
                return self._error_response(
                    status=502,
                    message=f"Patch engine failure (prompt): {exc}",
                    err_type="patch_engine_error",
                    code="patch_prompt_failure",
                )
            if not messages:
                return self._error_response(
                    status=400,
                    message="Request blocked by prompt-level patch policy.",
                    err_type="policy_block",
                    code="blocked_by_prompt_patch",
                )
            self.state.patches_applied_total += 1

        try:
            assistant_text = await self.target.generate(
                messages=messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except TypeError as exc:
            # Programming bug flowing through from the target's validator.
            return self._error_response(
                status=422,
                message=f"Invalid messages: {exc}",
                err_type="invalid_request_error",
                code="bad_messages",
            )
        except Exception as exc:  # noqa: BLE001 - upstream is heterogeneous
            self.state.upstream_errors_total += 1
            _LOG.exception("upstream generate() raised; returning 502.")
            return self._error_response(
                status=502,
                message=f"Upstream LLM failure: {exc}",
                err_type="upstream_error",
                code="upstream_failure",
            )

        if patched_flag and self.mitigator is not None:
            try:
                assistant_text, output_modified = await self.mitigator.patch_output(
                    assistant_text
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.exception("patch_output raised; returning 502.")
                return self._error_response(
                    status=502,
                    message=f"Patch engine failure (output): {exc}",
                    err_type="patch_engine_error",
                    code="patch_output_failure",
                )

        return JSONResponse(
            status_code=200,
            content=self._build_completion_response(
                assistant_text=assistant_text,
                model=body.model,
                patched=patched_flag,
                prompt_modified=prompt_modified,
                output_modified=output_modified,
            ),
        )

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def _build_completion_response(
        self,
        assistant_text: str,
        model: str,
        patched: bool,
        prompt_modified: bool = False,
        output_modified: bool = False,
    ) -> dict[str, Any]:
        """Assemble a standard OpenAI chat-completion JSON envelope."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "_healer": {
                "patched": patched,
                "apply_patches_state": self.state.apply_patches,
                "prompt_modified": prompt_modified,
                "output_modified": output_modified,
            },
        }

    def _error_response(
        self,
        status: int,
        message: str,
        err_type: str,
        code: str,
    ) -> JSONResponse:
        """Build an OpenAI-shaped error JSON envelope."""
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": message,
                    "type": err_type,
                    "code": code,
                }
            },
        )


# ---------------------------------------------------------------------------
# Background-execution helpers
# ---------------------------------------------------------------------------


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Return True when a TCP connect to ``host:port`` succeeds within ``timeout``.

    Used during startup to detect that uvicorn has bound the listening socket
    and is ready to serve requests — we cannot rely on a HTTP probe alone
    because long uvicorn startup messages can race the first connection.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _port_is_free(host: str, port: int) -> bool:
    """Return True if the port can be bound (i.e., it is free)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


@contextlib.asynccontextmanager
async def run_proxy_in_background_async(
    proxy: HealerProxy,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: float = 10.0,
) -> AsyncIterator[str]:
    """Async context manager that runs the proxy and yields its base URL.

    The proxy is driven by :class:`uvicorn.Server` whose ``serve()`` coroutine
    we ``create_task`` into the current loop; on exit we await ``server.shutdown()``
    so the socket is released and pending requests are drained cleanly.

    Args:
        proxy: A constructed :class:`HealerProxy`.
        host: Override ``proxy.config.proxy.host``; default keeps the config value.
        port: Override ``proxy.config.proxy.port``; default keeps the config value.
        ready_timeout: Seconds to wait for the port to accept connections.

    Yields:
        ``http://host:port`` base URL the caller can hand to ``httpx`` / garak.

    Raises:
        RuntimeError: The server did not become ready within ``ready_timeout``.
    """
    h = host or proxy.config.proxy.host
    p = port or proxy.config.proxy.port

    config = uvicorn.Config(
        app=proxy.app,
        host=h,
        port=p,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.config.load()

    serve_task = asyncio.create_task(server.serve())
    try:
        # Poll the port with periodic `await` so the server's serve() task
        # in this same loop gets CPU to bind and start accepting.
        if not await _await_port(h, p, timeout=ready_timeout):
            raise RuntimeError(
                f"Proxy at {h}:{p} did not become ready within {ready_timeout}s."
            )
        yield f"http://{h}:{p}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=5.0)
        except TimeoutError:
            server.force_exit = True
            await asyncio.wait_for(serve_task, timeout=2.0)
        if not _port_is_free(h, p):
            _LOG.warning("Port %s on %s still bound after shutdown.", p, h)


# `typing` asymmetric: this must be defined after the async trait use
async def _await_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Async version of :func:`_wait_for_port` that yields between attempts.

    Essential when the prober and the probed server share the same async
    loop (which is the common case in pytest-asyncio). Without the periodic
    ``await asyncio.sleep``, the server's serve coroutine would never get
    scheduled and the probe would time out.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            await asyncio.sleep(0.05)
    return False


@contextlib.contextmanager
def run_proxy_in_background(
    proxy: HealerProxy,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: float = 10.0,
) -> Iterator[str]:
    """Sync context manager that runs the proxy in a daemon thread.

    Wraps :func:`run_proxy_in_background_async` so callers that are not
    async-aware (pytest fixtures, the pipeline orchestrator) can use the
    proxy without an event loop of their own:

    .. code-block:: python

        with run_proxy_in_background(proxy) as base_url:
            requests.post(f"{base_url}/v1/chat/completions", json=...)

    Implementation: spawn a daemon thread that owns its own asyncio loop,
    drive the async context manager inside it, signal ``ready`` once the
    port accepts connections, then block the outer context. On exit, set
    the loop's ``stop_event`` to make the inner async block return and the
    async context manager's teardown path to run cleanly on the worker.

    Args:
        proxy: Constructed :class:`HealerProxy`.
        host: Host override.
        port: Port override.
        ready_timeout: Seconds to wait for the port to accept connections.

    Yields:
        ``http://host:port`` base URL usable by sync HTTP clients.

    Raises:
        RuntimeError: The proxy did not start within ``ready_timeout``.
    """
    h = host or proxy.config.proxy.host
    p = port or proxy.config.proxy.port

    # Holder is read by the worker thread, written under ready_event.
    base_url_holder: dict[str, str] = {}
    ready_event = threading.Event()
    error_holder: list[BaseException] = []

    # We need to share one Event across two threads; create it on the worker
    # loop side and stuff it back to the parent thread for signalling.
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
    stop_event_holder: dict[str, asyncio.Event] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        stop_event_holder["stop"] = asyncio.Event()
        try:
            loop.run_until_complete(
                _drive_async(
                    proxy, h, p, ready_timeout, ready_event,
                    error_holder, base_url_holder, stop_event_holder["stop"],
                )
            )
        except BaseException as exc:  # noqa: BLE001 - capture for parent
            error_holder.append(exc)
        finally:
            try:
                loop.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                _LOG.debug("Worker loop close failed: %r", exc)

    thread = threading.Thread(target=_worker, daemon=True, name="healer-proxy")
    thread.start()

    # Wait until either the worker signals ready or it died fast.
    deadline = time.perf_counter() + ready_timeout + 1.0
    while not ready_event.is_set():
        if not thread.is_alive() or time.perf_counter() > deadline:
            thread.join(timeout=1.0)
            if error_holder:
                raise RuntimeError(
                    f"Proxy worker died: {error_holder[0]!r}"
                ) from error_holder[0]
            raise RuntimeError(
                f"Proxy at {h}:{p} did not become ready in time."
            )
        time.sleep(0.05)

    if error_holder:
        raise RuntimeError(f"Proxy failed: {error_holder[0]!r}")

    base_url = base_url_holder.get("base_url", f"http://{h}:{p}")
    try:
        _LOG.info(
            "Proxy ready at %s (apply_patches=%s)",
            base_url, proxy.state.apply_patches,
        )
        yield base_url
    finally:
        _teardown_proxy(thread, loop_holder, stop_event_holder, h, p)


async def _drive_async(
    proxy: HealerProxy,
    host: str,
    port: int,
    ready_timeout: float,
    ready_event: threading.Event,
    error_holder: list[BaseException],
    base_url_holder: dict[str, str],
    stop_event: asyncio.Event,
) -> None:
    """Inner coroutine run on the worker loop; holds the async ctx manager open."""
    try:
        async with run_proxy_in_background_async(
            proxy, host=host, port=port, ready_timeout=ready_timeout
        ) as url:
            base_url_holder["base_url"] = url
            ready_event.set()
            # Hold the async ctx open until the parent signals stop.
            await stop_event.wait()
    except BaseException as exc:  # noqa: BLE001
        error_holder.append(exc)
        ready_event.set()  # unblock parent so it can observe the failure


def _teardown_proxy(
    thread: threading.Thread,
    loop_holder: dict[str, asyncio.AbstractEventLoop],
    stop_event_holder: dict[str, asyncio.Event],
    host: str,
    port: int,
) -> None:
    """Signal the worker loop to stop and join the proxy thread.

    Idempotent and safe to call even if the worker isn't running yet.
    Asserts the listening port has been released so callers can detect
    socket leaks at the next test boundary.
    """
    loop = loop_holder.get("loop")
    stop = stop_event_holder.get("stop")
    if loop is not None and stop is not None and not loop.is_closed():
        try:
            loop.call_soon_threadsafe(stop.set)
        except RuntimeError:
            # Loop already closed — nothing more we can do.
            pass

    thread.join(timeout=10.0)
    if thread.is_alive():
        _LOG.warning("Proxy thread at %s:%s did not join cleanly.", host, port)
    if not _port_is_free(host, port):
        _LOG.error("Port %s on %s still bound after teardown.", port, host)


__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "HealerProxy",
    "Mitigator",
    "ProxyState",
    "run_proxy_in_background",
    "run_proxy_in_background_async",
]
