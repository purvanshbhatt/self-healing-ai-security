"""Tests for :mod:`healer.llm.proxy`.

Coverage matrix:
* ``/health`` happy path and patch-state reflection.
* ``/v1/models`` minimal listing.
* Baseline (unpatched) request flow — content round-trips verbatim.
* Patched request flow — Mitigator.patch_prompt + patch_output are both
  invoked in the correct order and the response carries ``_healer.patched=True``.
* OpenAI schema: ``id``, ``object``, ``choices[*].message`` shape.
* Malformed payloads: non-JSON -> 400, schema-invalid -> 422.
* Upstream-error fallback JSON shape.
* Background lifecycle: async ctx manager exposes a reachable URL and
  tears the port down cleanly on exit.
* Runtime patch toggle via ``HealerProxy.set_apply_patches``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from healer.config import HealerConfig, Provider, TargetConfig
from healer.llm.proxy import HealerProxy, ProxyState
from healer.llm.target import TargetLLMClient

# ---------------------------------------------------------------------------
# Recording mock mitigator
# ---------------------------------------------------------------------------


class _RecordingMitigator:
    """Minimal Mitigator implementation that captures and rewrites inputs.

    Used as a synthetic patch engine in the patched-flow tests so we can
    assert call order and inputs without depending on ``patch/base.py``
    (which Step 4 will write against the same Protocol).
    """

    def __init__(self, *, prompt_suffix: str = "[PL]", out_suffix: str = "[OL]") -> None:
        self.prompt_suffix = prompt_suffix
        self.out_suffix = out_suffix
        self.prompt_calls: int = 0
        self.output_calls: int = 0
        self.last_prompt_messages: list[dict[str, str]] | None = None
        self.last_output_text: str | None = None

    async def patch_prompt(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], bool]:
        self.prompt_calls += 1
        self.last_prompt_messages = [dict(m) for m in messages]
        new = [
            {**m, "content": f"{m['content']} {self.prompt_suffix}"}
            for m in messages
        ]
        return new, True

    async def patch_output(self, response: str) -> tuple[str, bool]:
        self.output_calls += 1
        self.last_output_text = response
        return f"{response}{self.out_suffix}", True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_target_config() -> TargetConfig:
    """A target config with no upstream key -> TargetLLMClient goes mock mode."""
    return TargetConfig(
        provider=Provider.LOCAL,
        model="mock-target",
        api_base="http://127.0.0.1:1/v1",
        api_key_env="HEALER_PROXY_TEST_KEY_MISSING",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=2,
    )


@pytest.fixture()
def healer_config(local_target_config: TargetConfig) -> HealerConfig:
    """Full HealerConfig with apply_patches=True and ipv4 local proxy."""
    return HealerConfig.load().model_copy(deep=True)


@pytest.fixture()
def target_client(local_target_config: TargetConfig) -> TargetLLMClient:
    return TargetLLMClient(local_target_config)


@pytest.fixture()
def mitigator() -> _RecordingMitigator:
    return _RecordingMitigator()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the env-var referenced by local_target_config is absent."""
    monkeypatch.delenv("HEALER_PROXY_TEST_KEY_MISSING", raising=False)


# ---------------------------------------------------------------------------
# Construction & state
# ---------------------------------------------------------------------------


def test_proxy_state_initialises_from_config(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    assert isinstance(proxy.state, ProxyState)
    assert proxy.state.apply_patches is healer_config.proxy.apply_patches
    assert proxy.state.requests_total == 0


def test_set_apply_patches_toggles_runtime_state(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    assert proxy.state.apply_patches is False
    proxy.set_apply_patches(True)
    assert proxy.state.apply_patches is True


def test_app_exposes_routes(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    # BaseRoute lacks `path` on the static type; only API routes carry it.
    paths = {
        getattr(route, "path", None)
        for route in proxy.app.routes
        if getattr(route, "path", None) is not None
    }
    assert "/health" in paths
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_ok_and_state(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    with TestClient(proxy.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["apply_patches"] is True
    assert body["model"] == "gpt-4o-mini"
    assert body["requests_total"] == 0


def test_health_reflects_runtime_toggle(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.get("/health")
    assert resp.json()["apply_patches"] is False


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_lists_target_only(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    with TestClient(proxy.app) as client:
        resp = client.get("/v1/models")
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Baseline (unpatched) flow
# ---------------------------------------------------------------------------


def _chat_payload(content: str = "Hello", model: str = "gpt-4o-mini") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }


def test_baseline_flow_forwards_verbatim(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """apply_patches=False -> Mitigator is NOT called; target reply round-trips."""
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    mit = _RecordingMitigator()
    proxy.mitigator = mit
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload("abc"))
    assert resp.status_code == 200
    body = resp.json()
    # The mock target returns a deterministic string; no mitigator applied.
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["_healer"]["patched"] is False
    # Baseline implies neither layer modified content even when present.
    assert body["_healer"]["prompt_modified"] is False
    assert body["_healer"]["output_modified"] is False
    assert mit.prompt_calls == 0
    assert mit.output_calls == 0


def test_baseline_flow_increments_request_counter(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        for _ in range(3):
            client.post("/v1/chat/completions", json=_chat_payload())
        resp = client.get("/health")
    assert resp.json()["requests_total"] == 3
    assert resp.json()["patches_applied_total"] == 0


# ---------------------------------------------------------------------------
# Patched flow
# ---------------------------------------------------------------------------


def test_patched_flow_invokes_both_layers_in_order(
    healer_config: HealerConfig,
    target_client: TargetLLMClient,
    mitigator: _RecordingMitigator,
) -> None:
    """Mitigator.patch_prompt then patch_output, response flagged patched=True."""
    proxy = HealerProxy(healer_config, target_client, mitigator)
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload("hello"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["_healer"]["patched"] is True
    # New Step-4 audit flags: distinguishes patch-ran from patch-changed-content.
    assert body["_healer"]["prompt_modified"] is True
    assert body["_healer"]["output_modified"] is True

    assert mitigator.prompt_calls == 1
    assert mitigator.output_calls == 1
    # patch_prompt received the original content (before suffix was added).
    assert mitigator.last_prompt_messages is not None
    assert mitigator.last_prompt_messages[0]["content"] == "hello"
    # patch_output received the mock target's reply (which itself echoes part
    # of the patched input "hello [PL]").
    assert mitigator.last_output_text is not None
    # The suffix the prompt patch added should appear in the echoed output
    # the target received (because mock mode reflects user content prefix).
    assert "[PL]" in mitigator.last_output_text
    # The final response contains the output-layer suffix.
    assert body["choices"][0]["message"]["content"].endswith("[OL]")


def test_patched_flow_counter_increments(
    healer_config: HealerConfig,
    target_client: TargetLLMClient,
    mitigator: _RecordingMitigator,
) -> None:
    proxy = HealerProxy(healer_config, target_client, mitigator)
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        client.post("/v1/chat/completions", json=_chat_payload())
        client.post("/v1/chat/completions", json=_chat_payload())
        resp = client.get("/health")
    body = resp.json()
    assert body["requests_total"] == 2
    assert body["patches_applied_total"] == 2


def test_patched_flow_blocks_when_prompt_returns_empty(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """patch_prompt returning [] -> 400 policy_block, no upstream call."""
    # Block is a forward reference here; define the class first then use it.
    class _InlineBlock:
        async def patch_prompt(
            self, messages: list[dict[str, str]]
        ) -> tuple[list[dict[str, str]], bool]:
            return [], True

        async def patch_output(self, response: str) -> tuple[str, bool]:
            return response, False

    proxy = HealerProxy(healer_config, target_client, _InlineBlock())
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "blocked_by_prompt_patch"


def test_patched_flow_502_on_prompt_failure(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    class _Boom:
        async def patch_prompt(
            self, messages: list[dict[str, str]]
        ) -> tuple[list[dict[str, str]], bool]:
            raise RuntimeError("prompt patch exploded")

        async def patch_output(self, response: str) -> tuple[str, bool]:
            return response, False

    proxy = HealerProxy(healer_config, target_client, _Boom())
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "patch_prompt_failure"


def test_patched_flow_502_on_output_failure(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    class _BoomOut:
        async def patch_prompt(
            self, messages: list[dict[str, str]]
        ) -> tuple[list[dict[str, str]], bool]:
            return messages, False

        async def patch_output(self, response: str) -> tuple[str, bool]:
            raise RuntimeError("output patch exploded")

    proxy = HealerProxy(healer_config, target_client, _BoomOut())
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "patch_output_failure"


# ---------------------------------------------------------------------------
# OpenAI schema compatibility
# ---------------------------------------------------------------------------


def test_response_matches_openai_chat_completion_shape(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload())
    body = resp.json()
    # Required top-level fields.
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "gpt-4o-mini"
    # Choices.
    assert isinstance(body["choices"], list)
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    # Usage.
    assert set(body["usage"].keys()) == {
        "prompt_tokens", "completion_tokens", "total_tokens"
    }


def test_response_id_is_unique_across_requests(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        ids = {
            client.post("/v1/chat/completions", json=_chat_payload()).json()["id"]
            for _ in range(5)
        }
    assert len(ids) == 5  # all 5 distinct -> no id collisions


def test_per_call_overrides_propagate_to_target(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """temperature/max_tokens in the request reach target.generate kwargs."""
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    seen: dict[str, Any] = {}

    async def _spy_generate(
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        seen["temperature"] = temperature
        seen["max_tokens"] = max_tokens
        return "spy-reply"

    proxy.target.generate = _spy_generate  # type: ignore[method-assign]
    with TestClient(proxy.app) as client:
        client.post(
            "/v1/chat/completions",
            json={**_chat_payload(), "temperature": 0.42, "max_tokens": 17},
        )
    assert seen["temperature"] == 0.42
    assert seen["max_tokens"] == 17


# ---------------------------------------------------------------------------
# Error handling: malformed input
# ---------------------------------------------------------------------------


def test_non_json_body_returns_400(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"not really json",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["code"] == "bad_json"


def test_missing_messages_field_returns_422(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "gpt-4o-mini"}
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_empty_messages_returns_422(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": []},
        )
    assert resp.status_code == 422


def test_missing_model_returns_422(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 422


def test_error_payload_has_openai_error_envelope(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    with TestClient(proxy.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"{",
            headers={"content-type": "application/json"},
        )
    body = resp.json()
    assert isinstance(body.get("error"), dict)
    for key in ("message", "type", "code"):
        assert key in body["error"]


# ---------------------------------------------------------------------------
# Upstream-error fallback JSON shape
# ---------------------------------------------------------------------------


def test_upstream_exception_yields_502_with_envelope(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """target.generate raising a generic Exception -> 502 upstream_failure."""

    async def _boom(
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise RuntimeError("upstream detonated")

    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    proxy.target.generate = _boom  # type: ignore[method-assign]
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload())
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_failure"
    # The error counter incremented.
    with TestClient(proxy.app):
        pass
    assert proxy.state.upstream_errors_total == 1


# ---------------------------------------------------------------------------
# Background lifecycle (real uvicorn server, real socket)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_background_lifecycle(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """The async ctx manager exposes a reachable URL and tears down cleanly."""
    from healer.llm.proxy import _port_is_free

    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(False)
    port = 18999  # unlikely to collide on a dev box
    from healer.llm.proxy import run_proxy_in_background_async

    async with run_proxy_in_background_async(
        proxy,
        host="127.0.0.1",
        port=port,
        ready_timeout=10.0,
    ) as base_url:
        assert base_url == f"http://127.0.0.1:{port}"
        # Real HTTP request goes through.
        async with httpx.AsyncClient(base_url=base_url) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            chat = await client.post(
                "/v1/chat/completions", json=_chat_payload("lifecycle")
            )
            assert chat.status_code == 200
            assert chat.json()["_healer"]["patched"] is False

    # After exit, the port must be free.
    assert _port_is_free("127.0.0.1", port), (
        f"Port {port} leaked after async teardown."
    )


@pytest.mark.asyncio
async def test_sync_background_lifecycle_real_socket(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """The sync ctx manager also starts/stops cleanly on a real port."""
    from healer.llm.proxy import _port_is_free, run_proxy_in_background

    proxy = HealerProxy(healer_config, target_client)
    proxy.set_apply_patches(True)
    mit = _RecordingMitigator()
    proxy.mitigator = mit
    port = 19001

    with (
        run_proxy_in_background(
            proxy, host="127.0.0.1", port=port, ready_timeout=10.0
        ) as base_url,
        httpx.Client(base_url=base_url) as client,
    ):
        # Use sync httpx to exercise the non-async path of the orchestrator.
        chat = client.post(
            "/v1/chat/completions",
            json=_chat_payload("sync-lifecycle"),
        )
        assert chat.status_code == 200
        body = chat.json()
        assert body["_healer"]["patched"] is True
        assert body["choices"][0]["message"]["content"].endswith("[OL]")

    assert _port_is_free("127.0.0.1", port), (
        f"Port {port} leaked after sync teardown."
    )
    # The mitigator actually ran once.
    assert mit.prompt_calls == 1
    assert mit.output_calls == 1


# ---------------------------------------------------------------------------
# Structural typing: any object matching Mitigator works
# ---------------------------------------------------------------------------


def test_mitigator_protocol_is_structural(
    healer_config: HealerConfig, target_client: TargetLLMClient
) -> None:
    """A bare duck-typed class with the right methods is acceptable."""
    class _AdHoc:
        async def patch_prompt(
            self, messages: list[dict[str, str]]
        ) -> tuple[list[dict[str, str]], bool]:
            return messages, False

        async def patch_output(self, response: str) -> tuple[str, bool]:
            return response + "-adhoc", True

    proxy = HealerProxy(healer_config, target_client, _AdHoc())
    proxy.set_apply_patches(True)
    with TestClient(proxy.app) as client:
        resp = client.post("/v1/chat/completions", json=_chat_payload("adhoc"))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"].endswith("-adhoc")
