"""Unit tests for :mod:`healer.llm.target`.

Covers the four behaviour contract points:

* successful happy-path completion (mocked AsyncOpenAI).
* timeout, connection error, rate-limit, auth-error → fallback strings.
* API-key resolution via ``TargetConfig.api_key_env``.
* mock mode (offline, deterministic, hash-derived text).
* input validation (empty messages, missing fields, non-string content).
* ``generate_sync`` blocking facade and its runtime guard inside a live loop.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from healer.config import Provider, TargetConfig
from healer.llm.target import (
    MOCK_SIGNATURE,
    TargetLLMClient,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_completion(content: str, finish_reason: str = "stop") -> Any:
    """Build a minimal OpenAI ChatCompletion-shaped object for mocking.

    We use :class:`unittest.mock.MagicMock` rather than the real pydantic
    types so this test file has no hard dependency on the internals of any
    specific openai SDK version's type hierarchy.
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture()
def online_config() -> TargetConfig:
    """A TargetConfig whose env-var name points at a key we actually set."""
    return TargetConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        api_key_env="HEALER_TEST_OPENAI_KEY",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=5,
    )


@pytest.fixture()
def local_config() -> TargetConfig:
    """A local-provider config (no key needed); forces mock mode by default."""
    return TargetConfig(
        provider=Provider.LOCAL,
        model="local-mock",
        api_base="http://127.0.0.1:8899/v1",
        api_key_env="MISSING_KEY",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=2,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a deterministically-empty key env before each test."""
    monkeypatch.delenv("MISSING_KEY", raising=False)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_mode_when_api_key_missing(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key in env => client auto-enables mock mode."""
    monkeypatch.delenv(online_config.api_key_env, raising=False)
    client = TargetLLMClient(online_config)
    assert client.mock is True

    reply = await client.generate(
        messages=[{"role": "user", "content": "hello there"}]
    )
    assert MOCK_SIGNATURE in reply
    assert "hello there"[:10] in reply  # echo prefix included


@pytest.mark.asyncio
async def test_mock_mode_is_deterministic(local_config: TargetConfig) -> None:
    """Same input -> same output (sha256 prefix stable)."""
    client = TargetLLMClient(local_config)
    a = await client.generate(messages=[{"role": "user", "content": "abc"}])
    b = await client.generate(messages=[{"role": "user", "content": "abc"}])
    assert a == b


@pytest.mark.asyncio
async def test_mock_mode_differs_for_distinct_inputs(
    local_config: TargetConfig,
) -> None:
    client = TargetLLMClient(local_config)
    a = await client.generate(messages=[{"role": "user", "content": "abc"}])
    b = await client.generate(messages=[{"role": "user", "content": "xyz"}])
    assert a != b


def test_force_mock_toggles_and_drops_cached_client(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_mock(False) flips mode and clears any cached real client."""
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    assert client.mock is False

    client.force_mock(True)
    assert client.mock is True
    assert client._client is None


# ---------------------------------------------------------------------------
# Environment-key resolution
# ---------------------------------------------------------------------------


def test_api_key_resolution_from_env(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-resolved-example")
    client = TargetLLMClient(online_config)
    assert client.mock is False


def test_proxy_kwargs_carry_base_url_and_timeout(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the lazily-built AsyncOpenAI client receives our config."""
    monkeypatch.setenv(online_config.api_key_env, "sk-x")
    client = TargetLLMClient(online_config)

    with patch("healer.llm.target.AsyncOpenAI") as fake_ctor:
        fake_ctor.return_value = MagicMock(name="fake_async_client")
        client._get_or_build_client()
        fake_ctor.assert_called_once()
        _, kwargs = fake_ctor.call_args
        assert kwargs["api_key"] == "sk-x"
        assert kwargs["base_url"] == online_config.api_base
        assert kwargs["timeout"] == float(online_config.timeout_seconds)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_assistant_content(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocked AsyncOpenAI.chat.completions.create surfaces assistant text."""
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)

    fake_completion = _make_completion("Hello from upstream.")
    with patch.object(
        client, "_get_or_build_client"
    ) as fake_get:
        async_client = MagicMock()
        async_client.chat = MagicMock()
        async_client.chat.completions = MagicMock()
        async_client.chat.completions.create = AsyncMock(
            return_value=fake_completion
        )
        fake_get.return_value = async_client

        out = await client.generate(messages=[{"role": "user", "content": "hi"}])
        assert out == "Hello from upstream."

        # Confirm per-call overrides would have flowed to the SDK.
        await client.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=20,
        )
        # Latest call kwargs
        last_kwargs = async_client.chat.completions.create.call_args_list[-1].kwargs
        assert last_kwargs["temperature"] == 0.7
        assert last_kwargs["max_tokens"] == 20
        assert last_kwargs["model"] == online_config.model


@pytest.mark.asyncio
async def test_generate_empty_completion_returns_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)

    fake_completion = _make_completion("")
    with patch.object(client, "_get_or_build_client") as fake_get:
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(
            return_value=fake_completion
        )
        fake_get.return_value = async_client

        out = await client.generate(messages=[{"role": "user", "content": "hi"}])
        assert "healer-fallback" in out
        assert "empty_completion" in out


# ---------------------------------------------------------------------------
# Fallback on upstream failures
# ---------------------------------------------------------------------------


def _patch_create_with_exc(
    client: TargetLLMClient, exc: BaseException
) -> Any:
    """Patch ``_get_or_build_client`` to return an AsyncOpenAI whose
    create() raises ``exc``."""
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(side_effect=exc)
    return patch.object(client, "_get_or_build_client", return_value=async_client)


@pytest.mark.asyncio
async def test_timeout_yields_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    # APITimeoutError needs a request kwarg in the SDK; supply a dummy.
    exc = APITimeoutError(request=MagicMock())
    with _patch_create_with_exc(client, exc):
        out = await client.generate(messages=[{"role": "user", "content": "x"}])
    assert "healer-fallback" in out
    assert "timeout" in out


@pytest.mark.asyncio
async def test_connection_error_yields_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    exc = APIConnectionError(request=MagicMock())
    with _patch_create_with_exc(client, exc):
        out = await client.generate(messages=[{"role": "user", "content": "x"}])
    assert "connection_error" in out


@pytest.mark.asyncio
async def test_rate_limit_yields_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    # RateLimitError requires body + response kwargs.
    exc = RateLimitError(
        message="rate limit hit",
        response=MagicMock(),
        body=None,
    )
    with _patch_create_with_exc(client, exc):
        out = await client.generate(messages=[{"role": "user", "content": "x"}])
    assert "rate_limit" in out


@pytest.mark.asyncio
async def test_auth_error_yields_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    exc = AuthenticationError(
        message="bad key",
        response=MagicMock(),
        body=None,
    )
    with _patch_create_with_exc(client, exc):
        out = await client.generate(messages=[{"role": "user", "content": "x"}])
    assert "authentication_error" in out


@pytest.mark.asyncio
async def test_generic_api_error_yields_fallback(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    # APIError requires message + request; we use a low-severity subclass via
    # a raw APIError instance if available.
    exc = APIError(
        message="boom",
        request=MagicMock(),
        body=None,
    )
    with _patch_create_with_exc(client, exc):
        out = await client.generate(messages=[{"role": "user", "content": "x"}])
    assert "api_error" in out


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_messages_rejected(local_config: TargetConfig) -> None:
    client = TargetLLMClient(local_config)
    with pytest.raises(TypeError, match="non-empty"):
        await client.generate(messages=[])


@pytest.mark.asyncio
async def test_message_missing_role_rejected(local_config: TargetConfig) -> None:
    client = TargetLLMClient(local_config)
    with pytest.raises(TypeError, match="missing 'role'"):
        await client.generate(messages=[{"content": "hi"}])


@pytest.mark.asyncio
async def test_message_missing_content_rejected(local_config: TargetConfig) -> None:
    client = TargetLLMClient(local_config)
    with pytest.raises(TypeError, match="missing 'content'"):
        await client.generate(messages=[{"role": "user"}])


@pytest.mark.asyncio
async def test_non_string_content_rejected(local_config: TargetConfig) -> None:
    client = TargetLLMClient(local_config)
    with pytest.raises(TypeError, match="must be str"):
        await client.generate(messages=[{"role": "user", "content": 123}])  # type: ignore[dict-item]


@pytest.mark.asyncio
async def test_empty_content_rejected(local_config: TargetConfig) -> None:
    client = TargetLLMClient(local_config)
    with pytest.raises(TypeError, match="non-empty str"):
        await client.generate(messages=[{"role": "user", "content": ""}])


@pytest.mark.asyncio
async def test_non_positive_max_tokens_rejected(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    client = TargetLLMClient(online_config)
    client.force_mock(False)
    # Provide fake key path so we reach the max_tokens guard before any SDK call.
    monkeypatch.setenv(online_config.api_key_env, "sk-test")
    with pytest.raises(TypeError, match="max_tokens"):
        await client.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=0,
        )


# ---------------------------------------------------------------------------
# generate_sync facade
# ---------------------------------------------------------------------------


def test_generate_sync_runs_without_loop(local_config: TargetConfig) -> None:
    """Called outside an event loop, it runs the coroutine cleanly."""
    client = TargetLLMClient(local_config)
    out = client.generate_sync(messages=[{"role": "user", "content": "sync hi"}])
    assert MOCK_SIGNATURE in out
    assert "sync hi"[:7] in out


def test_generate_sync_rejects_running_loop(local_config: TargetConfig) -> None:
    """Called inside a running loop, it raises RuntimeError instead of deadlock."""

    async def _runner() -> None:
        client = TargetLLMClient(local_config)
        with pytest.raises(RuntimeError, match="running event loop"):
            client.generate_sync(messages=[{"role": "user", "content": "x"}])

    asyncio.run(_runner())


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_is_secret_free(
    online_config: TargetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(online_config.api_key_env, "sk-super-secret-12345")
    client = TargetLLMClient(online_config)
    r = repr(client)
    assert "sk-super-secret" not in r
    assert "gpt-4o-mini" in r
    assert "openai" in r
