"""Keep Codex OAuth budgets independent from the direct API model catalogue."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest

from agent import model_metadata as metadata
from agent.model_metadata import (
    CODEX_MODELS_CATALOG_URLS,
    fetch_codex_catalog_entries,
)
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def isolate_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(
        json.dumps({"client_version": "0.155.0"}), encoding="utf-8"
    )
    monkeypatch.setattr(metadata, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(metadata, "_codex_oauth_max_context_cache", {})
    monkeypatch.setattr(metadata, "save_context_length", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "agent.models_dev.lookup_models_dev_context", lambda *args: 1_050_000
    )


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_versioned_catalog_reaches_new_models(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """The former 0.0.0 sentinel omits new slugs even when the account has access."""

    def catalog(url: str, **kwargs: object) -> MagicMock:
        version = parse_qs(urlsplit(url).query)["client_version"][0]
        response = MagicMock(status_code=200)
        rows = [{"slug": "gpt-6-astra", "context_window": 272_000}]
        if tuple(map(int, version.split("."))) >= (0, 155, 0):
            rows.append({
                "slug": model,
                "context_window": 300_000,
                "max_context_window": 872_000,
            })
        response.json.return_value = {"models": rows}
        return response

    monkeypatch.setattr(metadata.requests, "get", catalog)
    context = metadata.get_model_context_length(
        model,
        "https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        provider="openai-codex",
    )
    assert context == 300_000


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "openai/gpt-6-sol"])
@pytest.mark.parametrize("status", [200, 401])
def test_missing_catalog_keeps_codex_fallback(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    status: int,
) -> None:
    response = MagicMock(status_code=status)
    response.json.return_value = {"models": []}
    monkeypatch.setattr(metadata.requests, "get", lambda *args, **kwargs: response)
    context = metadata.get_model_context_length(
        model,
        "https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        provider="openai-codex",
    )
    assert context == 272_000
    compressor = ContextCompressor(
        model, config_context_length=context, threshold_tokens_cap=256_000
    )
    assert compressor.should_compress(204_000)
    assert not compressor.should_compress(203_999)


def test_unknown_codex_model_does_not_inherit_api_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {"models": []}
    monkeypatch.setattr(metadata.requests, "get", lambda *args, **kwargs: response)
    assert (
        metadata.get_model_context_length(
            "future-model",
            "https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            provider="openai-codex",
        )
        == metadata.DEFAULT_FALLBACK_CONTEXT
    )


def test_direct_openai_keeps_its_own_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metadata, "_query_ollama_api_show", lambda *args, **kwargs: None
    )
    assert (
        metadata.get_model_context_length("gpt-6-sol", provider="openai") == 1_050_000
    )


@pytest.mark.parametrize("failure", [None, 400, 500, "empty", "malformed", "json", "invalid-rows"])
def test_version_failure_falls_back_once(failure: object) -> None:
    calls = []

    def fetch(url: str) -> MagicMock:
        calls.append(url)
        if len(calls) == 1:
            if failure is None:
                raise TimeoutError()
            if failure == "json":
                response = MagicMock(status_code=200)
                response.json.side_effect = ValueError("invalid JSON")
                return response
            data = {"models": []} if failure == "empty" else {"models": "invalid"}
            if failure == "invalid-rows":
                data = {"models": [None, {}, {"slug": " "}, {"slug": 123}]}
            return MagicMock(
                status_code=failure if isinstance(failure, int) else 200,
                json=lambda: data,
            )
        return MagicMock(
            status_code=200,
            json=lambda: {"models": [{"slug": "gpt-6-sol", "context_window": 272_000}]},
        )

    assert fetch_codex_catalog_entries(fetch)[0][0]["context_window"] == 272_000
    assert len(calls) == 2
    assert calls[-1] == CODEX_MODELS_CATALOG_URLS[-1]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_does_not_retry(status: int) -> None:
    fetch = MagicMock(return_value=MagicMock(status_code=status))
    assert fetch_codex_catalog_entries(fetch) == ([], status)
    fetch.assert_called_once()


@pytest.mark.parametrize("version", [None, "", "bad&account=other", "0.0.0", 155])
def test_local_cache_version_does_not_change_upstream_negotiation(
    tmp_path: Path, version: object
) -> None:
    (tmp_path / "models_cache.json").write_text(
        json.dumps({"client_version": version}), encoding="utf-8"
    )
    fetch = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"models": [{"slug": "example"}]}))
    assert fetch_codex_catalog_entries(fetch)[0] == [{"slug": "example"}]
    fetch.assert_called_once_with(CODEX_MODELS_CATALOG_URLS[0])


def test_token_scoped_cache_ttl_and_no_persistent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(metadata.time, "time", lambda: clock[0])
    save = MagicMock()
    monkeypatch.setattr(metadata, "save_context_length", save)
    fetch = MagicMock(
        side_effect=lambda url, headers, **kw: MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {
                        "slug": "gpt-6-sol",
                        "context_window": 280_000
                        if headers["Authorization"].endswith("one")
                        else 290_000,
                    }
                ]
            },
        )
    )
    monkeypatch.setattr(metadata.requests, "get", fetch)
    route = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    assert (
        metadata.get_model_context_length("gpt-6-sol", api_key="one", **route)
        == 280_000
    )
    assert (
        metadata.get_model_context_length("gpt-6-sol", api_key="one", **route)
        == 280_000
    )
    assert fetch.call_count == save.call_count == 1
    assert (
        metadata.get_model_context_length("gpt-6-sol", api_key="two", **route)
        == 290_000
    )
    clock[0] += metadata._CODEX_OAUTH_CONTEXT_CACHE_TTL + 1
    assert (
        metadata.get_model_context_length("gpt-6-sol", api_key="one", **route)
        == 280_000
    )
    assert fetch.call_count == save.call_count == 3
    fetch.side_effect = TimeoutError()
    assert (
        metadata.get_model_context_length("gpt-6-sol", api_key="offline", **route)
        == 272_000
    )
    assert save.call_count == 3


def test_picker_uses_same_authenticated_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.codex_models import get_codex_model_ids

    def fetch(url: str, **kwargs: object) -> MagicMock:
        assert parse_qs(urlsplit(url).query)["client_version"] == [metadata.CODEX_NEWEST_CLIENT_VERSION]
        return MagicMock(
            status_code=200,
            json=lambda: {
                "models": [{"slug": "gpt-6-sol", "visibility": "list", "priority": 0}]
            },
        )

    monkeypatch.setattr("httpx.get", fetch)
    assert "gpt-6-sol" in get_codex_model_ids(access_token="test-token")


def test_explicit_context_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = MagicMock(
        side_effect=AssertionError("Explicit overrides must not query the catalogue")
    )
    monkeypatch.setattr(metadata.requests, "get", fetch)
    assert (
        metadata.get_model_context_length(
            "gpt-6-sol",
            provider="openai-codex",
            config_context_length=400_000,
        )
        == 400_000
    )
    fetch.assert_not_called()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_context_fix_preserves_native_compaction_gate(model: str) -> None:
    from agent.native_compaction import resolve_native_compaction_capabilities

    assert resolve_native_compaction_capabilities(
        model=model,
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        is_codex_backend=True,
    ) == {"native_compaction": False}
