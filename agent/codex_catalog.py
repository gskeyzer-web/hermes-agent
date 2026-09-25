"""Shared Codex catalogue version negotiation; no cached model entitlements are reused."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

CODEX_UNGATED_CLIENT_VERSION = "0.0.0"
CODEX_MODELS_CATALOG_URL = (
    "https://chatgpt.com/backend-api/codex/models?client_version="
    + CODEX_UNGATED_CLIENT_VERSION
)


class _CatalogResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


def codex_catalog_urls() -> tuple[str, ...]:
    """Prefer the local Codex client's recorded version, then the legacy sentinel.

    Only the non-secret version is used from the cache. Models and their limits
    must come from an authenticated request for the current Hermes account.
    """
    codex_home = Path(
        os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    )
    try:
        data = json.loads(
            (codex_home.expanduser() / "models_cache.json").read_text(encoding="utf-8")
        )
        version = data.get("client_version") if isinstance(data, dict) else None
    except (OSError, ValueError):
        version = None
    if isinstance(version, str) and re.fullmatch(
        r"\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?", version
    ):
        url = CODEX_MODELS_CATALOG_URL.rsplit("=", 1)[0] + "=" + version
        if url != CODEX_MODELS_CATALOG_URL:
            return url, CODEX_MODELS_CATALOG_URL
    return (CODEX_MODELS_CATALOG_URL,)


def fetch_codex_catalog(fetch: Callable[[str], _CatalogResponse]) -> list[dict]:
    """Fetch at most two catalogues, retaining the caller's transport and auth.

    The historical 0.0.0 sentinel can omit new models. A valid local client
    version is preferred; errors/empty catalogues fall back to that sentinel.
    Authentication failures stop immediately. Neither path logs credentials.
    """
    for url in codex_catalog_urls():
        try:
            response = fetch(url)
            if response.status_code in (401, 403):
                return []
            if response.status_code != 200:
                continue
            data = response.json()
            entries = data.get("models") if isinstance(data, dict) else None
            if isinstance(entries, list):
                rows = [
                    item
                    for item in entries
                    if isinstance(item, dict)
                    and isinstance(item.get("slug"), str)
                    and item["slug"].strip()
                ]
                if rows:
                    return rows
        except Exception:
            # Discovery is best-effort; caller owns its offline fallback.
            continue
    return []
