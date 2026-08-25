from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time
from types import SimpleNamespace

import agent.auxiliary_client as auxiliary_client
import agent.context_compressor as context_compressor


def test_lean_chunk_digests_run_concurrently_and_preserve_segment_order(monkeypatch):
    lock = threading.Lock()
    active = 0
    max_active = 0

    chunks = [chr(ord("A") + index) * 1_000 for index in range(8)]
    monkeypatch.setattr(
        context_compressor,
        "_serialize_turns_for_digest",
        lambda turns, pristine=None: "".join(chunks),
    )
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_CHUNK_CHARS", 1_000)
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_MAX_CHUNKS", 8)

    def fake_call_llm(**kwargs):
        nonlocal active, max_active
        prompt = kwargs["messages"][0]["content"]
        marker = prompt.rsplit("TRANSCRIPT SEGMENT:\n", 1)[1].strip()[0]
        with lock:
            active += 1
            max_active = max(max_active, active)
        # Earlier chunks finish later, proving result ordering does not depend
        # on completion ordering.
        time.sleep((ord("I") - ord(marker)) * 0.01)
        with lock:
            active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"digest-{marker}"))]
        )

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    compressor = object.__new__(context_compressor.ContextCompressor)
    compressor._lean_pristine_tools = {}
    output = compressor._build_chunk_digests([{"role": "assistant", "content": "ignored"}])

    assert 2 <= max_active <= context_compressor._LEAN_DIGEST_MAX_WORKERS
    positions = [output.index(f"digest-{letter}") for letter in "ABCDEFGH"]
    assert positions == sorted(positions)


def test_lean_chunk_digest_workers_inherit_caller_context(monkeypatch):
    turn_scope = contextvars.ContextVar("test_lean_digest_scope", default="missing")
    turn_scope.set("profile-scope")
    observed = []
    lock = threading.Lock()

    monkeypatch.setattr(
        context_compressor,
        "_serialize_turns_for_digest",
        lambda turns, pristine=None: "A" * 2_000,
    )
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_CHUNK_CHARS", 1_000)

    def fake_call_llm(**kwargs):
        with lock:
            observed.append(turn_scope.get())
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
        )

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    compressor = object.__new__(context_compressor.ContextCompressor)
    compressor._lean_pristine_tools = {}
    compressor._build_chunk_digests([{"role": "assistant", "content": "ignored"}])

    assert observed == ["profile-scope", "profile-scope"]


def test_lean_chunk_digest_failure_isolated_to_its_segment(monkeypatch):
    monkeypatch.setattr(
        context_compressor,
        "_serialize_turns_for_digest",
        lambda turns, pristine=None: "A" * 1_000 + "B" * 1_000 + "C" * 1_000,
    )
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_CHUNK_CHARS", 1_000)

    def fake_call_llm(**kwargs):
        segment = kwargs["messages"][0]["content"].rsplit(
            "TRANSCRIPT SEGMENT:\n", 1
        )[1].strip()
        if segment.startswith("B"):
            raise RuntimeError("synthetic provider failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"digest-{segment[0]}"))]
        )

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    compressor = object.__new__(context_compressor.ContextCompressor)
    compressor._lean_pristine_tools = {}
    output = compressor._build_chunk_digests([{"role": "assistant", "content": "ignored"}])

    assert "### Segment 1/3\ndigest-A" in output
    assert "### Segment 2/3\n[digest unavailable for segment 2/3" in output
    assert "### Segment 3/3\ndigest-C" in output


def test_lean_chunk_digest_telemetry_counts_429_timeouts_and_other_failures(monkeypatch):
    class SyntheticRateLimitError(RuntimeError):
        status_code = 429

    monkeypatch.setattr(
        context_compressor,
        "_serialize_turns_for_digest",
        lambda turns, pristine=None: "A" * 1_000 + "B" * 1_000 + "C" * 1_000 + "D" * 1_000,
    )
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_CHUNK_CHARS", 1_000)

    def fake_call_llm(**kwargs):
        segment = kwargs["messages"][0]["content"].rsplit(
            "TRANSCRIPT SEGMENT:\n", 1
        )[1].strip()
        marker = segment[0]
        if marker == "B":
            raise SyntheticRateLimitError("rate limit exceeded")
        if marker == "C":
            raise TimeoutError("request timed out")
        if marker == "D":
            raise RuntimeError("synthetic provider failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest-A"))]
        )

    monkeypatch.setattr(auxiliary_client, "call_llm", fake_call_llm)

    telemetry = {}
    compressor = object.__new__(context_compressor.ContextCompressor)
    compressor._lean_pristine_tools = {}
    compressor._active_compression_telemetry = telemetry
    compressor._build_chunk_digests([{"role": "assistant", "content": "ignored"}])

    assert telemetry["lean_digest_segment_count"] == 4
    assert telemetry["lean_digest_success_count"] == 1
    assert telemetry["lean_digest_failure_count"] == 3
    assert telemetry["lean_digest_rate_limit_count"] == 1
    assert telemetry["lean_digest_timeout_count"] == 1
    assert telemetry["lean_digest_other_failure_count"] == 1
    assert telemetry["lean_digest_max_workers"] == 4
    assert telemetry["lean_digest_duration_ms"] >= 0


def test_concurrent_lean_compressions_share_global_auxiliary_cap(monkeypatch):
    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    monkeypatch.setattr(
        context_compressor,
        "_serialize_turns_for_digest",
        lambda turns, pristine=None: "A" * 4_000,
    )
    monkeypatch.setattr(context_compressor, "_LEAN_DIGEST_CHUNK_CHARS", 1_000)
    monkeypatch.setattr(
        auxiliary_client,
        "_get_auxiliary_task_config",
        lambda task: {"max_concurrency": 2},
    )
    auxiliary_client._reset_aux_semaphores()

    def fake_call_impl(**kwargs):
        nonlocal active, max_active, calls
        with lock:
            active += 1
            calls += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
        finally:
            with lock:
                active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
        )

    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_call_impl)

    def run_compression():
        compressor = object.__new__(context_compressor.ContextCompressor)
        compressor._lean_pristine_tools = {}
        return compressor._build_chunk_digests(
            [{"role": "assistant", "content": "ignored"}]
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda _index: run_compression(), range(2)))
    finally:
        auxiliary_client._reset_aux_semaphores()

    assert calls == 8
    assert max_active == 2
    assert all("### Segment 4/4" in output for output in outputs)
