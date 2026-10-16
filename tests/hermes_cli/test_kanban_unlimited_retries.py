"""Unlimited ordinary recovery retains explicit and terminal failure stops."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn

def test_zero_keeps_retrying_and_preserves_failure_evidence(board):
    tid = kb.create_task(board, title="recover", assignee="default")
    kb.recompute_ready(board)
    for attempt in range(5):
        assert not kbd._record_task_failure(board, tid, "transient", outcome="crashed", failure_limit=0)
        task = kb.get_task(board, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == attempt + 1

def test_zero_recovers_counter_block_but_retains_explicit_stop(board):
    tid = kb.create_task(board, title="counter", assignee="default")
    kb.recompute_ready(board)
    assert kbd._record_task_failure(board, tid, "transient", outcome="crashed", failure_limit=1)
    assert kb.recompute_ready(board, failure_limit=0) == 1
    assert kbd._record_task_failure(board, tid, "credential revoked", outcome="crashed", failure_limit=0, force_trip=True)
    assert kb.recompute_ready(board, failure_limit=0) == 0
    assert kb.get_task(board, tid).status == "blocked"

def test_crash_accounting_receives_unlimited_setting(board):
    tid = kb.create_task(board, title="crash", assignee="default")
    kb.recompute_ready(board)
    dead = SimpleNamespace(error_text="temporary worker failure", protocol_violation=False, terminal_provider=False)
    for _ in range(4):
        assert kbd._account_crashes(board, [(tid, 12345, "test", dead)], failure_limit=0) == []
    assert kb.get_task(board, tid).consecutive_failures == 4
    assert kb.get_task(board, tid).status == "ready"

def test_reclaim_propagates_zero_to_crash_and_timeout(board, monkeypatch):
    calls = []
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: None)
    monkeypatch.setattr(kbd, "reap_terminal_workers", lambda conn: [])
    monkeypatch.setattr(kbd, "detect_stale_running", lambda *a, **kw: [])
    def capture(kind):
        def run(*args, **kwargs):
            calls.append((kind, kwargs["failure_limit"]))
            return []
        return run
    monkeypatch.setattr(kbd, "detect_crashed_workers", capture("crash"))
    monkeypatch.setattr(kbd, "enforce_max_runtime", capture("timeout"))
    kbd._run_reclaim_phase(board, kbd.DispatchResult(), stale_timeout_seconds=60,
                          failure_limit=0, reconcile_orphans=False)
    assert calls == [("crash", 0), ("timeout", 0)]

def test_gateway_accepts_zero():
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    assert _resolve_dispatcher_settings({"failure_limit": 0}, kb).failure_limit == 0


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_unlimited_backoff_survives_new_database_connection(board, monkeypatch, lane):
    tid = kb.create_task(board, title="durable backoff", assignee="default")
    kb.recompute_ready(board)
    assert kb.claim_task(board, tid, claimer=kb._claimer_id())
    assert not kbd._record_task_failure(board, tid, "temporary", outcome="spawn_failed",
                                        failure_limit=0, release_claim=True, end_run=True)
    ended = board.execute("SELECT ended_at FROM task_runs WHERE task_id=?", (tid,)).fetchone()[0]
    ended = kb._to_epoch(ended)
    with kbc.connect() as reopened:
        monkeypatch.setattr(kbd.time, "time", lambda: ended + 59)
        assert kbd.check_respawn_guard(reopened, tid, lane=lane, failure_limit=0) == "retry_backoff"
        monkeypatch.setattr(kbd.time, "time", lambda: ended + 60)
        assert kbd.check_respawn_guard(reopened, tid, lane=lane, failure_limit=0) is None

def test_cli_inherits_zero_but_positive_argument_wins(monkeypatch):
    from hermes_cli import kanban_ops
    monkeypatch.setattr(kanban_ops, "_kanban_config", lambda: {"failure_limit": 0})
    assert kanban_ops._failure_limit(SimpleNamespace(failure_limit=None)) == 0
    assert kanban_ops._failure_limit(SimpleNamespace(failure_limit=4)) == 4


@pytest.mark.parametrize("value", [0.5, -0.5, float("inf"), True, -1])
def test_invalid_limits_cannot_silently_enable_unlimited(value, monkeypatch):
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    from hermes_cli import kanban_ops
    assert _resolve_dispatcher_settings({"failure_limit": value}, kb).failure_limit == kb.DEFAULT_FAILURE_LIMIT
    monkeypatch.setattr(kanban_ops, "_kanban_config", lambda: {"failure_limit": value})
    assert kanban_ops._failure_limit(SimpleNamespace(failure_limit=None)) == kb.DEFAULT_FAILURE_LIMIT

def test_explicit_task_limit_and_diagnostics_override_global_zero(board):
    from hermes_cli import kanban_diagnostics as diagnostics
    tid = kb.create_task(board, title="explicit", assignee="default", max_retries=2)
    kb.recompute_ready(board)
    assert not kbd._record_task_failure(board, tid, "temporary", outcome="crashed", failure_limit=0)
    assert kbd._record_task_failure(board, tid, "temporary", outcome="crashed", failure_limit=0)
    task = kb.get_task(board, tid)
    assert task.status == "blocked"
    result = diagnostics._rule_repeated_failures(task, [], [], 0, {"failure_limit": 0, "failure_threshold": 2})
    assert result[0].data["failure_limit"] == 2
    assert "no attempt ceiling" not in result[0].title

def test_systemic_crashes_stay_blocked_even_with_large_task_override(board):
    tids = [kb.create_task(board, title=f"systemic {i}", assignee="default", max_retries=20) for i in range(3)]
    kb.recompute_ready(board)
    dead = SimpleNamespace(error_text="same deterministic failure", protocol_violation=False, terminal_provider=False)
    assert set(kbd._account_crashes(board, [(tid, i+100, "test", dead) for i,tid in enumerate(tids)], failure_limit=0)) == set(tids)
    assert kb.recompute_ready(board, failure_limit=0) == 0
