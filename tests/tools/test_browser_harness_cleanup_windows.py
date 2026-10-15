"""Windows delete-pending cleanup: wait for disappearance, never delete twice."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load_runner():
    path = Path(__file__).parents[2] / "tools/browser_harness_runner.py"
    spec = importlib.util.spec_from_file_location("cleanup_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def denied():
    error = PermissionError("synthetic delete pending")
    error.winerror = 5
    return error


def test_delete_pending_disappears_without_second_unlink(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    endpoint = Mock()
    endpoint.stat.side_effect = [denied(), FileNotFoundError()]
    cleanup = Mock(side_effect=denied())
    ipc = SimpleNamespace(IS_WINDOWS=True, cleanup_endpoint=cleanup, port_path=lambda name: endpoint)
    runner.install_cleanup_guard(ipc)
    ipc.cleanup_endpoint("synthetic")
    assert cleanup.call_count == 1
    assert endpoint.stat.call_count == 2


def test_persistent_denial_is_not_hidden(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    error = denied()
    cleanup = Mock(side_effect=error)
    endpoint = Mock()
    endpoint.stat.side_effect = error
    ipc = SimpleNamespace(IS_WINDOWS=True, cleanup_endpoint=cleanup, port_path=lambda name: endpoint)
    runner.install_cleanup_guard(ipc)
    with pytest.raises(PermissionError) as caught:
        ipc.cleanup_endpoint("synthetic")
    assert caught.value is error
    assert cleanup.call_count == 1
    assert endpoint.stat.call_count <= 5


def test_successor_endpoint_is_never_deleted(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    cleanup = Mock(side_effect=denied())
    endpoint = Mock()  # stat succeeds: a file still exists, possibly a successor.
    ipc = SimpleNamespace(IS_WINDOWS=True, cleanup_endpoint=cleanup, port_path=lambda name: endpoint)
    runner.install_cleanup_guard(ipc)
    with pytest.raises(PermissionError):
        ipc.cleanup_endpoint("synthetic")
    assert cleanup.call_count == 1


@pytest.mark.parametrize("windows,winerror", [(False, 5), (True, 13), (True, None)])
def test_other_platform_or_error_is_not_retried(windows, winerror):
    runner = load_runner()
    error = denied()
    error.winerror = winerror
    cleanup = Mock(side_effect=error)
    endpoint = Mock()
    ipc = SimpleNamespace(IS_WINDOWS=windows, cleanup_endpoint=cleanup, port_path=lambda name: endpoint)
    runner.install_cleanup_guard(ipc)
    with pytest.raises(PermissionError):
        ipc.cleanup_endpoint("synthetic")
    endpoint.stat.assert_not_called()
    cleanup.assert_called_once_with("synthetic")
