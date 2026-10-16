"""Read-only review probes against a disposable board and real Windows child PIDs.

Run through the candidate scripts/run_tests.sh. No Hermes chat/provider is started.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


DISPATCHER = r'''
import json, os, subprocess, sys
from pathlib import Path
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
home = Path(os.environ['HERMES_HOME'])
conn = kbc.connect()
if sys.argv[1] == 'create':
    tid = kb.create_task(conn, title='synthetic recovery', body='keep checkpoint',
                         assignee='default', workspace_kind='scratch',
                         idempotency_key='autonomy-review-one')
    (home / 'task-id').write_text(tid)
else:
    tid = (home / 'task-id').read_text()
def spawn(task, workspace, board=None):
    # Real subprocess, no model/API. The checkpoint survives dispatcher/worker death.
    code = 'import pathlib,sys,threading; p=pathlib.Path(sys.argv[1]); p.write_text("resumed" if p.exists() else "started"); threading.Event().wait(300)'
    p = subprocess.Popen([sys.executable, '-c', code, str(home / 'checkpoint')],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return p.pid
result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1, max_in_progress=1)
task = kb.get_task(conn, tid)
print(json.dumps({'task': tid, 'status': task.status, 'pid': task.worker_pid,
                  'spawned': len(result.spawned), 'crashed': result.crashed,
                  'run_id': task.current_run_id,
                  'runs': [dict(r) for r in conn.execute('select id,status,outcome from task_runs where task_id=?',(tid,))]}))
conn.close()
'''


@pytest.mark.platforms('windows')
def test_worker_kill_and_dispatcher_restart_recover_once(tmp_path, monkeypatch):
    import psutil
    from tests.waiting import wait_until
    home = tmp_path / 'hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    env = os.environ.copy()
    env.pop('HERMES_PROFILE', None)
    env.pop('HERMES_PROFILE_ROOT', None)
    pids = set()
    def tick(mode):
        completed = subprocess.run([sys.executable, '-c', DISPATCHER, mode],
            env=env, capture_output=True, text=True, timeout=60, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW)
        assert completed.returncode == 0, completed.stderr
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        if data['pid']:
            pids.add(data['pid'])
        return data
    try:
        first = tick('create')
        assert first['status'] == 'running' and first['spawned'] == 1, first
        checkpoint = home / 'checkpoint'
        assert wait_until(checkpoint.exists, timeout=10)
        assert checkpoint.read_text() == 'started'
        # A fresh dispatcher process must see the persisted ownership, not duplicate it.
        live = tick('restart')
        assert live['pid'] == first['pid'] and live['spawned'] == 0, live
        assert live['run_id'] == first['run_id']
        worker = psutil.Process(first['pid'])
        worker.kill()
        worker.wait(timeout=10)
        resumed = tick('restart')
        assert resumed['spawned'] == 1 and resumed['status'] == 'running', resumed
        assert resumed['pid'] != first['pid'] and resumed['run_id'] != first['run_id']
        assert first['task'] == resumed['task']
        assert len(resumed['runs']) == 2
        assert wait_until(lambda: checkpoint.read_text() == 'resumed', timeout=10)
        again = tick('restart')
        assert again['spawned'] == 0 and again['pid'] == resumed['pid']
        assert len(again['runs']) == 2
    finally:
        for pid in pids:
            try:
                worker = psutil.Process(pid)
                worker.kill()
                worker.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
