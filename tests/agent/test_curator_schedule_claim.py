from unittest.mock import Mock
from agent import curator


def test_schedule_is_rechecked_after_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(curator, 'get_hermes_home', lambda: tmp_path)
    due = [True]
    monkeypatch.setattr(curator, 'should_run_now', lambda: due[0])
    claim = curator._claim_run
    def other_process_finished_before_claim():
        due[0] = False
        return claim()
    monkeypatch.setattr(curator, '_claim_run', other_process_finished_before_claim)
    review = Mock()
    monkeypatch.setattr(curator, 'run_curator_review', review)
    assert curator.maybe_run_curator() is None
    review.assert_not_called()
    assert not curator._run_claim_path().exists()


def test_due_review_runs_once_and_releases_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(curator, 'get_hermes_home', lambda: tmp_path)
    due = [True]
    monkeypatch.setattr(curator, 'should_run_now', lambda: due[0])
    def review(**kwargs):
        due[0] = False
        return {'started': True}
    run = Mock(side_effect=review)
    monkeypatch.setattr(curator, 'run_curator_review', run)
    assert curator.maybe_run_curator() == {'started': True}
    assert curator.maybe_run_curator() is None
    assert run.call_count == 1
    assert not curator._run_claim_path().exists()
