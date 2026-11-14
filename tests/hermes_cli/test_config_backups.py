"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
import os
from pathlib import Path

import pytest

from hermes_cli.config_backups import backup_config, list_config_backups, load_newest_good_backup


@pytest.fixture
def saved_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: safe\n", encoding="utf-8")
    stamps = iter(f"20260101-0000{i:02d}" for i in range(20))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))
    good = backup_config(cfg, "good")
    assert good is not None
    return cfg, good


def test_identical_verified_copy_still_prunes_legacy_reason_backups(saved_config):
    cfg, good = saved_config
    legacy = good.parent / "config.yaml.pre-setup.20251201-000000"
    legacy.write_bytes(b"model: older\n")
    assert backup_config(cfg, "good") is None
    assert list_config_backups(cfg) == [good]
    assert good.read_bytes() == cfg.read_bytes()


@pytest.mark.parametrize("bad", [b"model: [unclosed", b"- not-a-mapping", b"scalar", b"", b"\xff"])
@pytest.mark.parametrize("reason", ["good", "pre-setup"])
def test_invalid_config_never_replaces_functional_copy(saved_config, bad, reason):
    cfg, good = saved_config
    cfg.write_bytes(bad)
    assert backup_config(cfg, reason) is None
    assert list_config_backups(cfg) == [good]
    assert load_newest_good_backup(cfg) == {"model": "safe"}


@pytest.mark.parametrize("raises", [False, True])
def test_incomplete_copy_never_publishes_or_evicts_good(saved_config, monkeypatch, raises):
    cfg, good = saved_config
    cfg.write_text("model: replacement\n", encoding="utf-8")

    def partial_copy(src, dest):
        # Still valid YAML: validation must also compare bytes, not just parse.
        dest.write_bytes(b"model: partial\n")
        if raises:
            raise OSError("disk full")

    monkeypatch.setattr("hermes_cli.config_backups.shutil.copy2", partial_copy)
    assert backup_config(cfg, "pre-setup") is None
    assert list_config_backups(cfg) == [good]
    assert load_newest_good_backup(cfg) == {"model": "safe"}


def test_corrupt_diagnostic_is_bounded_and_never_used_for_recovery(saved_config):
    cfg, good = saved_config
    for broken in (b"model: [broken", b"model: [another"):
        cfg.write_bytes(broken)
        diagnostic = backup_config(cfg, "corrupt")
        assert diagnostic is not None and diagnostic.read_bytes() == broken
        assert len(list_config_backups(cfg, "corrupt")) == 1
        assert good.exists()
        assert load_newest_good_backup(cfg) == {"model": "safe"}


def test_successive_same_second_edits_replace_only_after_validation(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: "20260101-000000")
    for model in ("first", "second", "third"):
        cfg.write_text(f"model: {model}\n", encoding="utf-8")
        current = backup_config(cfg, "good")
        assert current is not None
        assert list_config_backups(cfg) == [current]
        assert load_newest_good_backup(cfg) == {"model": model}
    cfg.write_bytes(b"model: [broken")
    assert backup_config(cfg, "good") is None
    assert load_newest_good_backup(cfg) == {"model": "third"}


def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(3):
        cfg.write_text(f"model: {i}\n")
        backup_config(cfg, "pre-setup", keep=2)
    kept = list_config_backups(cfg, "pre-setup")
    assert len(kept) == 2 and kept[0].read_text() == "model: 2\n"
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_default_retention_is_global_across_reasons_and_recoverable(tmp_path, monkeypatch):
    from hermes_cli.config_backups import load_newest_good_backup

    cfg = tmp_path / "config.yaml"
    stamps = iter(f"20260101-00000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))
    previous = None
    for reason in ("good", "pre-setup", "pre-migrate-xai", "good"):
        cfg.write_text(f"model: {reason}\n", encoding="utf-8")
        current = backup_config(cfg, reason)
        assert current is not None
        assert list_config_backups(cfg) == [current]
        assert load_newest_good_backup(cfg) == {"model": reason}
        if previous:
            assert not previous.exists()
        previous = current


def test_same_size_edit_within_one_mtime_tick_is_still_backed_up(tmp_path: Path, monkeypatch):
    # Coarse-timestamp filesystems leave an immediate rewrite with the old mtime; dedupe must
    # compare content, not trust a (size, mtime) signature.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    assert backup_config(cfg, "pre-setup") is not None
    assert backup_config(cfg, "pre-setup") is None
    before = cfg.stat()
    cfg.write_text("model: b\n")
    os.utime(cfg, ns=(before.st_atime_ns, before.st_mtime_ns))

    second = backup_config(cfg, "pre-setup")
    assert second is not None and second.read_text() == "model: b\n"


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    # The new diagnostic replaces the legacy corrupt copy under keep=1.
    assert not (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert len(list_config_backups(cfg, "corrupt")) == 1
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))
