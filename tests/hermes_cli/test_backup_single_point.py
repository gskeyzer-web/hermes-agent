"""Automatic retention replaces a recovery point only after a complete copy."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import zipfile

import pytest

from hermes_cli import backup


@pytest.fixture
def home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("model: first\n", encoding="utf-8")
    ticks = iter(datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(30))

    class Clock:
        @staticmethod
        def now(tz=None):
            return next(ticks)

    monkeypatch.setattr(backup, "datetime", Clock)
    return tmp_path


@pytest.mark.parametrize("settings", ["defaults", "missing", "unreadable"])
def test_updater_full_backup_uses_single_point_default(home, monkeypatch, settings):
    from hermes_cli import update_cmd_maint as maint
    from hermes_cli.config import DEFAULT_CONFIG

    def updates():
        if settings == "unreadable":
            raise OSError("unreadable settings")
        return DEFAULT_CONFIG["updates"] if settings == "defaults" else {}

    monkeypatch.setattr(maint, "_load_updates_cfg", updates)
    monkeypatch.setattr(backup, "get_default_hermes_root", lambda: home)
    for _ in range(3):
        maint._run_full_backup()
        assert len(list((home / "backups").glob("pre-update-*.zip"))) == 1


def test_same_second_quick_replacement_keeps_the_new_point(home, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 1, 1)

    monkeypatch.setattr(backup, "datetime", Clock)
    previous = None
    for value in ("first", "second", "third", "fourth"):
        (home / "config.yaml").write_text(f"model: {value}\n", encoding="utf-8")
        result = backup.create_quick_snapshot(hermes_home=home)
        assert result is not None
        current = home / "state-snapshots" / result
        assert current.exists()
        assert (current / "config.yaml").read_text() == f"model: {value}\n"
        if previous:
            assert not previous.exists()
        previous = current


@pytest.mark.parametrize("relative", ["config.yaml", "platforms/pairing/approved.json"])
@pytest.mark.parametrize("failure", ["copy-error", "oversized"])
def test_incomplete_non_database_snapshot_keeps_previous_point(home, monkeypatch, relative, failure):
    source = home / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    content = "model: " + "x" * 100 + "\n"
    source.write_text(content, encoding="utf-8")
    (home / "auth.json").write_text("{}", encoding="utf-8")
    first = backup.create_quick_snapshot(hermes_home=home)
    assert first is not None
    previous = home / "state-snapshots" / first
    if failure == "copy-error":
        real_copy = backup.shutil.copy2

        def fail_one(src, dst):
            if src == source:
                raise OSError("unreadable source")
            return real_copy(src, dst)

        monkeypatch.setattr(backup.shutil, "copy2", fail_one)
        second = backup.create_quick_snapshot(hermes_home=home)
    else:
        second = backup.create_quick_snapshot(hermes_home=home, max_file_size=50)
    assert second is not None
    assert previous.exists(), "an incomplete snapshot evicted the only complete recovery point"
    assert (previous / relative).read_text() == content


@pytest.mark.parametrize("kind", ["quick", "pre-update", "pre-migration"])
def test_default_retention_replaces_previous_complete_point(home, kind):
    create = {
        "quick": backup.create_quick_snapshot,
        "pre-update": backup.create_pre_update_backup,
        "pre-migration": backup.create_pre_migration_backup,
    }[kind]
    previous = None
    for value in ("first", "second", "third"):
        (home / "config.yaml").write_text(f"model: {value}\n", encoding="utf-8")
        result = create(hermes_home=home)
        assert result is not None
        current = home / "state-snapshots" / result if kind == "quick" else result
        if previous is not None:
            assert not previous.exists()
        if kind == "quick":
            assert (current / "config.yaml").read_text() == f"model: {value}\n"
        else:
            with zipfile.ZipFile(current) as archive:
                assert archive.testzip() is None
                assert archive.read("config.yaml") == (home / "config.yaml").read_bytes()
        previous = current


@pytest.mark.parametrize("relative", ["config.yaml", "platforms/pairing/approved.json"])
@pytest.mark.parametrize("damage", ["truncated", "same-size"])
def test_unverified_quick_copy_keeps_previous_point(home, monkeypatch, relative, damage):
    source = home / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"model: original\n")
    (home / "auth.json").write_bytes(b"{}")
    first = backup.create_quick_snapshot(hermes_home=home)
    previous = home / "state-snapshots" / first
    real_copy = backup.shutil.copy2

    def damaged_copy(src, dst):
        result = real_copy(src, dst)
        if src == source:
            Path(dst).write_bytes(b"model: " if damage == "truncated" else b"model: modified\n")
        return result

    monkeypatch.setattr(backup.shutil, "copy2", damaged_copy)
    second = backup.create_quick_snapshot(hermes_home=home)
    assert previous.is_dir(), "unverified copy evicted the valid snapshot"
    assert (previous / relative).read_bytes() == b"model: original\n"
    if second is not None:
        current = home / "state-snapshots" / second
        manifest = json.loads((current / "manifest.json").read_text())
        assert relative in manifest["failed_files"]
        assert relative not in manifest["files"]


@pytest.mark.parametrize("content", [b"model: [unterminated\n", b"- not-a-mapping\n"])
def test_invalid_quick_config_keeps_previous_point(home, content):
    original = (home / "config.yaml").read_bytes()
    (home / "auth.json").write_bytes(b"{}")
    first = backup.create_quick_snapshot(hermes_home=home)
    previous = home / "state-snapshots" / first
    (home / "config.yaml").write_bytes(content)
    second = backup.create_quick_snapshot(hermes_home=home)
    assert previous.is_dir(), "invalid YAML evicted the valid snapshot"
    assert (previous / "config.yaml").read_bytes() == original
    if second is not None:
        manifest = json.loads((home / "state-snapshots" / second / "manifest.json").read_text())
        assert "config.yaml" in manifest["failed_files"]
        assert "config.yaml" not in manifest["files"]


@pytest.mark.parametrize("kind", ["pre-update", "pre-migration"])
def test_closed_zip_crc_failure_keeps_previous_point(home, monkeypatch, kind):
    create = backup.create_pre_update_backup if kind == "pre-update" else backup.create_pre_migration_backup
    previous = create(hermes_home=home)
    original = previous.read_bytes()
    real_close = zipfile.ZipFile.close

    def corrupt_after_close(archive):
        corrupt = archive.mode == "w" and archive.fp is not None
        if corrupt:
            info = archive.getinfo("config.yaml")
            filename = archive.filename
        real_close(archive)
        if corrupt:
            # Damage member data after final close; the central directory stays readable.
            with open(filename, "r+b") as stream:
                stream.seek(info.header_offset + 30 + len(info.filename.encode()) + len(info.extra))
                value = stream.read(1)
                stream.seek(-1, 1)
                stream.write(bytes([value[0] ^ 0xFF]))

    monkeypatch.setattr(zipfile.ZipFile, "close", corrupt_after_close)
    result = create(hermes_home=home)
    assert result is None, "damaged closed archive was published as complete"
    assert previous.read_bytes() == original
    assert list((home / "backups").glob(f"{kind}-*.zip")) == [previous]
