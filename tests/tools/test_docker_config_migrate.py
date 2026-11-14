from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.config_backups import list_config_backups

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker_config_migrate.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("docker_config_migrate_test_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_migration(hermes_home: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_SKIP_CHMOD": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_reuses_global_identical_copies_for_rollback(tmp_path):
    from hermes_cli.config_backups import backup_config

    module = _load_script_module()
    cfg = tmp_path / "config.yaml"
    env = tmp_path / ".env"
    cfg.write_text("model: safe\n", encoding="utf-8")
    env.write_text("EXAMPLE=test-fixture\n", encoding="utf-8")
    saved = {path: backup_config(path, "pre-setup") for path in (cfg, env)}
    assert all(saved.values())
    backups = module._backup_existing((cfg, env))
    assert backups == saved
    for path in (cfg, env):
        path.write_bytes(b"changed")
    assert set(module._restore_backups(backups)) == {cfg, env}
    assert cfg.read_text() == "model: safe\n"
    assert env.read_text() == "EXAMPLE=test-fixture\n"


def test_late_failure_after_real_load_restores_original_bytes(tmp_path, monkeypatch):
    from hermes_cli import config

    module = _load_script_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SKIP_CHMOD", "1")
    cfg = tmp_path / "config.yaml"
    env = tmp_path / ".env"
    originals = {
        cfg: b"# original\r\n_config_version: 12\r\nmodel_catalog:\r\n  ttl_hours: 24\r\n",
        env: b"# fixture only\r\nEXAMPLE=test-fixture\r\n",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    real_missing = config.get_missing_config_fields
    intermediate = []

    def missing_after_real_load():
        result = real_missing()  # Real load_config writes 'good' and prunes globally.
        retained = list_config_backups(cfg)
        assert len(retained) == 1
        assert ".good." in retained[0].name
        assert retained[0].read_bytes() != originals[cfg]
        assert not list_config_backups(cfg, "pre-docker-migrate")
        intermediate.append(retained[0])
        return result

    def fail_late():
        assert intermediate, "the real intermediate load must run before failure"
        env.write_bytes(b"EXAMPLE=changed\n")
        raise RuntimeError("late migration failure")

    monkeypatch.setattr(config, "get_missing_config_fields", missing_after_real_load)
    monkeypatch.setattr(config, "get_missing_skill_config_vars", fail_late)
    with pytest.raises(RuntimeError, match="late migration failure"):
        module.main()
    assert {path: path.read_bytes() for path in originals} == originals


def test_docker_config_migrate_backs_up_and_migrates_legacy_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 12,
                "model_catalog": {"ttl_hours": 24},
                "delegation": {"max_async_children": 8},
            }
        ),
        encoding="utf-8",
    )
    env_path.write_text("OPENROUTER_API_KEY=test\n", encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema 12 ->" in proc.stdout
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    # v24→25 lowers the old default model_catalog TTL to 1h, v39→40 drops
    # that default so ttl_minutes (20) applies; v32→33 folds
    # max_async_children into max_concurrent_children.
    assert "ttl_hours" not in raw["model_catalog"]
    assert raw["delegation"] == {"max_concurrent_children": 8}
    # Successful loads during migration may replace the reason and original bytes globally.
    retained = list_config_backups(config_path)
    assert len(retained) == 1
    saved = yaml.safe_load(retained[0].read_text(encoding="utf-8"))
    assert saved["delegation"] == {"max_concurrent_children": 8}
    assert saved["model_catalog"] == raw["model_catalog"]
    assert list((tmp_path / "backups" / "config").glob(".env.pre-docker-migrate.*"))


def test_docker_config_migrate_skips_below_floor_config_untouched(tmp_path: Path) -> None:
    """Configs below the v12 auto-migration support floor are refused with a
    warning: no migration, no backup, no rewrite — and the boot continues."""
    config_path = tmp_path / "config.yaml"
    original = (
        yaml.safe_dump(
            {
                "_config_version": 11,
                "custom_providers": [
                    {
                        "name": "Local API",
                        "base_url": "http://localhost:8080/v1",
                        "api_key": "test-key",
                    }
                ],
            }
        )
    )
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema" not in proc.stdout
    assert "can no longer be auto-migrated" in proc.stderr
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_stamps_unversioned_config(tmp_path: Path) -> None:
    """A config with no _config_version (a template-seeded volume) is not below the floor, as in
    migrate_config(): it is stamped and keeps its values."""
    config_path = tmp_path / "config.yaml"
    model = {"default": "m", "provider": "openrouter"}
    config_path.write_text(yaml.safe_dump({"model": model}), encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "can no longer be auto-migrated" not in proc.stderr
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert raw["model"] == model


@pytest.mark.parametrize("original", ["model: [unterminated\n", "- a\n- b\n"])
def test_docker_config_migrate_does_not_rewrite_invalid_yaml(tmp_path: Path, original: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Migrating config schema" not in proc.stdout
    assert "leaving config.yaml untouched" in proc.stderr
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_skip_env_leaves_config_unchanged(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    original = yaml.safe_dump({"_config_version": 11})
    config_path.write_text(original, encoding="utf-8")

    proc = _run_migration(tmp_path, HERMES_SKIP_CONFIG_MIGRATION="1")

    assert proc.returncode == 0, proc.stderr
    assert "skipping config migration" in proc.stdout
    assert config_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_docker_config_migrate_restores_backups_after_failed_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    original_config = yaml.safe_dump({"_config_version": 12, "gateway": {"provider": "telegram"}})
    original_env = "TELEGRAM_BOT_TOKEN=test-token\n"
    config_path.write_text(original_config, encoding="utf-8")
    env_path.write_text(original_env, encoding="utf-8")

    monkeypatch.setattr(
        module, "read_config_version_stamp",
        lambda *, raise_on_parse_error=False: (12, DEFAULT_CONFIG["_config_version"]))
    monkeypatch.setattr(module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(module, "get_env_path", lambda: env_path)

    def _failing_migrate(*, interactive: bool, quiet: bool):
        config_path.write_text("gateway: {}\n", encoding="utf-8")
        env_path.write_text("", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "migrate_config", _failing_migrate)

    with pytest.raises(RuntimeError, match="boom"):
        module.main()

    assert config_path.read_text(encoding="utf-8") == original_config
    assert env_path.read_text(encoding="utf-8") == original_env
    assert list((tmp_path / "backups" / "config").glob("config.yaml.pre-docker-migrate.*"))
    assert list((tmp_path / "backups" / "config").glob(".env.pre-docker-migrate.*"))


def test_docker_config_migrate_restores_backups_when_version_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    original_config = yaml.safe_dump({"_config_version": 12, "gateway": {"provider": "telegram"}})
    original_env = "TELEGRAM_BOT_TOKEN=test-token\n"
    config_path.write_text(original_config, encoding="utf-8")
    env_path.write_text(original_env, encoding="utf-8")

    monkeypatch.setattr(
        module, "read_config_version_stamp",
        lambda *, raise_on_parse_error=False: (12, DEFAULT_CONFIG["_config_version"]))
    monkeypatch.setattr(module, "check_config_version", lambda: (12, DEFAULT_CONFIG["_config_version"]))
    monkeypatch.setattr(module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(module, "get_env_path", lambda: env_path)

    def _non_advancing_migrate(*, interactive: bool, quiet: bool):
        config_path.write_text("gateway: {}\n", encoding="utf-8")
        env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "migrate_config", _non_advancing_migrate)

    with pytest.raises(RuntimeError, match="did not advance config version"):
        module.main()

    assert config_path.read_text(encoding="utf-8") == original_config
    assert env_path.read_text(encoding="utf-8") == original_env


def test_docker_config_migrate_second_boot_preserves_env_byte_for_byte(tmp_path: Path) -> None:
    """Regression for #51579: booting ``gateway run`` twice (i.e. a host
    reboot under ``--restart unless-stopped``) must not strip or rewrite
    ``$HERMES_HOME/.env``. The first boot migrates the stale config and bumps
    ``_config_version``; the second boot must be a no-op that leaves ``.env``
    byte-identical to what the user supplied.

    This exercises the real script + real ``migrate_config`` + real file I/O
    via subprocess — not mocks — so it covers the actual Docker boot path,
    not just the failure-rollback shapes above.
    """
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 12,
                "gateway": {"provider": "telegram"},
            }
        ),
        encoding="utf-8",
    )
    original_env = (
        "TELEGRAM_BOT_TOKEN=secret-bot-token\n"
        "TELEGRAM_ALLOWED_USERS=123456789\n"
        "OPENROUTER_API_KEY=sk-test-provider-key\n"
    )
    env_path.write_text(original_env, encoding="utf-8")
    env_bytes_before = env_path.read_bytes()

    # ── First boot: stale config migrates, version advances. ──
    first = _run_migration(tmp_path)
    assert first.returncode == 0, first.stderr
    assert "Migrating config schema 12 ->" in first.stdout
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    # The token (and every other credential) must survive the migration.
    assert env_path.exists(), ".env must never be deleted by the boot migration"
    assert env_path.read_bytes() == env_bytes_before

    config_after_first = config_path.read_bytes()
    first_boot_backups = list_config_backups(config_path)
    assert len(first_boot_backups) == 1

    # ── Second boot (host reboot): version is current, must be a no-op. ──
    second = _run_migration(tmp_path)
    assert second.returncode == 0, second.stderr
    assert "Migrating config schema" not in second.stdout
    # .env is still present and byte-for-byte identical to the original.
    assert env_path.exists()
    assert env_path.read_bytes() == env_bytes_before
    # config.yaml is untouched by the second boot, and no new backup is made.
    assert config_path.read_bytes() == config_after_first
    assert list_config_backups(config_path) == first_boot_backups
