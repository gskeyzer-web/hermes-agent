"""Point-in-time copies of ``config.yaml``: one directory, one naming scheme, bounded count.

Every writer that wants a "before" copy of the user's config (setup wizard, corrupt-file
snapshot, model migrations) goes through :func:`backup_config`. Copies live in
``<HERMES_HOME>/backups/config/`` — ``backups/`` is already excluded from full backups, so they
never nest — as ``config.yaml.<reason>.<YYYYMMDD-HHMMSS>``. A copy identical to the newest one
across reasons is skipped; by default one functional copy survives globally per config.
Only a byte-verified YAML mapping can replace it. One separate ``corrupt`` diagnostic is
retained, never used for recovery. Paired ``.env`` migration copies are byte-verified only.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BACKUPS_SUBDIR = Path("backups") / "config"
DEFAULT_KEEP = 1

# Names earlier code wrote next to config.yaml (setup wizard, corrupt snapshot, xai migration).
# Moved into the backups dir on first use so they stop accumulating in the home root; hand-named
# copies (``config.yaml.bak-my-note``) are the user's and are never touched.
_LEGACY_SIBLING_GLOBS = ("config.yaml.bak.[0-9]*", "config.yaml.corrupt.*.bak", "config.yaml.bak-pre-migrate-*")


def backups_dir(config_path: Path) -> Path:
    return config_path.parent / BACKUPS_SUBDIR


def list_config_backups(config_path: Path, reason: Optional[str] = None) -> list[Path]:
    """Existing backups, newest first; filtered to one *reason* when given."""
    root = backups_dir(config_path)
    if not root.is_dir():
        return []
    prefix = f"{config_path.name}.{reason}." if reason else f"{config_path.name}."
    return sorted((p for p in root.iterdir() if p.is_file() and p.name.startswith(prefix)),
                  key=lambda p: p.name.rsplit(".", 1)[-1], reverse=True)


def _recovery_backups(config_path: Path) -> list[Path]:
    # Broken-YAML diagnostics are not recovery points and must never evict one.
    return [p for p in list_config_backups(config_path)
            if not p.name.startswith(f"{config_path.name}.corrupt.")]


def _parse_config_copy(content: bytes) -> Optional[dict]:
    from utils import fast_safe_load
    try:
        data = fast_safe_load(content.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def backup_config(config_path: Path, reason: str, *, keep: int = DEFAULT_KEEP) -> Optional[Path]:
    """Copy *config_path* to the backups dir; return the new path, or None when skipped/failed.

    Skips when the file is missing/empty, or when the newest functional backup already holds
    identical bytes. Never raises: a failed backup must not block the write it precedes.
    ``keep`` is global across functional reasons (minimum one); ``corrupt`` has a separate cap.
    Skipped entirely for a config backend without file tooling (no local copies, D2).
    """
    from hermes_cli.config_backend import supports_file_tooling
    if not supports_file_tooling():
        return None
    try:
        if not config_path.is_file() or config_path.stat().st_size == 0:  # config-reader: ok — file tooling, gated above
            return None
        content = config_path.read_bytes()  # config-reader: ok — file tooling, gated above
        if config_path.name != ".env" and reason != "corrupt" and _parse_config_copy(content) is None:
            return None
        root = backups_dir(config_path)
        root.mkdir(parents=True, exist_ok=True)
        _sweep_legacy_siblings(config_path, root)
        existing = (list_config_backups(config_path, reason) if reason == "corrupt"
                    else _recovery_backups(config_path))
        # Not filecmp.cmp: its process-wide cache keys on (size, mtime), so an equal-size edit
        # within one mtime tick of an earlier comparison is reported identical and never backed up.
        if existing and content == existing[0].read_bytes():
            # The existing point is byte-identical and the source was validated above;
            # still converge historical per-reason copies to the global retention limit.
            for stale in existing[max(1, keep):]:
                stale.unlink(missing_ok=True)
            return None
        dest = root / f"{config_path.name}.{reason}.{time.strftime('%Y%m%d-%H%M%S')}"
        if dest.is_symlink():  # never write through a planted link
            return None
        # Only publish a byte-verified copy. Failed/truncated writes stay hidden and cannot
        # become the newest recovery candidate or trigger eviction of the previous point.
        with tempfile.TemporaryDirectory(prefix=".config-backup-", dir=root) as staging:
            staged = Path(staging) / config_path.name
            shutil.copy2(config_path, staged)  # config-reader: ok — file tooling, gated above
            if staged.read_bytes() != content:
                return None
            staged.replace(dest)
        # Same-second edits can replace the same name atomically; never prune that name twice.
        for stale in [dest, *(p for p in existing if p != dest)][max(1, keep):]:
            stale.unlink(missing_ok=True)
        return dest
    except OSError as exc:
        logger.warning("Could not back up %s (%s): %s", config_path, reason, exc)
        return None


def load_newest_good_backup(config_path: Path) -> Optional[dict]:
    """Parse the newest functional backup, regardless of the writer's reason.

    Returns the raw mapping, or None when there is no usable copy. Older copies are not
    tried: a backup that fails to parse means the on-disk copy was damaged after the fact, and
    guessing further back would serve a config the user never saw as current.
    """
    from hermes_cli.config_backend import supports_file_tooling
    if not supports_file_tooling():
        return None
    newest = _recovery_backups(config_path)[:1]
    if not newest:
        return None
    try:
        return _parse_config_copy(newest[0].read_bytes())
    except Exception as exc:
        logger.warning("Last-known-good backup %s is unreadable: %s", newest[0], exc)
        return None


def _sweep_legacy_siblings(config_path: Path, root: Path) -> None:
    for pattern in _LEGACY_SIBLING_GLOBS:
        for old in config_path.parent.glob(pattern):
            if not old.is_file() or old.is_symlink():
                continue
            try:
                old.replace(root / old.name)
            except OSError as exc:
                logger.debug("Could not move legacy backup %s: %s", old, exc)
