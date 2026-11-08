"""Permission checks and path policy for local filesystem indexing."""

from __future__ import annotations

import os
import stat
from pathlib import Path

SYSTEM_EXCLUDED_ROOTS = (Path("/proc"), Path("/sys"), Path("/dev"))
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".next",
        ".npm",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".yarn",
        "__pycache__",
        "checkpoints",
        "env",
        "hf_cache",
        "huggingface",
        "model_cache",
        "models",
        "node_modules",
        "ollama",
        "pgdata",
        "postgres-data",
        "qdrant",
        "venv",
        "virtualenv",
    }
)
EXCLUDED_FILE_NAMES = frozenset(
    {
        "appendonly.aof",
        "dump.rdb",
        "storage.sqlite",
    }
)
EXCLUDED_FILE_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".sqlite-wal", ".sqlite-shm")


class FilesystemPermissionError(PermissionError):
    """Raised when the current process cannot safely authorize a path."""


class RootUnavailableError(OSError):
    """A registered indexing root is currently absent, unreadable, or remounted."""


def ensure_not_root() -> None:
    """Refuse to scan as root, whose privilege bypasses user permission checks."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and geteuid() == 0:
        raise FilesystemPermissionError("ScoutX filesystem scanning must not run as root")


def canonicalize_authorized_directory(path: str | os.PathLike[str]) -> Path:
    """Resolve and validate a directory the current OS user explicitly selected."""
    ensure_not_root()
    candidate = Path(path).expanduser()
    try:
        canonical = candidate.resolve(strict=True)
        info = canonical.stat()
    except OSError as exc:
        raise FilesystemPermissionError(f"Cannot access selected directory: {candidate}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise FilesystemPermissionError(f"Selected path is not a directory: {canonical}")
    if _is_within_system_excluded_root(canonical):
        raise FilesystemPermissionError(f"System directory cannot be indexed: {canonical}")
    if not _access(canonical, os.R_OK | os.X_OK):
        raise FilesystemPermissionError(f"Current user cannot read and traverse: {canonical}")
    if canonical == Path(canonical.anchor):
        raise FilesystemPermissionError("A filesystem root cannot be registered for indexing")
    return canonical


def require_readable_directory(path: str | os.PathLike[str]) -> Path:
    """Check a registered directory without treating unavailable roots as empty."""
    ensure_not_root()
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise RootUnavailableError(f"Registered root is unavailable: {candidate}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or not _access(candidate, os.R_OK | os.X_OK)
    ):
        raise RootUnavailableError(f"Registered root is not readable: {candidate}")
    if _is_within_system_excluded_root(candidate):
        raise RootUnavailableError(f"Registered root is excluded: {candidate}")
    return candidate


def require_readable_file(path: str | os.PathLike[str], *, follow_symlinks: bool = False) -> Path:
    """Validate a regular file using the process's real filesystem privileges."""
    ensure_not_root()
    candidate = Path(path)
    try:
        info = candidate.stat() if follow_symlinks else candidate.lstat()
    except OSError as exc:
        raise FilesystemPermissionError(f"Cannot stat file: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) and not follow_symlinks:
        raise FilesystemPermissionError(f"Symbolic links are not indexed: {candidate}")
    if not stat.S_ISREG(info.st_mode):
        raise FilesystemPermissionError(f"Not a regular file: {candidate}")
    if not _access(candidate, os.R_OK):
        raise FilesystemPermissionError(f"Current user cannot read file: {candidate}")
    return candidate


def resolve_file_under_root(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    """Resolve a registered file without allowing symlink components or root escape.

    This rechecks the filesystem immediately before use. The final file open must
    still use O_NOFOLLOW where available because filesystem entries can change
    after this check.
    """
    root_path = require_readable_directory(root).resolve(strict=True)
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not is_within(candidate, root_path):
        raise FilesystemPermissionError("File path is outside the authorized directory")
    relative = candidate.relative_to(root_path)
    current = root_path
    try:
        for component in relative.parts:
            current = current / component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise FilesystemPermissionError("Symbolic links are not accessible")
        resolved = candidate.resolve(strict=True)
    except FilesystemPermissionError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FilesystemPermissionError("File is missing or cannot be resolved") from exc
    if not is_within(resolved, root_path):
        raise FilesystemPermissionError("Resolved file is outside the authorized directory")
    return require_readable_file(resolved)


def resolve_directory_under_root(
    path: str | os.PathLike[str], root: str | os.PathLike[str]
) -> Path:
    """Resolve a directory under an authorized root without traversing symlinks."""
    root_path = require_readable_directory(root).resolve(strict=True)
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not is_within(candidate, root_path):
        raise FilesystemPermissionError("Directory path is outside the authorized root")
    current = root_path
    try:
        for component in candidate.relative_to(root_path).parts:
            current = current / component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise FilesystemPermissionError("Directory contains an unsafe path component")
        resolved = candidate.resolve(strict=True)
    except FilesystemPermissionError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FilesystemPermissionError("Directory is missing or cannot be resolved") from exc
    if not is_within(resolved, root_path):
        raise FilesystemPermissionError("Resolved directory is outside the authorized root")
    return require_readable_directory(resolved)


def is_excluded_path(
    path: str | os.PathLike[str],
    *,
    configured_paths: tuple[str | os.PathLike[str], ...] | list[str | os.PathLike[str]] = (),
) -> bool:
    """Apply built-in exclusions and caller-configured exact/subtree exclusions."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    if _is_within_system_excluded_root(candidate):
        return True
    if candidate.name in EXCLUDED_FILE_NAMES or candidate.name.lower().endswith(
        EXCLUDED_FILE_SUFFIXES
    ):
        return True
    for parent in (candidate, *candidate.parents):
        if parent.name.lower() in EXCLUDED_DIRECTORY_NAMES:
            return True
    for configured in configured_paths:
        excluded = Path(os.path.abspath(os.path.expanduser(os.fspath(configured))))
        if _is_relative_to(candidate, excluded):
            return True
    return False


def is_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """Return whether path is lexically beneath root (both should be canonical)."""
    return _is_relative_to(Path(os.path.abspath(path)), Path(os.path.abspath(root)))


def _access(path: Path, mode: int) -> bool:
    try:
        return os.access(path, mode, effective_ids=True)
    except (TypeError, NotImplementedError):
        return os.access(path, mode)


def _is_within_system_excluded_root(path: Path) -> bool:
    return any(_is_relative_to(path, root) for root in SYSTEM_EXCLUDED_ROOTS)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
