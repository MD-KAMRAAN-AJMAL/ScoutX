"""Stable file metadata and streaming SHA-256 fingerprints."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.security.permissions import require_readable_file

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


class FileChangedDuringHashing(OSError):
    """Raised if a file keeps changing while a content hash is being computed."""


@dataclass(frozen=True, slots=True)
class FileStat:
    size_bytes: int
    modified_at: datetime
    modified_ns: int
    device_id: int | None
    inode: int | None


@dataclass(frozen=True, slots=True)
class FileFingerprint(FileStat):
    canonical_path: str
    filename: str
    extension: str | None
    content_hash: str
    mime_type: str | None = None


def stat_file(path: str | os.PathLike[str]) -> FileStat:
    """Read metadata without following a symlink."""
    file_path = require_readable_file(path)
    info = file_path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise OSError(f"Not a regular file: {file_path}")
    return _file_stat(info)


def hash_file_streaming(
    path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE
) -> str:
    """Compute SHA-256 by streaming bounded chunks, without following symlinks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = require_readable_file(path)
    digest = hashlib.sha256()
    descriptor = _open_without_following_symlinks(file_path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
    except BaseException:
        # fdopen owns the descriptor once created; this only covers fdopen failure.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return digest.hexdigest()


def fingerprint_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    retries: int = 2,
    retry_delay: float = 0.05,
) -> FileFingerprint:
    """Hash a stable file and return its canonical metadata and digest.

    Device and inode are best-effort because some filesystems do not expose them.
    A pre/post fstat comparison prevents a partial write from entering the inventory.
    """
    if retries < 0:
        raise ValueError("retries must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = require_readable_file(path)
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        descriptor = _open_without_following_symlinks(file_path)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OSError(f"Not a regular file: {file_path}")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=True) as source:
                while chunk := source.read(chunk_size):
                    digest.update(chunk)
                after = os.fstat(source.fileno())
            if _stable_stat_tuple(before) == _stable_stat_tuple(after):
                after_path = file_path.lstat()
                if _stable_stat_tuple(after) == _stable_stat_tuple(after_path):
                    metadata = _file_stat(after)
                    return FileFingerprint(
                        canonical_path=os.path.abspath(file_path),
                        filename=file_path.name,
                        extension=file_path.suffix.lower() or None,
                        content_hash=digest.hexdigest(),
                        **asdict(metadata),
                    )
            last_error = FileChangedDuringHashing(f"File changed while hashing: {file_path}")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        if attempt < retries and retry_delay > 0:
            time.sleep(retry_delay)
    raise last_error or FileChangedDuringHashing(f"Could not hash stable file: {file_path}")


def content_version(
    previous_hash: str | None, current_hash: str, previous_version: int | None
) -> int:
    """Keep the version stable for identical bytes, increment it for content changes."""
    if previous_version is None:
        return 1
    if previous_hash is None or previous_hash == current_hash:
        return previous_version
    return previous_version + 1


def _file_stat(info: os.stat_result) -> FileStat:
    modified_ns = getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))
    return FileStat(
        size_bytes=info.st_size,
        modified_at=datetime.fromtimestamp(info.st_mtime, tz=UTC),
        modified_ns=modified_ns,
        device_id=getattr(info, "st_dev", None),
        inode=getattr(info, "st_ino", None),
    )


def _stable_stat_tuple(info: os.stat_result) -> tuple[int, int, int, int | None, int | None]:
    return (
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
        getattr(info, "st_dev", None),
        getattr(info, "st_ino", None),
    )


def _open_without_following_symlinks(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        descriptor_info = os.fstat(descriptor)
        path_info = path.lstat()
        if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(descriptor_info.st_mode):
            raise OSError(f"File changed type while being opened: {path}")
        if (descriptor_info.st_dev, descriptor_info.st_ino) != (path_info.st_dev, path_info.st_ino):
            raise FileChangedDuringHashing(f"File was replaced while opening: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
