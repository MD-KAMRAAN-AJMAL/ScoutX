"""Content-based MIME detection through libmagic."""

from __future__ import annotations

import os
from pathlib import Path

MAGIC_SAMPLE_BYTES = 8192


class MimeDetectionError(OSError):
    """Raised when libmagic is unavailable or cannot inspect a file."""


class FileDetector:
    """Detect MIME from file bytes; extensions are collected separately as metadata."""

    def __init__(self, sample_size: int = MAGIC_SAMPLE_BYTES) -> None:
        if sample_size < 2048:
            raise ValueError("sample_size should be at least 2048 bytes for accurate libmagic results")
        self.sample_size = sample_size

    def detect(self, path: str | os.PathLike[str]) -> str:
        file_path = Path(path)
        descriptor = _open_no_follow(file_path)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as source:
                sample = source.read(self.sample_size)
        except OSError as exc:
            raise MimeDetectionError(f"Could not read file header: {file_path}") from exc
        if not sample:
            return "application/x-empty"
        try:
            import magic
        except ImportError as exc:
            raise MimeDetectionError(
                "python-magic and the system libmagic library are required for MIME detection"
            ) from exc
        try:
            result = magic.from_buffer(sample, mime=True)
        except Exception as exc:
            raise MimeDetectionError(f"libmagic could not identify: {file_path}") from exc
        if not isinstance(result, str) or not result.strip():
            return "application/octet-stream"
        return result.strip().lower()


def detect_mime_type(path: str | os.PathLike[str]) -> str:
    """Convenience function used by ingestion callers and tests."""
    return FileDetector().detect(path)


def _open_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        path_info = path.lstat()
        descriptor_info = os.fstat(descriptor)
        if path_info.st_dev != descriptor_info.st_dev or path_info.st_ino != descriptor_info.st_ino:
            raise OSError(f"File changed while opening: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
