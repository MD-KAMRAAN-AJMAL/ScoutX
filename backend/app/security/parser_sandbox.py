"""Input and subprocess controls for parsers handling untrusted user files.

Python model libraries execute in the Celery worker process, so these checks
limit exposure but are not an OS-level sandbox. External tools are always
invoked with an explicit executable, argument vector, timeout and no shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class ParserInputError(ValueError):
    """The supplied file is not safe or suitable for parsing."""


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_file_bytes: int = 512 * 1024 * 1024
    max_text_bytes: int = 64 * 1024 * 1024
    subprocess_timeout_seconds: int = 300
    max_media_seconds: int = 4 * 60 * 60


DEFAULT_LIMITS = ParserLimits()


def validate_input(path: str | Path, *, max_bytes: int | None = None) -> Path:
    """Require an existing, regular, non-symlink file within configured size."""
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise ParserInputError(f"Cannot access parser input: {exc}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ParserInputError("Parser input must be a regular, non-symlink file")
    limit = DEFAULT_LIMITS.max_file_bytes if max_bytes is None else max_bytes
    if info.st_size > limit:
        raise ParserInputError(f"File exceeds parser limit of {limit} bytes")
    return candidate.resolve(strict=True)


def run_tool(
    executable: str,
    args: Sequence[str],
    *,
    timeout: int = DEFAULT_LIMITS.subprocess_timeout_seconds,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a known external media tool without a shell or inherited secrets."""
    resolved = shutil.which(executable) if not Path(executable).is_absolute() else executable
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"Required executable is unavailable: {executable}")
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}
    }
    return subprocess.run(
        [resolved, *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=env,
    )
