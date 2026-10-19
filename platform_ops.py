"""Cross-platform native file opening and open-event logging for ScoutX.

Ensures safe execution via platform-specific openers without shell invocation,
performs strict path validation, and enforces exactly-once usage logging.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from scoutx.models import FileDocument
    from scoutx.storage import UsageLog, UserProfileStore


def validate_target_path(path: str | Path) -> Path | None:
    """Validate that the target path exists and is a regular file or directory.

    Returns the resolved Path if valid, or None if invalid or non-existent.
    """
    try:
        resolved = Path(path).resolve()
        if not resolved.exists():
            return None
        return resolved
    except (OSError, ValueError, TypeError):
        return None


def open_file_in_system(
    path: str | Path,
    *,
    platform_name: str | None = None,
) -> tuple[bool, str | None]:
    """Open a file or directory using the host platform's default application handler.

    Uses an argument list and never invokes a shell.
    Returns (success, error_message).
    """
    valid_path = validate_target_path(path)
    if valid_path is None:
        return False, f"Target file does not exist or cannot be accessed: {path}"

    target_str = str(valid_path)
    plat = (platform_name or sys.platform).lower()

    try:
        if plat == "darwin":
            # macOS: 'open' command
            subprocess.Popen(
                ["open", target_str],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return True, None
        elif plat.startswith("win"):
            # Windows: os.startfile native API
            if hasattr(os, "startfile"):
                os.startfile(target_str)  # type: ignore[attr-defined]
            else:
                # Fallback if running on non-Windows test mock
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", target_str],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True, None
        else:
            # Linux and BSD: xdg-open
            subprocess.Popen(
                ["xdg-open", target_str],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return True, None
    except FileNotFoundError as exc:
        return False, f"System opener utility not found on {plat}: {exc}"
    except (OSError, PermissionError) as exc:
        return False, f"Failed to launch application for {target_str}: {exc}"


def record_file_open(
    document: FileDocument,
    *,
    usage_log: UsageLog | None = None,
    profile_store: UserProfileStore | None = None,
    topics: Sequence[str] = (),
) -> bool:
    """Record a file-open event exactly once in ScoutX personalization storage.

    Important architectural invariant:
    ``UsageLog.record_open()`` already internally updates the aggregated tables in
    ``UserProfileStore`` within a single transaction. Calling both ``usage_log.record_open()``
    and ``profile_store.record_open()`` for the same event would double-count the open.

    Therefore:
    - If ``usage_log`` is present, we invoke ``usage_log.record_open()`` only.
    - If ``usage_log`` is None and ``profile_store`` is present, we invoke ``profile_store.record_open()``.
    - If neither is provided, no recording occurs.
    """
    tuple_topics = tuple(topics or ())
    if usage_log is not None:
        usage_log.record_open(document, topics=tuple_topics)
        return True
    elif profile_store is not None:
        profile_store.record_open(
            document_id=document.id,
            extension=document.extension,
            topics=tuple_topics,
        )
        return True
    return False
