"""Watchdog integration with event debouncing and write-stability checks."""

from __future__ import annotations

import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, Self

from sqlalchemy.orm import Session
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.config import get_settings
from app.database.models import IndexRoot
from app.database.repositories import IndexRootRepository
from app.security.permissions import (
    RootUnavailableError,
    ensure_not_root,
    is_excluded_path,
    is_within,
    require_readable_directory,
    require_readable_file,
)


class TaskPublisher(Protocol):
    def discover_file(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        path: str,
        *,
        old_path: str | None = None,
    ) -> object: ...

    def delete_file(self, user_id: uuid.UUID, root_id: uuid.UUID, path: str) -> object: ...


class CeleryTaskPublisher:
    """Lazy Celery task publisher used by the Watchdog observer thread."""

    def discover_file(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        path: str,
        *,
        old_path: str | None = None,
    ) -> object:
        from app.workers.tasks import discover_file

        return discover_file.apply_async(
            args=[str(user_id), str(root_id), path],
            kwargs={"old_path": old_path},
        )

    def delete_file(self, user_id: uuid.UUID, root_id: uuid.UUID, path: str) -> object:
        from app.workers.tasks import delete_file

        return delete_file.delay(str(user_id), str(root_id), path)


class DebouncedFilesystemEventHandler(FileSystemEventHandler):
    """Translate Watchdog notifications into scoped, debounced Celery messages."""

    def __init__(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        root_path: str | os.PathLike[str],
        *,
        publisher: TaskPublisher | None = None,
        excluded_paths: Iterable[str | os.PathLike[str]] = (),
        expected_device_id: int | None = None,
        debounce_seconds: float = 0.35,
        stability_timeout: float = 3.0,
        stability_interval: float = 0.15,
        stable_samples: int = 2,
    ) -> None:
        super().__init__()
        if debounce_seconds < 0 or stability_timeout < 0 or stability_interval < 0:
            raise ValueError("watcher delays must be nonnegative")
        if stable_samples < 1:
            raise ValueError("stable_samples must be positive")
        self.user_id = user_id
        self.root_id = root_id
        self.root_path = require_readable_directory(root_path).resolve(strict=True)
        self.expected_device_id = (
            expected_device_id
            if expected_device_id is not None
            else getattr(self.root_path.stat(), "st_dev", None)
        )
        self.publisher = publisher or CeleryTaskPublisher()
        self.excluded_paths = tuple(excluded_paths)
        self.debounce_seconds = debounce_seconds
        self.stability_timeout = stability_timeout
        self.stability_interval = stability_interval
        self.stable_samples = stable_samples
        self._lock = threading.Lock()
        self._pending: dict[str, threading.Timer] = {}

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_discover(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_discover(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_allowed_path(event.src_path, allow_missing=True):
            self._schedule_delete(os.path.abspath(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        source = os.path.abspath(event.src_path)
        destination = os.path.abspath(event.dest_path)
        source_inside = self._is_allowed_path(source, allow_missing=True)
        destination_inside = self._is_allowed_path(destination, allow_missing=True)
        if source_inside and destination_inside:
            self._schedule_discover(destination, old_path=source)
        elif source_inside:
            self._schedule_delete(source)
        elif destination_inside:
            self._schedule_discover(destination)

    def close(self) -> None:
        """Cancel timers when the owning observer is stopped."""
        with self._lock:
            timers = tuple(self._pending.values())
            self._pending.clear()
        for timer in timers:
            timer.cancel()

    def _schedule_discover(self, path: str, *, old_path: str | None = None) -> None:
        if not self._is_allowed_path(path, allow_missing=False):
            return
        canonical = os.path.abspath(path)

        def publish_when_stable() -> None:
            if not self.is_file_stable(canonical):
                return
            # Validate again after the debounce window to close the common rename race.
            if self._is_allowed_path(canonical, allow_missing=False):
                self.publisher.discover_file(
                    self.user_id,
                    self.root_id,
                    canonical,
                    old_path=old_path,
                )

        self._schedule(canonical, publish_when_stable)

    def _schedule_delete(self, path: str) -> None:
        def publish_if_root_is_still_mounted() -> None:
            if self._root_is_available():
                self.publisher.delete_file(self.user_id, self.root_id, path)

        self._schedule(path, publish_if_root_is_still_mounted)

    def _root_is_available(self) -> bool:
        try:
            current = require_readable_directory(self.root_path)
            device_id = getattr(current.stat(), "st_dev", None)
            return self.expected_device_id is None or device_id == self.expected_device_id
        except OSError:
            return False

    def _schedule(self, key: str, callback: Callable[[], object]) -> None:
        timer: threading.Timer

        def run() -> None:
            try:
                callback()
            finally:
                with self._lock:
                    if self._pending.get(key) is timer:
                        self._pending.pop(key, None)

        timer = threading.Timer(self.debounce_seconds, run)
        timer.daemon = True
        with self._lock:
            previous = self._pending.pop(key, None)
            if previous is not None:
                previous.cancel()
            self._pending[key] = timer
        timer.start()

    def _is_allowed_path(self, value: str | os.PathLike[str], *, allow_missing: bool) -> bool:
        candidate = Path(os.path.abspath(value))
        if not is_within(candidate, self.root_path) or is_excluded_path(
            candidate, configured_paths=self.excluded_paths
        ):
            return False
        try:
            relative = candidate.relative_to(self.root_path)
            current = self.root_path
            for part in relative.parts:
                current = current / part
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    return allow_missing
                if stat.S_ISLNK(info.st_mode):
                    return False
            require_readable_file(candidate)
        except FileNotFoundError:
            return allow_missing
        except (OSError, PermissionError):
            return False
        return True

    def is_file_stable(self, value: str | os.PathLike[str]) -> bool:
        """Require unchanged size and mtime for consecutive samples before queuing."""
        path = Path(value)
        deadline = time.monotonic() + self.stability_timeout
        last: tuple[int, int] | None = None
        stable = 0
        while True:
            try:
                require_readable_file(path)
                info = path.stat()
                current = (
                    info.st_size,
                    getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
                )
            except OSError:
                return False
            if current == last:
                stable += 1
                if stable >= self.stable_samples:
                    return True
            else:
                last = current
                stable = 1
            if time.monotonic() >= deadline:
                return False
            time.sleep(self.stability_interval)


class RootWatcher:
    """Manage a recursive Watchdog observer for one authorized root."""

    def __init__(self, observer: Observer, handler: DebouncedFilesystemEventHandler) -> None:
        self.observer = observer
        self.handler = handler

    def start(self) -> None:
        self.observer.start()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        self.handler.close()
        self.observer.stop()
        self.observer.join(timeout=timeout)


class FilesystemWatcherManager:
    """Own and restart the live observers for one authenticated user's roots."""

    def __init__(self, *, publisher: TaskPublisher | None = None) -> None:
        self.publisher = publisher
        self._watchers: dict[uuid.UUID, RootWatcher] = {}
        self._lock = threading.RLock()

    def start_root(self, session: Session, user_id: uuid.UUID, root_id: uuid.UUID) -> RootWatcher:
        with self._lock:
            current = self._watchers.get(root_id)
            if current is not None and current.observer.is_alive():
                return current
            watcher = create_root_watcher(
                session,
                user_id,
                root_id,
                publisher=self.publisher,
            )
            watcher.start()
            self._watchers[root_id] = watcher
            return watcher

    def start_for_user(self, session: Session, user_id: uuid.UUID) -> int:
        roots = IndexRootRepository(session).list_for_user(user_id, enabled_only=True)
        started = 0
        for root in roots:
            try:
                self.start_root(session, user_id, root.id)
                started += 1
            except RootUnavailableError:
                # Keep the user's other roots active if one removable drive is absent.
                continue
        return started

    def stop_root(self, root_id: uuid.UUID) -> bool:
        with self._lock:
            watcher = self._watchers.pop(root_id, None)
        if watcher is None:
            return False
        watcher.stop()
        return True

    def stop_all(self) -> None:
        with self._lock:
            watchers = tuple(self._watchers.values())
            self._watchers.clear()
        for watcher in watchers:
            watcher.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_all()


def create_root_watcher(
    session: Session,
    user_id: uuid.UUID,
    root_id: uuid.UUID,
    *,
    publisher: TaskPublisher | None = None,
    debounce_seconds: float = 0.35,
    stability_timeout: float = 3.0,
) -> RootWatcher:
    """Build a Watchdog observer after verifying the registered root and OS access."""
    ensure_not_root()
    root: IndexRoot | None = IndexRootRepository(session).get(user_id, root_id)
    if root is None or not root.is_enabled:
        raise LookupError("Enabled index root does not exist for this user")
    root_path = require_readable_directory(root.canonical_path).resolve(strict=True)
    current_device = getattr(root_path.stat(), "st_dev", None)
    expected_device = (root.settings or {}).get("filesystem_device_id")
    if expected_device is not None and expected_device != current_device:
        raise RootUnavailableError(f"Registered root is on a different device: {root_path}")
    handler = DebouncedFilesystemEventHandler(
        user_id,
        root_id,
        root_path,
        publisher=publisher,
        excluded_paths=(
            *get_settings().excluded_paths,
            *((root.settings or {}).get("excluded_paths", [])),
        ),
        expected_device_id=(root.settings or {}).get("filesystem_device_id"),
        debounce_seconds=debounce_seconds,
        stability_timeout=stability_timeout,
    )
    observer = Observer()
    observer.schedule(handler, str(root_path), recursive=True)
    return RootWatcher(observer, handler)
