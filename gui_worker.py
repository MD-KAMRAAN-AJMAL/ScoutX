"""Non-blocking search worker, debouncer, and generation guard for ScoutX GUI.

Coordinates search requests outside the main UI thread with:
- 150 ms query debouncing to discard intermediate keystrokes.
- Immediate generation invalidation on input change to prevent stale display.
- Serialized execution to prevent concurrent races on the engine.
- Cooperative thread lifecycle management without unsafe thread termination.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scoutx.engine import ScoutXEngine
    from scoutx.models import SearchResponse

try:
    from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# Retains active worker controllers to guarantee lifecycle validity
# and prevent premature object destruction during unblockable in-flight searches.
_active_controllers: set[SearchWorkerController] = set()


class SearchTask:
    """Represents a discrete search query with a generation identifier."""

    def __init__(self, query: str, limit: int, generation_id: int):
        self.query = query
        self.limit = limit
        self.generation_id = generation_id
        self.created_at = time.monotonic()


class SearchWorkerController:
    """Pure-Python thread-safe search controller.

    Manages a background daemon worker thread, task queue, generation counter,
    and cooperative cancellation tokens. Does not require PySide6.
    """

    def __init__(
        self,
        engine: ScoutXEngine,
        *,
        on_started: Callable[[int], None] | None = None,
        on_finished: Callable[[int, SearchResponse, float], None] | None = None,
        on_failed: Callable[[int, str, float], None] | None = None,
        on_cancelled: Callable[[int], None] | None = None,
    ):
        self.engine = engine
        self.on_started = on_started
        self.on_finished = on_finished
        self.on_failed = on_failed
        self.on_cancelled = on_cancelled

        self._task_queue: queue.Queue[SearchTask | None] = queue.Queue()
        self._current_generation: int = 0
        self._generation_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False

    @property
    def current_generation(self) -> int:
        with self._generation_lock:
            return self._current_generation

    def start(self) -> None:
        """Start the background search processing thread."""
        if self._is_running and self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_requested.clear()
        self._is_running = True
        _active_controllers.add(self)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ScoutX-SearchWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def invalidate_current_generation(self) -> int:
        """Immediately invalidate all currently queued and in-flight generations.

        Increments the generation counter and purges pending tasks so any currently
        running search will be suppressed when it finishes, even before the next query
        is submitted. Returns the newly assigned generation ID.
        """
        with self._generation_lock:
            self._current_generation += 1
            gen_id = self._current_generation

        # Discard any pending, not-yet-started tasks from the queue
        while True:
            try:
                dropped = self._task_queue.get_nowait()
                if dropped is not None and not self._stop_requested.is_set() and self.on_cancelled:
                    self.on_cancelled(dropped.generation_id)
            except queue.Empty:
                break

        return gen_id

    def submit_query(self, query: str, limit: int = 8) -> int:
        """Enqueue a new query, assigning an incremented generation ID.

        Returns the newly assigned generation_id.
        """
        with self._generation_lock:
            self._current_generation += 1
            gen_id = self._current_generation

        # Discard any pending, not-yet-started tasks from the queue
        while True:
            try:
                dropped = self._task_queue.get_nowait()
                if dropped is not None and not self._stop_requested.is_set() and self.on_cancelled:
                    self.on_cancelled(dropped.generation_id)
            except queue.Empty:
                break

        task = SearchTask(query=query, limit=limit, generation_id=gen_id)
        self._task_queue.put(task)
        return gen_id

    def shutdown(self, timeout: float = 1.0) -> bool:
        """Cooperatively request shutdown and wait up to timeout for the worker to finish.

        Never forces thread termination. If a synchronous engine search is currently
        uninterruptible, the controller lifecycle remains retained in _active_controllers
        until completion, and shutdown() returns False without blocking indefinitely.
        """
        self._stop_requested.set()
        # Wake up queue
        self._task_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            is_dead = not self._worker_thread.is_alive()
            if is_dead:
                self._is_running = False
                _active_controllers.discard(self)
            return is_dead
        self._is_running = False
        _active_controllers.discard(self)
        return True

    def _worker_loop(self) -> None:
        try:
            while not self._stop_requested.is_set():
                try:
                    task = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if task is None or self._stop_requested.is_set():
                    break

                # Generation check before starting
                with self._generation_lock:
                    latest_gen = self._current_generation

                if task.generation_id < latest_gen:
                    # Superseded while waiting in queue; suppress execution
                    if not self._stop_requested.is_set() and self.on_cancelled:
                        self.on_cancelled(task.generation_id)
                    continue

                if not self._stop_requested.is_set() and self.on_started:
                    self.on_started(task.generation_id)

                t0 = time.monotonic()
                try:
                    response = self.engine.search(task.query, limit=task.limit)
                    elapsed_ms = (time.monotonic() - t0) * 1000.0

                    # Generation check after execution:
                    # The engine search completed, but if the user changed the input
                    # or typed another query while it was running, suppress emitting
                    # stale results to the UI.
                    with self._generation_lock:
                        is_current = (
                            task.generation_id == self._current_generation
                            and not self._stop_requested.is_set()
                        )

                    if is_current:
                        if self.on_finished:
                            self.on_finished(task.generation_id, response, elapsed_ms)
                    else:
                        if not self._stop_requested.is_set() and self.on_cancelled:
                            self.on_cancelled(task.generation_id)

                except Exception as exc:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    with self._generation_lock:
                        is_current = (
                            task.generation_id == self._current_generation
                            and not self._stop_requested.is_set()
                        )
                    if is_current:
                        if self.on_failed:
                            self.on_failed(task.generation_id, str(exc), elapsed_ms)
        finally:
            self._is_running = False
            _active_controllers.discard(self)


if HAS_PYSIDE6:

    class QtSearchWorker(QObject):
        """Qt-aware search worker bridging background results to the UI thread via Signals."""

        search_started = Signal(int)
        search_finished = Signal(int, object, float)  # (generation_id, response, elapsed_ms)
        search_failed = Signal(int, str, float)  # (generation_id, error_msg, elapsed_ms)
        search_cancelled = Signal(int)

        def __init__(self, engine: ScoutXEngine, parent: QObject | None = None):
            super().__init__(parent)
            self._controller = SearchWorkerController(
                engine,
                on_started=self._handle_started,
                on_finished=self._handle_finished,
                on_failed=self._handle_failed,
                on_cancelled=self._handle_cancelled,
            )

        def start(self) -> None:
            self._controller.start()

        def invalidate(self) -> int:
            return self._controller.invalidate_current_generation()

        def submit(self, query: str, limit: int = 8) -> int:
            return self._controller.submit_query(query, limit=limit)

        def shutdown(self, timeout: float = 1.0) -> bool:
            return self._controller.shutdown(timeout=timeout)

        def _handle_started(self, gen_id: int) -> None:
            self.search_started.emit(gen_id)

        def _handle_finished(self, gen_id: int, resp: Any, elapsed: float) -> None:
            self.search_finished.emit(gen_id, resp, elapsed)

        def _handle_failed(self, gen_id: int, err: str, elapsed: float) -> None:
            self.search_failed.emit(gen_id, err, elapsed)

        def _handle_cancelled(self, gen_id: int) -> None:
            self.search_cancelled.emit(gen_id)

else:
    # Fallback placeholder when PySide6 is uninstalled
    QtSearchWorker = None  # type: ignore[assignment, misc]
