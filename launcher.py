"""Spotlight-style desktop file discovery launcher for ScoutX.

Features:
- Floating, centered, frameless keyboard-first window.
- Instant reactive search input with 150 ms debounce.
- Immediate generation invalidation on input change to suppress stale results.
- Up/Down arrow list navigation, Enter to open, Escape to clear/dismiss.
- Truthful display of grounded explanations, reflection confidence, and forced-exit warnings.
- Cooperative, non-blocking background search execution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scoutx.engine import ScoutXEngine
    from scoutx.models import Recommendation, SearchCandidate, SearchResponse

from scoutx.platform_ops import open_file_in_system, record_file_open
from scoutx.settings import LauncherSettings, load_settings

try:
    from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QKeyEvent, QPalette
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from scoutx.gui_worker import QtSearchWorker
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


if HAS_PYSIDE6:

    class CandidateListItemWidget(QWidget):
        """Custom rendered item in the search results list."""

        def __init__(self, rec: Recommendation, parent: QWidget | None = None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(2)

            top_row = QHBoxLayout()
            top_row.setSpacing(8)

            self.title_label = QLabel(rec.document.title or rec.document.path.name)
            font = self.title_label.font()
            font.setPointSize(13)
            font.setBold(True)
            self.title_label.setFont(font)
            top_row.addWidget(self.title_label)

            ext_text = rec.document.extension or "file"
            self.badge = QLabel(f"[{ext_text}]")
            self.badge.setStyleSheet("color: #718096; font-size: 11px;")
            top_row.addWidget(self.badge)

            top_row.addStretch()

            score_val = rec.score
            self.score_label = QLabel(f"{score_val:.2f}")
            self.score_label.setStyleSheet("color: #4a5568; font-size: 11px;")
            top_row.addWidget(self.score_label)

            layout.addLayout(top_row)

            # Secondary path context
            parent_dir = str(rec.document.path.parent)
            self.path_label = QLabel(parent_dir)
            self.path_label.setStyleSheet("color: #a0aec0; font-size: 11px;")
            layout.addWidget(self.path_label)


    class LauncherWindow(QDialog):
        """Spotlight-style floating desktop search dialog."""

        def __init__(
            self,
            engine: ScoutXEngine,
            settings: LauncherSettings | None = None,
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.engine = engine
            self.settings = settings or LauncherSettings()
            self._current_response: SearchResponse | None = None
            self._current_gen: int = 0

            self._init_window()
            self._init_ui()
            self._init_worker()

        def _init_window(self) -> None:
            self.setWindowTitle("ScoutX")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setFixedSize(680, 480)
            self._center_on_screen()

        def _center_on_screen(self) -> None:
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                x = geo.x() + (geo.width() - self.width()) // 2
                y = geo.y() + (geo.height() - self.height()) // 3  # Upper third (Spotlight style)
                self.move(x, y)

        def _init_ui(self) -> None:
            self.setStyleSheet(
                """
                QDialog {
                    background-color: #1a202c;
                    border: 1px solid #4a5568;
                    border-radius: 12px;
                }
                QLineEdit {
                    background-color: #2d3748;
                    color: #f7fafc;
                    border: 1px solid #4a5568;
                    border-radius: 8px;
                    padding: 10px 14px;
                    font-size: 16px;
                }
                QLineEdit:focus {
                    border: 1px solid #63b3ed;
                }
                QListWidget {
                    background-color: #2d3748;
                    border: 1px solid #4a5568;
                    border-radius: 8px;
                    color: #f7fafc;
                    outline: none;
                }
                QListWidget::item {
                    border-bottom: 1px solid #4a5568;
                    border-radius: 4px;
                }
                QListWidget::item:selected {
                    background-color: #3182ce;
                    color: #ffffff;
                }
                """
            )

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(16, 16, 16, 16)
            main_layout.setSpacing(10)

            # 1. Search Bar
            self.search_input = QLineEdit(self)
            self.search_input.setPlaceholderText("Search files with ScoutX...")
            self.search_input.textChanged.connect(self._on_search_text_changed)
            main_layout.addWidget(self.search_input)

            # 2. Results List
            self.results_list = QListWidget(self)
            self.results_list.currentRowChanged.connect(self._on_result_selected)
            self.results_list.itemDoubleClicked.connect(self._on_item_double_clicked)
            main_layout.addWidget(self.results_list, stretch=3)

            # 3. Details & Explanation Panel
            self.detail_frame = QFrame(self)
            self.detail_frame.setStyleSheet(
                "background-color: #232d3d; border-radius: 6px; padding: 6px;"
            )
            detail_layout = QVBoxLayout(self.detail_frame)
            detail_layout.setContentsMargins(10, 8, 10, 8)
            detail_layout.setSpacing(4)

            # Warning banner for forced exits or low confidence
            self.warning_label = QLabel(self)
            self.warning_label.setWordWrap(True)
            self.warning_label.setStyleSheet("color: #f6e05e; font-weight: bold; font-size: 11px;")
            self.warning_label.hide()
            detail_layout.addWidget(self.warning_label)

            # Grounded explanation
            self.explanation_label = QLabel("Type a query to search files.", self)
            self.explanation_label.setWordWrap(True)
            self.explanation_label.setStyleSheet("color: #e2e8f0; font-size: 12px;")
            detail_layout.addWidget(self.explanation_label)

            # Status footer (latency, lane, laya shadow status)
            self.footer_label = QLabel("", self)
            self.footer_label.setStyleSheet("color: #718096; font-size: 10px;")
            detail_layout.addWidget(self.footer_label)

            main_layout.addWidget(self.detail_frame, stretch=1)

            # Focus search bar by default
            self.search_input.setFocus()

        def _init_worker(self) -> None:
            # Debounce timer (150 ms default)
            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.setInterval(self.settings.debounce_ms)
            self._debounce_timer.timeout.connect(self._execute_debounced_search)

            # Background worker
            self.worker = QtSearchWorker(self.engine, parent=self)
            self.worker.search_started.connect(self._on_worker_started)
            self.worker.search_finished.connect(self._on_worker_finished)
            self.worker.search_failed.connect(self._on_worker_failed)
            self.worker.start()

        def _on_search_text_changed(self, text: str) -> None:
            # 1. Stop any pending debounced call
            self._debounce_timer.stop()

            # 2. Immediately invalidate all earlier generations
            # Prevents in-flight searches from displaying during the debounce window
            if hasattr(self, "worker") and self.worker:
                self._current_gen = self.worker.invalidate()

            trimmed = text.strip()
            if not trimmed:
                self.results_list.clear()
                self.warning_label.hide()
                self.explanation_label.setText("Type a query to search files.")
                self.footer_label.setText("")
                return

            # 3. Start debounce timer for the new query
            self._debounce_timer.start()

        def _execute_debounced_search(self) -> None:
            query = self.search_input.text().strip()
            if not query:
                return
            self._current_gen = self.worker.submit(query, limit=self.settings.result_limit)

        def _on_worker_started(self, gen_id: int) -> None:
            if gen_id == self._current_gen:
                self.footer_label.setText("Searching...")

        def _on_worker_finished(
            self, gen_id: int, response: SearchResponse, elapsed_ms: float
        ) -> None:
            if gen_id != self._current_gen:
                return  # Obsolete result; suppressed by generation guard

            self._current_response = response
            self.results_list.clear()

            recs = response.recommendations
            if not recs:
                self.warning_label.hide()
                self.explanation_label.setText(
                    f"No matching files found for '{self.search_input.text().strip()}'."
                )
                self.footer_label.setText(f"Completed in {elapsed_ms:.1f}ms (0 results)")
                return

            for rec in recs:
                item = QListWidgetItem(self.results_list)
                widget = CandidateListItemWidget(rec)
                item.setSizeHint(widget.sizeHint())
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, widget)

            # Auto-select the first recommendation
            self.results_list.setCurrentRow(0)

            # Update footer with lane and latency
            lane_str = "lexical"
            laya_str = ""
            if response.router_result and response.router_result.trace:
                first_step = response.router_result.trace[0]
                last_step = response.router_result.trace[-1]
                lane_str = last_step.strategy.lane.value
                if first_step.laya_advisory and first_step.laya_advisory.status == "ok":
                    s_lane = first_step.laya_advisory.suggested_lane
                    laya_str = f" | laya shadow: {s_lane.value if s_lane else 'none'}"

            self.footer_label.setText(
                f"{len(recs)} results in {elapsed_ms:.1f}ms | lane: {lane_str}{laya_str}"
            )

        def _on_worker_failed(self, gen_id: int, err_msg: str, elapsed_ms: float) -> None:
            if gen_id != self._current_gen:
                return
            self.results_list.clear()
            self.warning_label.setText("⚠ Search encountered an error")
            self.warning_label.show()
            self.explanation_label.setText(f"Error: {err_msg}")
            self.footer_label.setText(f"Failed after {elapsed_ms:.1f}ms")

        def _on_result_selected(self, row: int) -> None:
            if not self._current_response or row < 0:
                return
            recs = self._current_response.recommendations
            if row >= len(recs):
                return
            rec = recs[row]

            # Render truthful warnings
            warnings: list[str] = []
            if self._current_response.router_result:
                if self._current_response.router_result.forced_exit:
                    warnings.append(
                        "⚠ Escalation cap reached — showing best candidate despite low confidence"
                    )
                trace = self._current_response.router_result.trace
                if trace and not trace[-1].reflection.confident:
                    warnings.append("⚠ Low confidence search result")

            if warnings:
                self.warning_label.setText("\n".join(warnings))
                self.warning_label.show()
            else:
                self.warning_label.hide()

            # Render grounded explanation
            self.explanation_label.setText(rec.explanation or "No explanation available.")

        def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
            self._open_selected_result()

        def _open_selected_result(self) -> None:
            row = self.results_list.currentRow()
            if not self._current_response or row < 0:
                return
            recs = self._current_response.recommendations
            if row >= len(recs):
                return
            selected_rec = recs[row]
            doc = selected_rec.document

            # Open file natively without shell
            success, err = open_file_in_system(doc.path)
            if success:
                # Record file open exactly once
                record_file_open(
                    doc,
                    usage_log=self.engine.usage_log,
                    profile_store=self.engine.profile_store,
                )
                self.accept()  # Close dialog
            else:
                self.warning_label.setText(f"⚠ Could not open file: {err}")
                self.warning_label.show()

        def keyPressEvent(self, event: QKeyEvent) -> None:
            key = event.key()

            # Escape: clear query if not empty, otherwise close launcher
            if key == Qt.Key.Key_Escape:
                if self.search_input.text():
                    self.search_input.clear()
                else:
                    self.reject()
                return

            # Enter / Return: open highlighted result
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._open_selected_result()
                return

            # Up / Down navigation
            if key == Qt.Key.Key_Down:
                if self.search_input.hasFocus():
                    if self.results_list.count() > 0:
                        self.results_list.setFocus()
                        self.results_list.setCurrentRow(0)
                    return
                else:
                    curr = self.results_list.currentRow()
                    if curr < self.results_list.count() - 1:
                        self.results_list.setCurrentRow(curr + 1)
                    return

            if key == Qt.Key.Key_Up:
                if not self.search_input.hasFocus():
                    curr = self.results_list.currentRow()
                    if curr > 0:
                        self.results_list.setCurrentRow(curr - 1)
                    else:
                        self.search_input.setFocus()
                    return

            # Cmd+R or Ctrl+R: immediate re-run search bypassing debounce
            if key == Qt.Key.Key_R and (
                event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            ):
                self._debounce_timer.stop()
                self._execute_debounced_search()
                return

            super().keyPressEvent(event)

        def changeEvent(self, event: QEvent) -> None:
            # Auto-hide on focus loss if configured
            if event.type() == QEvent.Type.ActivationChange:
                if not self.isActiveWindow() and self.settings.hide_on_focus_loss:
                    self.reject()
            super().changeEvent(event)

        def _cleanup_worker(self) -> None:
            """Safely stop debouncing and request cooperative worker shutdown."""
            if hasattr(self, "_debounce_timer") and self._debounce_timer.isActive():
                self._debounce_timer.stop()
            if hasattr(self, "worker") and self.worker:
                self.worker.invalidate()
                self.worker.shutdown(timeout=0.1)

        def accept(self) -> None:
            self._cleanup_worker()
            super().accept()

        def reject(self) -> None:
            self._cleanup_worker()
            super().reject()

        def closeEvent(self, event: Any) -> None:
            # Cooperative shutdown: stop debounce timer and request worker stop
            self._cleanup_worker()
            super().closeEvent(event)


def run_launcher(args: Any = None) -> int:
    """Launch the ScoutX Spotlight desktop interface."""
    if not HAS_PYSIDE6:
        raise ImportError(
            "PySide6 is required to run the ScoutX GUI launcher.\n"
            "Install with: pip install 'scoutx[gui]' (or pip install PySide6)"
        )

    # Initialize Qt Application
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ScoutX")

    # Load settings
    config_path = getattr(args, "config", None) if args else None
    settings, diagnostics = load_settings(config_path)

    # Determine search root
    search_root = getattr(args, "root", None) or settings.search_root
    usage_db = getattr(args, "usage_db", None) or settings.usage_db
    index_db = getattr(args, "index_db", None) or settings.index_db

    # Instantiate engine
    from scoutx.engine import ScoutXEngine

    engine = ScoutXEngine.from_directory(
        search_root,
        usage_db=usage_db,
        index_db=index_db,
    )

    window = LauncherWindow(engine, settings=settings)
    window.show()
    return app.exec()
