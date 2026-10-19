"""Deterministic unit and integration tests for Slice 6:
ScoutX Spotlight-style Desktop Launcher.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

from scoutx.cli import main as cli_main
from scoutx.gui_worker import SearchWorkerController, _active_controllers
from scoutx.models import (
    FileDocument,
    ParsedQuery,
    ReasoningStatus,
    Recommendation,
    Reflection,
    RetrievalTool,
    RouterResult,
    RouterTraceStep,
    SearchCandidate,
    SearchLane,
    SearchResponse,
    SearchStrategy,
)
from scoutx.platform_ops import (
    open_file_in_system,
    record_file_open,
    validate_target_path,
)
from scoutx.settings import (
    LauncherSettings,
    get_default_config_dir,
    get_default_data_dir,
    load_settings,
    save_settings,
)
from scoutx.storage import UsageLog, UserProfileStore


class Slice6LauncherTests(unittest.TestCase):
    def test_01_settings_defaults_and_os_path_resolution(self) -> None:
        # macOS directory resolution
        mac_cfg = get_default_config_dir("darwin")
        mac_data = get_default_data_dir("darwin")
        self.assertIn("Library/Application Support/ScoutX", str(mac_cfg))
        self.assertIn("Library/Application Support/ScoutX", str(mac_data))

        # Windows directory resolution
        with patch.dict(
            os.environ,
            {
                "APPDATA": "C:\\Users\\Test\\AppData\\Roaming",
                "LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local",
            },
        ):
            win_cfg = get_default_config_dir("win32")
            win_data = get_default_data_dir("win32")
            self.assertEqual(win_cfg.name, "ScoutX")
            self.assertIn("Roaming", str(win_cfg))
            self.assertEqual(win_data.name, "ScoutX")
            self.assertIn("Local", str(win_data))

        # Linux / XDG directory resolution
        with patch.dict(
            os.environ,
            {
                "XDG_CONFIG_HOME": "/tmp/custom_config",
                "XDG_DATA_HOME": "/tmp/custom_data",
            },
        ):
            linux_cfg = get_default_config_dir("linux")
            linux_data = get_default_data_dir("linux")
            self.assertEqual(str(linux_cfg), "/tmp/custom_config/scoutx")
            self.assertEqual(str(linux_data), "/tmp/custom_data/scoutx")

        # Secret rejection in settings
        s = LauncherSettings(
            llm_api_key_env="sk-secret-with spaces-is-invalid",
            embedding_api_key_env="secret with space",
            result_limit=100,  # Clamped to 50
            debounce_ms=-20,  # Clamped to 0
        )
        self.assertIsNone(s.llm_api_key_env)
        self.assertIsNone(s.embedding_api_key_env)
        self.assertEqual(s.result_limit, 50)
        self.assertEqual(s.debounce_ms, 0)

    def test_02_settings_persistence_and_malformed_json_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"

            # 1. Save and reload roundtrip
            initial = LauncherSettings(
                search_root="/test/search/path",
                result_limit=12,
                debounce_ms=200,
                hide_on_focus_loss=False,
            )
            save_settings(initial, cfg_path)
            loaded, diags = load_settings(cfg_path)
            self.assertEqual(len(diags), 0)
            self.assertEqual(loaded.search_root, "/test/search/path")
            self.assertEqual(loaded.result_limit, 12)
            self.assertEqual(loaded.debounce_ms, 200)
            self.assertFalse(loaded.hide_on_focus_loss)

            # 2. Corrupted JSON file recovery
            cfg_path.write_text("{this is not valid json!}", encoding="utf-8")
            recovered, err_diags = load_settings(cfg_path)
            self.assertEqual(len(err_diags), 1)
            self.assertIn("Failed to read configuration", err_diags[0])
            self.assertEqual(recovered.result_limit, 8)  # Safe default

            # 3. Non-dict JSON recovery
            cfg_path.write_text('["item1", "item2"]', encoding="utf-8")
            recovered_arr, arr_diags = load_settings(cfg_path)
            self.assertEqual(len(arr_diags), 1)
            self.assertIn("not a JSON object", arr_diags[0])
            self.assertEqual(recovered_arr.result_limit, 8)

    def test_03_platform_file_opener_and_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "report.pdf"
            test_file.write_text("dummy report content", encoding="utf-8")

            # 1. Non-existent file validation
            missing_path = Path(tmpdir) / "does_not_exist.txt"
            self.assertIsNone(validate_target_path(missing_path))
            success, err = open_file_in_system(missing_path)
            self.assertFalse(success)
            self.assertIn("does not exist", err or "")

            # 2. Existing file validation
            valid_p = validate_target_path(test_file)
            self.assertIsNotNone(valid_p)

            # 3. macOS opener (open without shell)
            with patch("subprocess.Popen") as mock_popen:
                success_mac, err_mac = open_file_in_system(test_file, platform_name="darwin")
                self.assertTrue(success_mac)
                self.assertIsNone(err_mac)
                mock_popen.assert_called_once()
                args, kwargs = mock_popen.call_args
                self.assertEqual(args[0], ["open", str(test_file.resolve())])
                self.assertNotIn("shell", kwargs)

            # 4. Linux opener (xdg-open without shell)
            with patch("subprocess.Popen") as mock_popen_linux:
                success_lin, err_lin = open_file_in_system(test_file, platform_name="linux")
                self.assertTrue(success_lin)
                self.assertIsNone(err_lin)
                mock_popen_linux.assert_called_once()
                args, kwargs = mock_popen_linux.call_args
                self.assertEqual(args[0], ["xdg-open", str(test_file.resolve())])

            # 5. Opener failure / missing executable
            with patch("subprocess.Popen", side_effect=FileNotFoundError("open not found")):
                success_fail, err_fail = open_file_in_system(test_file, platform_name="darwin")
                self.assertFalse(success_fail)
                self.assertIn("opener utility not found", err_fail or "")

    def test_04_exactly_once_file_open_logging_regression(self) -> None:
        """Regression test: prove that recording a file-open event via record_file_open
        with UsageLog increments profile open counts by EXACTLY 1, avoiding duplicate counts.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "usage.db"
            test_doc_path = Path(tmpdir) / "quarterly_results.xlsx"
            test_doc_path.write_text("data", encoding="utf-8")
            doc = FileDocument(
                id="doc-xlsx-101",
                path=test_doc_path,
                title="Quarterly Results",
                extension="xlsx",
                content="",
            )

            usage_log = UsageLog(db_path)
            profile_store = UserProfileStore(db_path)

            # Initial profile count should be 0
            profile_initial = profile_store.load_profile()
            self.assertEqual(profile_initial.file_opens.get("doc-xlsx-101", 0), 0)
            self.assertEqual(profile_initial.preferred_extensions.get("xlsx", 0), 0)

            # 1. Record single open event via record_file_open
            logged = record_file_open(
                doc,
                usage_log=usage_log,
                profile_store=profile_store,
                topics=("finance",),
            )
            self.assertTrue(logged)

            # Assert count is EXACTLY 1 (not 2)
            profile_after = profile_store.load_profile()
            self.assertEqual(profile_after.file_opens.get("doc-xlsx-101", 1), 1)
            self.assertEqual(profile_after.preferred_extensions.get("xlsx", 0), 1)
            self.assertEqual(profile_after.topic_interests.get("finance", 0), 1)

            # 2. Test fallback when usage_log is None (direct profile_store call)
            doc2 = FileDocument(
                id="doc-pdf-202",
                path=test_doc_path,
                title="Document PDF",
                extension="pdf",
                content="",
            )
            logged2 = record_file_open(doc2, usage_log=None, profile_store=profile_store)
            self.assertTrue(logged2)
            profile_after2 = profile_store.load_profile()
            self.assertEqual(profile_after2.file_opens.get("doc-pdf-202", 0), 1)

    def test_05_search_worker_debouncing_and_generation_guard(self) -> None:
        """Verify that the SearchWorkerController increments generations, suppresses
        obsolete results, and only emits results for the latest generation.
        """
        mock_engine = MagicMock()

        # Engine returns response with delay
        def slow_search(query: str, limit: int = 8) -> SearchResponse:
            time.sleep(0.05)
            doc = FileDocument(
                id=f"doc-{query}",
                path=Path(f"/{query}.txt"),
                title=query,
                extension="txt",
                content="",
            )
            cand = SearchCandidate(document=doc, score=0.95, tool=RetrievalTool.KEYWORD)
            rec = Recommendation(
                document=doc,
                score=0.95,
                explanation=f"Found {query}",
                source_candidate=cand,
            )
            step = RouterTraceStep(
                iteration=0,
                complexity_score=0.1,
                strategy=SearchStrategy(lane=SearchLane.FAST, tools=(RetrievalTool.KEYWORD,)),
                candidates_seen=1,
                reflection=Reflection(
                    confident=True,
                    coverage=1,
                    top_score=0.95,
                    relevance_to_intent=0.95,
                    reason="direct match",
                ),
            )
            res = RouterResult(
                parsed_query=ParsedQuery(raw=query, intent="search"),
                candidates=(cand,),
                trace=(step,),
                forced_exit=False,
            )
            return SearchResponse(recommendations=(rec,), router_result=res)

        mock_engine.search.side_effect = slow_search

        started_gens: list[int] = []
        finished_gens: list[int] = []
        cancelled_gens: list[int] = []

        controller = SearchWorkerController(
            mock_engine,
            on_started=lambda gen: started_gens.append(gen),
            on_finished=lambda gen, resp, elapsed: finished_gens.append(gen),
            on_cancelled=lambda gen: cancelled_gens.append(gen),
        )
        controller.start()
        try:
            # Rapid query submissions (simulating typing "a", "ab", "abc")
            gen1 = controller.submit_query("a")
            gen2 = controller.submit_query("ab")
            gen3 = controller.submit_query("abc")

            self.assertEqual(gen1, 1)
            self.assertEqual(gen2, 2)
            self.assertEqual(gen3, 3)

            # Wait for execution to settle
            time.sleep(0.35)

            # Only the final generation should have finished and emitted to UI
            self.assertIn(3, finished_gens)
            # Generations 1 and 2 should have been suppressed or cancelled
            self.assertNotIn(1, finished_gens)
            self.assertNotIn(2, finished_gens)
        finally:
            controller.shutdown(timeout=0.5)

    def test_06_worker_cooperative_shutdown_without_termination(self) -> None:
        """Verify that shutting down the search worker stops the worker thread
        cleanly within a bounded timeout without forcibly killing the thread.
        """
        mock_engine = MagicMock()
        step = RouterTraceStep(
            iteration=0,
            complexity_score=0.0,
            strategy=SearchStrategy(lane=SearchLane.FAST, tools=()),
            candidates_seen=0,
            reflection=Reflection(
                confident=True,
                coverage=0,
                top_score=0.0,
                relevance_to_intent=0.0,
                reason="ok",
            ),
        )
        res = RouterResult(
            parsed_query=ParsedQuery(raw="", intent=""),
            candidates=(),
            trace=(step,),
            forced_exit=False,
        )
        mock_engine.search.return_value = SearchResponse(recommendations=(), router_result=res)

        controller = SearchWorkerController(mock_engine)
        controller.start()
        self.assertTrue(controller._is_running)
        assert controller._worker_thread is not None
        self.assertTrue(controller._worker_thread.is_alive())

        stopped_cleanly = controller.shutdown(timeout=1.0)
        self.assertTrue(stopped_cleanly)
        self.assertFalse(controller._worker_thread.is_alive())
        self.assertFalse(controller._is_running)

    def test_07_cli_launch_and_gui_alias_missing_pyside6_exit_code_2(self) -> None:
        """Verify that 'scoutx launch' and 'scoutx gui' output a helpful diagnostic
        and exit with status code 2 when PySide6 is not installed.
        """
        with patch.dict(sys.modules, {"scoutx.launcher": None}):
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                exit_code_launch = cli_main(["launch"])
            self.assertEqual(exit_code_launch, 2)
            self.assertIn("GUI dependencies not installed", err_buf.getvalue())
            self.assertIn("pip install 'scoutx[gui]'", err_buf.getvalue())

            err_buf_gui = io.StringIO()
            with redirect_stderr(err_buf_gui):
                exit_code_gui = cli_main(["gui"])
            self.assertEqual(exit_code_gui, 2)
            self.assertIn("GUI dependencies not installed", err_buf_gui.getvalue())

    def test_08_pyproject_packaging_zero_mandatory_dependencies(self) -> None:
        """Verify that pyproject.toml preserves dependencies = [] and defines
        gui under optional-dependencies.
        """
        pyproject_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_file.read_text(encoding="utf-8")

        self.assertIn('dependencies = []', content)
        self.assertIn('[project.optional-dependencies]', content)
        self.assertIn('gui = ["PySide6>=6.5.0"]', content)

    def test_09_truthful_warning_states_and_reflection_notices(self) -> None:
        """Verify that forced-exit and low-confidence states from the engine router
        are faithfully represented without fabricating confidence.
        """
        doc = FileDocument(id="doc-1", path=Path("/docs/notes.md"), title="Notes", extension="md", content="")
        cand = SearchCandidate(document=doc, score=0.45, tool=RetrievalTool.KEYWORD)
        rec = Recommendation(document=doc, score=0.45, explanation="Best effort match", source_candidate=cand)

        # 1. Normal confident search response
        step_normal = RouterTraceStep(
            iteration=0,
            complexity_score=0.2,
            strategy=SearchStrategy(lane=SearchLane.STANDARD, tools=()),
            candidates_seen=1,
            reflection=Reflection(
                confident=True,
                coverage=1,
                top_score=0.8,
                relevance_to_intent=0.8,
                reason="High keyword overlap",
            ),
        )
        res_normal = RouterResult(
            parsed_query=ParsedQuery(raw="notes", intent="find"),
            candidates=(cand,),
            trace=(step_normal,),
            forced_exit=False,
        )
        resp_normal = SearchResponse(recommendations=(rec,), router_result=res_normal)
        self.assertFalse(resp_normal.router_result.forced_exit)
        self.assertTrue(resp_normal.router_result.trace[-1].reflection.confident)

        # 2. Forced exit after escalation cap
        step_forced = RouterTraceStep(
            iteration=2,
            complexity_score=0.8,
            strategy=SearchStrategy(lane=SearchLane.DEEP, tools=()),
            candidates_seen=1,
            reflection=Reflection(
                confident=False,
                coverage=1,
                top_score=0.45,
                relevance_to_intent=0.45,
                reason="Low relevance after deep scan",
            ),
            forced_exit=True,
        )
        res_forced = RouterResult(
            parsed_query=ParsedQuery(raw="vague query", intent="find"),
            candidates=(cand,),
            trace=(step_forced,),
            forced_exit=True,
        )
        resp_forced = SearchResponse(recommendations=(rec,), router_result=res_forced)
        self.assertTrue(resp_forced.router_result.forced_exit)
        self.assertFalse(resp_forced.router_result.trace[-1].reflection.confident)

    def test_10_immediate_invalidation_on_input_change_during_debounce(self) -> None:
        """Deterministic regression test:
        1. Start a search for query A and keep it blocked.
        2. Change input to query B (calls invalidate_current_generation()),
           keeping B's debounce timer unexpired (B not yet submitted).
        3. Allow query A to complete.
        4. Verify that A's result is suppressed and never emitted.
        5. Allow B's debounce and search to complete; verify B's result is displayed.
        6. Verify clearing input while a search is running suppresses the search result.
        """
        block_a = threading.Event()
        block_c = threading.Event()
        search_a_started = threading.Event()
        search_c_started = threading.Event()

        def controlled_search(query: str, limit: int = 8) -> SearchResponse:
            if query == "query_A":
                search_a_started.set()
                block_a.wait(timeout=5.0)
            elif query == "query_C":
                search_c_started.set()
                block_c.wait(timeout=5.0)

            doc = FileDocument(id=f"doc-{query}", path=Path(f"/{query}"), title=query, extension="", content="")
            cand = SearchCandidate(document=doc, score=0.9, tool=RetrievalTool.KEYWORD)
            rec = Recommendation(document=doc, score=0.9, explanation=f"Result for {query}", source_candidate=cand)
            step = RouterTraceStep(
                iteration=0,
                complexity_score=0.1,
                strategy=SearchStrategy(lane=SearchLane.FAST, tools=()),
                candidates_seen=1,
                reflection=Reflection(confident=True, coverage=1, top_score=0.9, relevance_to_intent=0.9, reason="ok"),
            )
            res = RouterResult(parsed_query=ParsedQuery(raw=query, intent="find"), candidates=(cand,), trace=(step,))
            return SearchResponse(recommendations=(rec,), router_result=res)

        mock_engine = MagicMock()
        mock_engine.search.side_effect = controlled_search

        emitted_results: list[tuple[int, str]] = []
        cancelled_gens: list[int] = []

        controller = SearchWorkerController(
            mock_engine,
            on_finished=lambda gen, resp, elapsed: emitted_results.append(
                (gen, resp.recommendations[0].document.title)
            ),
            on_cancelled=lambda gen: cancelled_gens.append(gen),
        )
        controller.start()
        try:
            # 1. Submit query A (gen 1)
            gen_a = controller.submit_query("query_A")
            self.assertEqual(gen_a, 1)

            # Wait until query A is confirmed executing inside engine.search
            self.assertTrue(search_a_started.wait(timeout=1.0))

            # 2. Input changes to "query_B", triggering immediate invalidation
            # but debounce timer has NOT expired yet (query B not yet submitted)
            invalidated_gen = controller.invalidate_current_generation()
            self.assertEqual(invalidated_gen, 2)

            # 3. Release query A
            block_a.set()
            time.sleep(0.05)  # Allow worker loop to process completion of search A

            # 4. Verify A's result was NOT emitted
            self.assertEqual(len(emitted_results), 0)
            self.assertIn(gen_a, cancelled_gens)

            # 5. Debounce timer for query B now expires -> submit query B (gen 3)
            gen_b = controller.submit_query("query_B")
            self.assertEqual(gen_b, 3)
            time.sleep(0.1)

            # Verify B's results are emitted normally
            self.assertEqual(len(emitted_results), 1)
            self.assertEqual(emitted_results[0], (3, "query_B"))

            # 6. Test clearing input while search C is running
            emitted_results.clear()
            gen_c = controller.submit_query("query_C")
            self.assertTrue(search_c_started.wait(timeout=1.0))

            # User clears input -> invalidate called
            controller.invalidate_current_generation()

            # Release C
            block_c.set()
            time.sleep(0.05)

            # Verify C is suppressed; nothing emitted for cleared input
            self.assertEqual(len(emitted_results), 0)
        finally:
            controller.shutdown(timeout=0.5)

    def test_11_cooperative_shutdown_with_uninterruptible_search(self) -> None:
        """Deterministic test for shutdown during an uninterruptible search:
        1. Start a search that blocks until explicitly released.
        2. Close the launcher UI while that search is running.
        3. Verify that the UI event handler returns without waiting indefinitely
           and that no running thread is forcibly terminated.
        4. Verify that worker objects are not destroyed while their thread is still active.
        5. Release the blocked operation and verify that shutdown completes cleanly,
           with no lingering worker thread or process and no Qt thread-destruction warning.
        """
        search_started = threading.Event()
        release_search = threading.Event()

        def uninterruptible_search(query: str, limit: int = 8) -> SearchResponse:
            search_started.set()
            release_search.wait(timeout=5.0)
            doc = FileDocument(id="doc-hang", path=Path("/hang"), title="hang", extension="", content="")
            cand = SearchCandidate(document=doc, score=0.5, tool=RetrievalTool.KEYWORD)
            rec = Recommendation(document=doc, score=0.5, explanation="Done", source_candidate=cand)
            step = RouterTraceStep(
                iteration=0,
                complexity_score=0.0,
                strategy=SearchStrategy(lane=SearchLane.FAST, tools=()),
                candidates_seen=0,
                reflection=Reflection(confident=True, coverage=0, top_score=0.0, relevance_to_intent=0.0, reason=""),
            )
            res = RouterResult(parsed_query=ParsedQuery(raw=query, intent=""), candidates=(), trace=(step,))
            return SearchResponse(recommendations=(rec,), router_result=res)

        mock_engine = MagicMock()
        mock_engine.search.side_effect = uninterruptible_search

        emitted_after_shutdown: list[Any] = []
        controller = SearchWorkerController(
            mock_engine,
            on_finished=lambda gen, resp, elapsed: emitted_after_shutdown.append(resp),
        )
        controller.start()

        # Mock launcher UI mirroring LauncherWindow closeEvent / _cleanup_worker
        class MockEvent:
            pass

        class MockLauncherUI:
            def __init__(self, engine: Any, worker: SearchWorkerController):
                self.engine = engine
                self.worker = worker
                self._debounce_timer_active = True

            def closeEvent(self, event: Any) -> None:
                self._cleanup_worker()

            def _cleanup_worker(self) -> None:
                self._debounce_timer_active = False
                if hasattr(self, "worker") and self.worker:
                    self.worker.invalidate_current_generation()
                    self.worker.shutdown(timeout=0.05)

        launcher_ui = MockLauncherUI(mock_engine, controller)

        # 1. Start blocked search
        controller.submit_query("blocking_query")
        self.assertTrue(search_started.wait(timeout=1.0))

        # 2. Close the launcher while that search is running
        t0 = time.monotonic()
        launcher_ui.closeEvent(MockEvent())
        t_elapsed = time.monotonic() - t0

        # 3. Verify that the UI event handler returns without waiting indefinitely
        # and that no running thread is forcibly terminated.
        self.assertLess(t_elapsed, 0.25)
        assert controller._worker_thread is not None
        # Thread was NOT killed
        self.assertTrue(controller._worker_thread.is_alive())

        # 4. Verify that worker objects are not destroyed while their thread is still active.
        self.assertIn(controller, _active_controllers)

        # 5. Release the blocked operation and verify that shutdown completes cleanly,
        # with no lingering worker thread or process and no Qt thread-destruction warning.
        release_search.set()

        # Wait for thread to finish cleanly
        controller._worker_thread.join(timeout=1.0)
        self.assertFalse(controller._worker_thread.is_alive())

        # No results should have been emitted after shutdown was requested
        self.assertEqual(len(emitted_after_shutdown), 0)

        # Controller is cleanly removed from _active_controllers
        self.assertNotIn(controller, _active_controllers)


if __name__ == "__main__":
    unittest.main()
