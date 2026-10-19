"""Baseline characterization tests for ScoutX (Slice 1)."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

from scoutx import __version__
from scoutx.cli import main as cli_main
from scoutx.document_loader import DocumentLoader
from scoutx.downstream import (
    ContextMemoryModule,
    ExplanationModule,
    PersonalizationModule,
    RerankingModule,
)
from scoutx.engine import ScoutXEngine
from scoutx.models import (
    FileDocument,
    Reflection,
    RetrievalTool,
    RouterResult,
    RouterTraceStep,
    SearchCandidate,
    SearchLane,
    SearchStrategy,
    UserProfile,
)
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent
from scoutx.storage import UsageLog


class BaselineCLITests(unittest.TestCase):
    def test_cli_version_flag_prints_package_version(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = cli_main(["--version"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(buf.getvalue().strip(), f"ScoutX {__version__}")

    def test_cli_search_and_log_open_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            target_file = root / "budget_2026.csv"
            target_file.write_text("department,budget,cost,forecast\nfinance,1000,800,900\n", encoding="utf-8")
            db_path = Path(tmpdir) / "usage.sqlite"

            open_buf = io.StringIO()
            with redirect_stdout(open_buf):
                open_code = cli_main(["log-open", str(target_file), "--usage-db", str(db_path), "--topic", "budget"])
            self.assertEqual(open_code, 0)
            self.assertIn("Recorded open event", open_buf.getvalue())

            search_buf = io.StringIO()
            with redirect_stdout(search_buf):
                search_code = cli_main(
                    ["search", "find the budget sheet", "--root", str(root), "--usage-db", str(db_path), "--trace"]
                )
            self.assertEqual(search_code, 0)
            output = search_buf.getvalue()
            self.assertIn("budget_2026.csv", output)
            self.assertIn("Router trace:", output)


class BaselineQueryUnderstandingTests(unittest.TestCase):
    def test_parses_intent_date_window_file_types_and_keywords(self) -> None:
        fixed_now = datetime(2026, 10, 19, 12, 0, 0)
        parser = QueryUnderstandingModule(now=fixed_now)

        parsed = parser.parse("the pricing deck I was working on last week")
        self.assertEqual(parsed.intent, "find")
        self.assertEqual(parsed.keywords, ("pricing",))
        self.assertEqual(parsed.file_types, ("key", "ppt", "pptx"))
        self.assertIsNotNone(parsed.date_window)
        assert parsed.date_window is not None
        self.assertEqual(parsed.date_window.label, "last week")
        self.assertEqual(parsed.filename_hints, ())
        self.assertGreater(parsed.parser_confidence, 0.5)

    def test_parses_quoted_and_explicit_filename_hints(self) -> None:
        parser = QueryUnderstandingModule(now=datetime(2026, 10, 19, 12, 0, 0))
        parsed = parser.parse('open "Pricing_v3.pptx"')
        self.assertEqual(parsed.intent, "find")
        self.assertIn("Pricing_v3.pptx", parsed.filename_hints)
        self.assertIn("pptx", parsed.file_types)

    def test_vague_query_produces_high_ambiguity(self) -> None:
        parser = QueryUnderstandingModule(now=datetime(2026, 10, 19, 12, 0, 0))
        parsed = parser.parse("that thing about numbers from a while ago")
        self.assertGreaterEqual(parsed.ambiguity_score, 0.5)
        self.assertIsNotNone(parsed.date_window)
        assert parsed.date_window is not None
        self.assertEqual(parsed.date_window.label, "a while ago")


class BaselineRetrievalAndRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.now = datetime(2026, 10, 19, 12, 0, 0)

        self.pricing_file = self.root / "Pricing_v3.pptx"
        self.pricing_file.write_text("enterprise pricing forecast revenue deck slides", encoding="utf-8")
        last_week_ts = (self.now - timedelta(days=3)).timestamp()
        os.utime(self.pricing_file, (last_week_ts, last_week_ts))

        self.notes_file = self.root / "meeting_notes.md"
        self.notes_file.write_text("general architecture discussion and weekly sync", encoding="utf-8")

        self.corpus = DocumentCorpus.from_directory(self.root)
        self.toolbox = RetrievalToolbox(self.corpus)
        self.parser = QueryUnderstandingModule(now=self.now)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_all_five_retrieval_tools_execute_without_error(self) -> None:
        parsed = self.parser.parse("the pricing deck I was working on last week")
        for tool in RetrievalTool:
            candidates = self.toolbox.search(tool, parsed, limit=5)
            self.assertIsInstance(candidates, tuple)
            self.assertGreaterEqual(len(candidates), 1)
            self.assertEqual(candidates[0].tool, tool)

    def test_router_fast_lane_for_explicit_filename_query(self) -> None:
        router = ComplexityRoutingAgent(self.toolbox)
        parsed = self.parser.parse('"Pricing_v3.pptx"')
        result = router.run(parsed)
        self.assertGreaterEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].strategy.lane, SearchLane.FAST)
        self.assertFalse(result.forced_exit)
        self.assertEqual(result.candidates[0].document.path.name, "Pricing_v3.pptx")

    def test_router_iteration_guard_caps_escalations_when_no_candidates_match(self) -> None:
        empty_corpus = DocumentCorpus(())
        router = ComplexityRoutingAgent(RetrievalToolbox(empty_corpus), max_escalations=2)
        parsed = self.parser.parse("nonexistent_artifact_xyz")
        result = router.run(parsed)
        self.assertEqual(len(result.trace), 3)
        self.assertTrue(result.forced_exit)
        self.assertEqual(result.trace[-1].strategy.lane, SearchLane.DEEP)


class BaselineDownstreamAndStorageTests(unittest.TestCase):
    def test_personalization_enforces_relevance_floor_and_tie_band(self) -> None:
        doc_high = FileDocument(
            id="doc_high",
            path=Path("/tmp/high.md"),
            title="high",
            extension="md",
            content="pricing",
        )
        doc_tie = FileDocument(
            id="doc_tie",
            path=Path("/tmp/tie.md"),
            title="tie",
            extension="md",
            content="pricing",
        )
        doc_outside_band = FileDocument(
            id="doc_outside_band",
            path=Path("/tmp/outside.md"),
            title="outside",
            extension="md",
            content="pricing",
        )
        doc_low = FileDocument(
            id="doc_low",
            path=Path("/tmp/low.md"),
            title="low",
            extension="md",
            content="unrelated",
        )
        candidates = (
            SearchCandidate(document=doc_high, score=0.80, tool=RetrievalTool.KEYWORD),
            SearchCandidate(document=doc_tie, score=0.76, tool=RetrievalTool.KEYWORD),
            SearchCandidate(document=doc_outside_band, score=0.50, tool=RetrievalTool.KEYWORD),
            SearchCandidate(document=doc_low, score=0.15, tool=RetrievalTool.KEYWORD),
        )
        profile = UserProfile(
            file_opens={"doc_tie": 10, "doc_outside_band": 10, "doc_low": 10},
            preferred_extensions={"md": 5},
            topic_interests={"pricing": 5},
        )
        parsed = QueryUnderstandingModule().parse("pricing document")
        personalizer = PersonalizationModule(relevance_floor=0.25, tie_band=0.15)
        ranked = personalizer.personalize(candidates, parsed, profile)

        # doc_tie is within tie_band (0.80 - 0.76 <= 0.15) and >= relevance_floor (0.25), so its open boost nudges it to #1.
        self.assertEqual(ranked[0].document.id, "doc_tie")
        self.assertEqual(ranked[1].document.id, "doc_high")
        # doc_outside_band is outside tie_band (0.80 - 0.50 > 0.15), so its score is unchanged.
        self.assertEqual(ranked[2].document.id, "doc_outside_band")
        self.assertEqual(ranked[2].score, 0.50)
        # doc_low is below relevance_floor (0.15 < 0.25), so its score is unchanged.
        self.assertEqual(ranked[-1].document.id, "doc_low")
        self.assertEqual(ranked[-1].score, 0.15)

    def test_reranking_reorders_by_keyword_overlap_and_updates_reason(self) -> None:
        parsed = QueryUnderstandingModule().parse("pricing forecast")
        doc_no_overlap = FileDocument(
            id="doc_no_overlap",
            path=Path("/tmp/notes.md"),
            title="notes",
            extension="md",
            content="general sync",
        )
        doc_full_overlap = FileDocument(
            id="doc_full_overlap",
            path=Path("/tmp/pricing.md"),
            title="pricing",
            extension="md",
            content="forecast model",
        )
        cand_no_overlap = SearchCandidate(
            document=doc_no_overlap,
            score=0.75,
            tool=RetrievalTool.SEMANTIC,
            reason="hybrid: semantic",
        )
        cand_full_overlap = SearchCandidate(
            document=doc_full_overlap,
            score=0.70,
            tool=RetrievalTool.KEYWORD,
            reason="hybrid: keyword",
        )

        reranked = RerankingModule().rerank((cand_no_overlap, cand_full_overlap), parsed)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].document.id, "doc_full_overlap")
        self.assertEqual(reranked[0].score, 0.745)
        self.assertEqual(reranked[0].reason, "hybrid: keyword; reranked")
        self.assertEqual(reranked[1].document.id, "doc_no_overlap")
        self.assertEqual(reranked[1].score, 0.6375)
        self.assertEqual(reranked[1].reason, "hybrid: semantic; reranked")

    def test_explanation_includes_matched_terms_date_opens_and_forced_exit_notice(self) -> None:
        fixed_now = datetime(2026, 10, 19, 12, 0, 0)
        parsed = QueryUnderstandingModule(now=fixed_now).parse("the pricing deck I was working on last week")
        doc = FileDocument(
            id="doc_pricing",
            path=Path("/tmp/Pricing_v3.pptx"),
            title="Pricing_v3",
            extension="pptx",
            content="pricing forecast",
            modified_at=fixed_now - timedelta(days=3),
        )
        candidate = SearchCandidate(
            document=doc,
            score=0.82,
            tool=RetrievalTool.KEYWORD,
            matched_terms=("pricing", "forecast"),
            reason="keyword content match",
        )
        standard_step = RouterTraceStep(
            iteration=0,
            complexity_score=0.5,
            strategy=SearchStrategy(lane=SearchLane.STANDARD, tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC)),
            candidates_seen=1,
            reflection=Reflection(
                confident=True,
                coverage=1,
                top_score=0.82,
                relevance_to_intent=0.75,
                reason="standard lane produced enough coverage and relevance",
            ),
        )
        normal_router_result = RouterResult(
            parsed_query=parsed,
            candidates=(candidate,),
            trace=(standard_step,),
            forced_exit=False,
        )
        profile = UserProfile(file_opens={"doc_pricing": 1})

        recommendations = ExplanationModule().explain((candidate,), normal_router_result, profile)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(
            recommendations[0].explanation,
            "Recommended because matches 'pricing', 'forecast'; was modified during last week; you opened it 1 time.",
        )

        # Forced-exit Deep Lane case with no matched terms, date window, or opens
        vague_parsed = QueryUnderstandingModule(now=fixed_now).parse("mystery request")
        unmatched_candidate = SearchCandidate(
            document=doc,
            score=0.20,
            tool=RetrievalTool.HYBRID,
            matched_terms=(),
            reason="hybrid: semantic",
        )
        deep_step = RouterTraceStep(
            iteration=2,
            complexity_score=0.85,
            strategy=SearchStrategy(
                lane=SearchLane.DEEP,
                tools=(RetrievalTool.HYBRID,),
                requires_reranking=True,
                uses_llm_reasoning=True,
            ),
            candidates_seen=1,
            reflection=Reflection(
                confident=True,
                coverage=1,
                top_score=0.20,
                relevance_to_intent=0.14,
                reason="iteration guard forced Deep Lane exit after two escalations",
            ),
        )
        forced_router_result = RouterResult(
            parsed_query=vague_parsed,
            candidates=(unmatched_candidate,),
            trace=(deep_step,),
            forced_exit=True,
        )
        forced_recommendations = ExplanationModule().explain((unmatched_candidate,), forced_router_result, UserProfile())
        self.assertEqual(len(forced_recommendations), 1)
        self.assertEqual(
            forced_recommendations[0].explanation,
            "Recommended because this is a best-effort match because the router reached its escalation guard.",
        )

    def test_usage_log_round_trip_and_engine_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "docs"
            root.mkdir()
            file_a = root / "Pricing_v3.txt"
            file_a.write_text("pricing revenue cost forecast", encoding="utf-8")
            db_path = Path(tmpdir) / "usage.db"

            doc_a = DocumentLoader().load_file(file_a)
            self.assertIsNotNone(doc_a)
            assert doc_a is not None

            usage_log = UsageLog(db_path)
            usage_log.record_open(doc_a, topics=("pricing",))
            profile = usage_log.load_profile()
            self.assertEqual(profile.file_opens.get(doc_a.id), 1)
            self.assertEqual(profile.preferred_extensions.get("txt"), 1)
            self.assertEqual(profile.topic_interests.get("pricing"), 1)
            self.assertEqual(ContextMemoryModule(profile).load_context(), profile)

            engine = ScoutXEngine.from_directory(root, usage_db=db_path)
            response = engine.search("pricing forecast")
            self.assertGreaterEqual(len(response.recommendations), 1)
            self.assertEqual(response.recommendations[0].document.id, doc_a.id)
            self.assertIn("you opened it 1 time", response.recommendations[0].explanation)


if __name__ == "__main__":
    unittest.main()
