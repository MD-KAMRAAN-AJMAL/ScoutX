"""Slice 2 unit and integration tests for ComplexityRoutingAgent."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from scoutx.models import (
    DateWindow,
    FileDocument,
    ParsedQuery,
    ReasoningStatus,
    RetrievalTool,
    RouterThresholds,
    SearchCandidate,
    SearchLane,
    SearchStrategy,
    ToolExecutionOutcome,
    TraceCandidateEvidence,
)
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent


class EmptyToolbox:
    """Deterministic retrieval toolbox that always returns zero candidates."""

    def __init__(self) -> None:
        self.calls: list[tuple[RetrievalTool, int, bool]] = []

    def search(
        self,
        tool: RetrievalTool,
        parsed_query: ParsedQuery,
        limit: int = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        self.calls.append((tool, limit, widened))
        return ()


class PartiallyFailingToolbox:
    """Deterministic toolbox where one tool raises an exception and another returns empty or results."""

    def __init__(
        self,
        failing_tools: set[RetrievalTool],
        results_by_tool: dict[RetrievalTool, tuple[SearchCandidate, ...]] | None = None,
    ) -> None:
        self.failing_tools = failing_tools
        self.results_by_tool = results_by_tool or {}

    def search(
        self,
        tool: RetrievalTool,
        parsed_query: ParsedQuery,
        limit: int = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        if tool in self.failing_tools:
            raise RuntimeError(f"simulated {tool.value} index corruption")
        return self.results_by_tool.get(tool, ())


class MultiPassDistinctEvidenceToolbox:
    """Returns distinct candidates on each pass (FAST -> STANDARD -> DEEP) to test trace evidence."""

    def __init__(self, root: Path, now: datetime) -> None:
        self.in_deep_pass = False
        self.doc_pass0 = FileDocument(
            id="doc-pass-0",
            path=root / "drafts/unrelated_0.txt",
            title="unrelated_0",
            extension="txt",
            content="unrelated draft 0",
            modified_at=now - timedelta(days=20),
        )
        self.doc_pass1 = FileDocument(
            id="doc-pass-1",
            path=root / "drafts/unrelated_1.txt",
            title="unrelated_1",
            extension="txt",
            content="unrelated draft 1",
            modified_at=now - timedelta(days=20),
        )
        self.doc_pass2 = FileDocument(
            id="doc-pass-2",
            path=root / "drafts/unrelated_2.txt",
            title="unrelated_2",
            extension="txt",
            content="unrelated draft 2",
            modified_at=now - timedelta(days=20),
        )

    def search(
        self,
        tool: RetrievalTool,
        parsed_query: ParsedQuery,
        limit: int = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        if tool == RetrievalTool.HYBRID:
            self.in_deep_pass = True
            return (
                SearchCandidate(
                    document=self.doc_pass2,
                    score=0.38,
                    tool=RetrievalTool.HYBRID,
                    matched_terms=("hyb2",),
                    reason="pass 2 hybrid hit",
                ),
            )
        if self.in_deep_pass:
            return ()
        if tool == RetrievalTool.FILENAME and not widened:
            return (
                SearchCandidate(
                    document=self.doc_pass0,
                    score=0.30,
                    tool=RetrievalTool.FILENAME,
                    matched_terms=("hint0",),
                    reason="pass 0 filename hit",
                ),
            )
        if tool in (RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC):
            return (
                SearchCandidate(
                    document=self.doc_pass1,
                    score=0.35 if tool == RetrievalTool.KEYWORD else 0.40,
                    tool=tool,
                    matched_terms=("kw1",) if tool == RetrievalTool.KEYWORD else ("sem1",),
                    reason=f"pass 1 {tool.value} hit",
                ),
            )
        return ()


class DeterministicFakeReasoner:
    """Single-turn deterministic test double for CandidateReasoner."""

    def __init__(self, *, should_fail: bool = False, boost: float = 0.08) -> None:
        self.should_fail = should_fail
        self.boost = boost
        self.calls: list[tuple[str, int, bool, bool]] = []

    def reason_candidates(
        self,
        parsed_query: ParsedQuery,
        candidates: tuple[SearchCandidate, ...],
        *,
        widened: bool = False,
        forced_guard: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        self.calls.append((parsed_query.raw, len(candidates), widened, forced_guard))
        if self.should_fail:
            raise RuntimeError("simulated reasoning failure")
        updated: list[SearchCandidate] = []
        for idx, candidate in enumerate(candidates):
            delta = self.boost if idx == 0 else 0.0
            updated.append(
                SearchCandidate(
                    document=candidate.document,
                    score=min(1.0, candidate.score + delta),
                    tool=candidate.tool,
                    matched_terms=candidate.matched_terms,
                    reason=f"{candidate.reason}; reasoned single-turn rerank",
                )
            )
        return tuple(updated)


class ComplexityRoutingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.now = datetime(2026, 3, 18, 12, 0, 0)

        self._write_fixture(
            "decks/Pricing_v3.pptx",
            "Enterprise pricing deck with Q3 discount tiers and margin guidance.",
            days_ago=5,
        )
        self._write_fixture(
            "finance/Q3_numbers_summary.xlsx",
            "Quarterly revenue spreadsheet with bookings, churn, and forecast metrics.",
            days_ago=60,
        )
        self._write_fixture(
            "notes/meeting_notes.md",
            "General team sync action items and administrative updates.",
            days_ago=2,
        )

        self.parser = QueryUnderstandingModule(now=self.now)
        self.corpus = DocumentCorpus.from_directory(self.root)
        self.toolbox = RetrievalToolbox(self.corpus)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _write_fixture(self, relative_path: str, content: str, *, days_ago: int) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        timestamp = (self.now - timedelta(days=days_ago)).timestamp()
        os.utime(path, (timestamp, timestamp))
        return path

    def test_1_clear_query_selects_standard_and_exits_after_one_pass(self) -> None:
        parsed = self.parser.parse("the pricing deck I was working on last week")
        router = ComplexityRoutingAgent(self.toolbox)

        result = router.run(parsed)

        self.assertEqual(len(result.trace), 1)
        step = result.trace[0]
        self.assertEqual(step.iteration, 0)
        self.assertGreaterEqual(step.complexity_score, router.thresholds.fast_max_complexity)
        self.assertLess(step.complexity_score, router.thresholds.standard_max_complexity)
        self.assertEqual(step.strategy.lane, SearchLane.STANDARD)
        self.assertEqual(
            step.strategy.tools,
            (RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )
        self.assertFalse(step.strategy.widened)
        self.assertFalse(step.strategy.uses_llm_reasoning)
        self.assertEqual(step.reasoning_status, ReasoningStatus.NOT_REQUESTED)
        self.assertTrue(step.reflection.confident)
        self.assertFalse(step.forced_exit)
        self.assertFalse(result.forced_exit)
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].document.path.name, "Pricing_v3.pptx")
        self.assertGreaterEqual(len(step.candidate_evidence), 1)
        self.assertEqual(step.candidate_evidence[0].document_id, result.candidates[0].document.id)

    def test_2_vague_query_enters_deep_widens_on_escalation_and_triggers_guard(self) -> None:
        parsed = self.parser.parse("that thing about numbers from a while ago")
        router = ComplexityRoutingAgent(self.toolbox, max_escalations=2, search_limit=5)

        result = router.run(parsed)

        self.assertEqual(len(result.trace), 3)
        self.assertTrue(result.forced_exit)

        # Pass 0: initial DEEP pass (unwidened)
        step0 = result.trace[0]
        self.assertEqual(step0.iteration, 0)
        self.assertGreaterEqual(step0.complexity_score, router.thresholds.standard_max_complexity)
        self.assertEqual(step0.strategy.lane, SearchLane.DEEP)
        self.assertEqual(step0.strategy.tools, (RetrievalTool.HYBRID,))
        self.assertFalse(step0.strategy.widened)
        self.assertEqual(step0.strategy.candidate_limit, 5)
        self.assertFalse(step0.reflection.confident)
        self.assertFalse(step0.forced_exit)

        # Pass 1: first escalation adapts strategy and widens DEEP retrieval
        step1 = result.trace[1]
        self.assertEqual(step1.iteration, 1)
        self.assertEqual(step1.strategy.lane, SearchLane.DEEP)
        self.assertTrue(step1.strategy.widened)
        self.assertEqual(step1.strategy.candidate_limit, 10)
        self.assertGreater(len(step1.strategy.tools), 1)
        self.assertIn(RetrievalTool.HYBRID, step1.strategy.tools)
        self.assertFalse(step1.reflection.confident)
        self.assertFalse(step1.forced_exit)

        # Pass 2: second escalation reaches Iteration Guard
        step2 = result.trace[2]
        self.assertEqual(step2.iteration, 2)
        self.assertEqual(step2.strategy.lane, SearchLane.DEEP)
        self.assertTrue(step2.strategy.widened)
        self.assertTrue(step2.strategy.uses_llm_reasoning)
        self.assertTrue(step2.forced_exit)
        self.assertFalse(step2.reflection.confident)
        self.assertIn("iteration guard forced Deep Lane exit", step2.reflection.reason)

    def test_3_weak_initial_fast_search_escalates_based_on_reflection(self) -> None:
        # Document content mentions "Margin_Playbook.pdf" inside body text, not in filename.
        # Quoted hint routes to FAST initially; FILENAME/METADATA miss it, so reflection
        # sees 0 candidates and escalates to STANDARD (KEYWORD + SEMANTIC), which finds it.
        self._write_fixture(
            "notes/archive_reference.txt",
            "See Margin_Playbook.pdf for the complete margin playbook and pricing guidance.",
            days_ago=3,
        )
        corpus = DocumentCorpus.from_directory(self.root)
        toolbox = RetrievalToolbox(corpus)
        router = ComplexityRoutingAgent(toolbox)

        parsed = self.parser.parse('"Margin_Playbook.pdf"')
        result = router.run(parsed)

        self.assertGreaterEqual(len(result.trace), 2)
        first_step = result.trace[0]
        self.assertEqual(first_step.strategy.lane, SearchLane.FAST)
        self.assertEqual(
            first_step.strategy.tools,
            (RetrievalTool.FILENAME, RetrievalTool.METADATA),
        )
        self.assertEqual(first_step.candidates_seen, 0)
        self.assertFalse(first_step.reflection.confident)

        second_step = result.trace[1]
        self.assertEqual(second_step.strategy.lane, SearchLane.STANDARD)
        self.assertIn(RetrievalTool.KEYWORD, second_step.strategy.tools)
        self.assertIn(RetrievalTool.SEMANTIC, second_step.strategy.tools)
        self.assertTrue(second_step.strategy.widened)
        self.assertGreaterEqual(second_step.candidates_seen, 1)

    def test_4_always_empty_toolbox_terminates_after_three_passes_without_fabricating_confidence(
        self,
    ) -> None:
        empty_toolbox = EmptyToolbox()
        router = ComplexityRoutingAgent(empty_toolbox, max_escalations=2, search_limit=6)  # type: ignore[arg-type]
        parsed = self.parser.parse('"Pricing_v3.pptx"')

        result = router.run(parsed)

        self.assertEqual(len(result.trace), 3)
        self.assertEqual([step.iteration for step in result.trace], [0, 1, 2])
        self.assertEqual(
            [step.strategy.lane for step in result.trace],
            [SearchLane.FAST, SearchLane.STANDARD, SearchLane.DEEP],
        )
        self.assertTrue(result.forced_exit)
        self.assertEqual(result.candidates, ())
        for idx, step in enumerate(result.trace):
            self.assertFalse(step.reflection.confident)
            self.assertEqual(step.reflection.coverage, 0)
            self.assertEqual(step.forced_exit, idx == 2)

    def test_5_final_guard_pass_uses_deep_enables_llm_reasoning_and_preserves_false_confidence(
        self,
    ) -> None:
        empty_toolbox = EmptyToolbox()
        fake_reasoner = DeterministicFakeReasoner()
        router = ComplexityRoutingAgent(
            empty_toolbox,  # type: ignore[arg-type]
            max_escalations=2,
            reasoner=fake_reasoner,
        )
        parsed = self.parser.parse("the pricing deck I was working on last week")

        result = router.run(parsed)

        self.assertEqual(len(result.trace), 3)
        final_step = result.trace[-1]
        self.assertEqual(final_step.strategy.lane, SearchLane.DEEP)
        self.assertTrue(final_step.strategy.requires_reranking)
        self.assertTrue(final_step.strategy.uses_llm_reasoning)
        self.assertTrue(final_step.strategy.widened)
        self.assertEqual(final_step.reasoning_status, ReasoningStatus.EXECUTED)
        self.assertTrue(final_step.forced_exit)
        self.assertTrue(result.forced_exit)
        self.assertFalse(final_step.reflection.confident)

    def test_6_observe_merges_and_deduplicates_candidates_with_provenance(self) -> None:
        router = ComplexityRoutingAgent(self.toolbox)
        parsed = self.parser.parse("pricing deck")
        doc = self.corpus.documents[0]

        outcomes = {
            RetrievalTool.KEYWORD: ToolExecutionOutcome(
                tool=RetrievalTool.KEYWORD,
                candidates=(
                    SearchCandidate(
                        document=doc,
                        score=0.60,
                        tool=RetrievalTool.KEYWORD,
                        matched_terms=("pricing",),
                        reason="keyword matched title/body terms (pricing)",
                    ),
                ),
            ),
            RetrievalTool.SEMANTIC: ToolExecutionOutcome(
                tool=RetrievalTool.SEMANTIC,
                candidates=(
                    SearchCandidate(
                        document=doc,
                        score=0.70,
                        tool=RetrievalTool.SEMANTIC,
                        matched_terms=("deck", "presentation"),
                        reason="semantic matched expanded intent terms (deck, presentation)",
                    ),
                ),
            ),
        }
        strategy = SearchStrategy(
            lane=SearchLane.STANDARD,
            tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )

        merged = router.observe(outcomes, strategy, parsed)

        self.assertEqual(len(merged), 1)
        candidate = merged[0]
        self.assertEqual(candidate.document.id, doc.id)
        self.assertAlmostEqual(candidate.score, 0.75, places=4)
        self.assertEqual(candidate.matched_terms, ("deck", "presentation", "pricing"))
        self.assertIn("router merged evidence from keyword, semantic", candidate.reason)
        self.assertIn("keyword matched title/body terms", candidate.reason)
        self.assertIn("semantic matched expanded intent terms", candidate.reason)

    def test_7_reflection_rejects_unrelated_candidate_even_with_high_raw_score(self) -> None:
        router = ComplexityRoutingAgent(self.toolbox)
        parsed = self.parser.parse("quarterly revenue spreadsheet")
        unrelated_doc = FileDocument(
            id="unrelated-1",
            path=self.root / "notes/holiday_party.txt",
            title="holiday_party",
            extension="txt",
            content="Catering menu and music playlist for the winter celebration.",
            modified_at=self.now - timedelta(days=1),
        )
        bogus_candidate = SearchCandidate(
            document=unrelated_doc,
            score=0.96,
            tool=RetrievalTool.KEYWORD,
            matched_terms=(),
            reason="artificially high score on unrelated file",
        )
        strategy = SearchStrategy(
            lane=SearchLane.STANDARD,
            tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )

        reflection = router.reflect(parsed, strategy, (bogus_candidate,))

        self.assertFalse(reflection.confident)
        self.assertLess(reflection.relevance_to_intent, router.thresholds.standard_min_relevance)
        self.assertIn("did not satisfy parsed intent", reflection.reason)

    def test_8_keyword_filetype_and_date_constraints_affect_reflection(self) -> None:
        router = ComplexityRoutingAgent(self.toolbox)
        window = DateWindow(
            start=self.now - timedelta(days=7),
            end=self.now,
            label="last week",
        )
        parsed = ParsedQuery(
            raw="pricing deck last week",
            intent="find",
            keywords=("pricing", "deck"),
            file_types=("pptx",),
            date_window=window,
            ambiguity_score=0.10,
            parser_confidence=0.90,
        )
        strategy = SearchStrategy(
            lane=SearchLane.STANDARD,
            tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )

        matching_doc = FileDocument(
            id="match-doc",
            path=self.root / "decks/Pricing_v3.pptx",
            title="Pricing_v3",
            extension="pptx",
            content="Enterprise pricing deck for Q3.",
            modified_at=self.now - timedelta(days=3),
        )
        wrong_type_and_date_doc = FileDocument(
            id="mismatch-doc",
            path=self.root / "decks/Pricing_v1.txt",
            title="Pricing_v1",
            extension="txt",
            content="Enterprise pricing deck notes.",
            modified_at=self.now - timedelta(days=45),
        )

        good_reflection = router.reflect(
            parsed,
            strategy,
            (
                SearchCandidate(
                    document=matching_doc,
                    score=0.80,
                    tool=RetrievalTool.KEYWORD,
                    matched_terms=("pricing", "deck"),
                ),
            ),
        )
        weak_reflection = router.reflect(
            parsed,
            strategy,
            (
                SearchCandidate(
                    document=wrong_type_and_date_doc,
                    score=0.80,
                    tool=RetrievalTool.KEYWORD,
                    matched_terms=("pricing", "deck"),
                ),
            ),
        )

        self.assertTrue(good_reflection.confident)
        self.assertFalse(weak_reflection.confident)
        self.assertGreater(
            good_reflection.relevance_to_intent,
            weak_reflection.relevance_to_intent,
        )

    def test_9_failed_retrieval_tool_is_distinguished_from_empty_result(self) -> None:
        failing_toolbox = PartiallyFailingToolbox(
            failing_tools={RetrievalTool.KEYWORD},
            results_by_tool={RetrievalTool.SEMANTIC: ()},
        )
        router = ComplexityRoutingAgent(failing_toolbox, max_escalations=0)  # type: ignore[arg-type]
        parsed = self.parser.parse("the pricing deck I was working on last week")
        strategy = SearchStrategy(
            lane=SearchLane.STANDARD,
            tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )

        outcomes = router.act(strategy, parsed)

        self.assertFalse(outcomes[RetrievalTool.KEYWORD].succeeded)
        self.assertIn("RuntimeError", outcomes[RetrievalTool.KEYWORD].error or "")
        self.assertTrue(outcomes[RetrievalTool.SEMANTIC].succeeded)
        self.assertIsNone(outcomes[RetrievalTool.SEMANTIC].error)
        self.assertEqual(outcomes[RetrievalTool.SEMANTIC].candidates, ())

        result = router.run(parsed)
        step = result.trace[0]
        self.assertIn(RetrievalTool.KEYWORD, step.tool_errors)
        self.assertNotIn(RetrievalTool.SEMANTIC, step.tool_errors)
        self.assertIn("tool failure (keyword:", step.reflection.reason)

    def test_10_deterministic_fake_reasoner_verifies_bounded_llm_interface(self) -> None:
        fake_reasoner = DeterministicFakeReasoner(boost=0.05)
        router = ComplexityRoutingAgent(
            self.toolbox,
            max_escalations=1,
            reasoner=fake_reasoner,
        )
        parsed = self.parser.parse("that thing about numbers from a while ago")

        result = router.run(parsed)

        # Every step in this run is in DEEP lane, and fake_reasoner is called exactly once per pass
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(len(fake_reasoner.calls), 2)
        self.assertEqual(fake_reasoner.calls[0][2:], (False, False))
        self.assertEqual(fake_reasoner.calls[1][2:], (True, True))
        for step in result.trace:
            self.assertEqual(step.reasoning_status, ReasoningStatus.EXECUTED)

        # Without a configured reasoner, Deep Lane reports UNAVAILABLE rather than claiming execution
        router_without_llm = ComplexityRoutingAgent(self.toolbox, max_escalations=0)
        no_llm_result = router_without_llm.run(parsed)
        self.assertEqual(
            no_llm_result.trace[0].reasoning_status,
            ReasoningStatus.UNAVAILABLE,
        )

    def test_11_invalid_thresholds_and_router_configuration_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            RouterThresholds(fast_max_complexity=0.80, standard_max_complexity=0.50)

        with self.assertRaises(ValueError):
            RouterThresholds(min_coverage=0)

        with self.assertRaises(ValueError):
            RouterThresholds(fast_min_top_score=0.0)

        with self.assertRaises(ValueError):
            RouterThresholds(deep_min_relevance=1.2)

        with self.assertRaises(ValueError):
            ComplexityRoutingAgent(self.toolbox, max_escalations=-1)

        with self.assertRaises(ValueError):
            ComplexityRoutingAgent(self.toolbox, search_limit=0)

    def test_12_complexity_score_uses_history_signal_and_avoids_double_counting_uncertainty(
        self,
    ) -> None:
        base_router = ComplexityRoutingAgent(self.toolbox)
        history_router = ComplexityRoutingAgent(
            self.toolbox,
            history_difficulty_fn=lambda q: 1.0,
        )

        # Correlated ambiguity (0.60) and parser uncertainty (1 - 0.40 = 0.60) should not double-count
        q_single = ParsedQuery(
            raw="budget notes",
            intent="find",
            keywords=("budget",),
            ambiguity_score=0.60,
            parser_confidence=1.0,
        )
        q_correlated = ParsedQuery(
            raw="budget notes",
            intent="find",
            keywords=("budget",),
            ambiguity_score=0.60,
            parser_confidence=0.40,
        )
        self.assertAlmostEqual(
            base_router.score_complexity(q_single),
            base_router.score_complexity(q_correlated),
            places=6,
        )
        self.assertGreater(
            history_router.score_complexity(q_single),
            base_router.score_complexity(q_single),
        )

    def test_13_approved_default_thresholds_and_reflection_boundary(self) -> None:
        thresholds = RouterThresholds()
        self.assertEqual(thresholds.fast_min_top_score, 0.65)
        self.assertEqual(thresholds.standard_min_top_score, 0.50)
        self.assertEqual(thresholds.deep_min_top_score, 0.45)
        self.assertEqual(thresholds.fast_min_relevance, 0.55)
        self.assertEqual(thresholds.standard_min_relevance, 0.50)
        self.assertEqual(thresholds.deep_min_relevance, 0.50)

        router = ComplexityRoutingAgent(self.toolbox, thresholds=thresholds)
        parsed = ParsedQuery(
            raw="pricing deck",
            intent="find",
            keywords=("pricing", "deck"),
            file_types=("pptx",),
            ambiguity_score=0.10,
            parser_confidence=0.90,
        )
        doc = FileDocument(
            id="pricing-doc",
            path=self.root / "decks/Pricing_v3.pptx",
            title="Pricing_v3",
            extension="pptx",
            content="Enterprise pricing deck.",
            modified_at=self.now - timedelta(days=1),
        )

        standard_strategy = SearchStrategy(
            lane=SearchLane.STANDARD,
            tools=(RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC),
        )
        below_standard = router.reflect(
            parsed,
            standard_strategy,
            (SearchCandidate(document=doc, score=0.48, tool=RetrievalTool.KEYWORD),),
        )
        at_standard = router.reflect(
            parsed,
            standard_strategy,
            (SearchCandidate(document=doc, score=0.50, tool=RetrievalTool.KEYWORD),),
        )
        self.assertFalse(below_standard.confident)
        self.assertIn("top score 0.48 below standard threshold", below_standard.reason)
        self.assertTrue(at_standard.confident)

        deep_strategy = SearchStrategy(
            lane=SearchLane.DEEP,
            tools=(RetrievalTool.HYBRID,),
            requires_reranking=True,
            uses_llm_reasoning=True,
        )
        below_deep = router.reflect(
            parsed,
            deep_strategy,
            (SearchCandidate(document=doc, score=0.42, tool=RetrievalTool.HYBRID),),
        )
        at_deep = router.reflect(
            parsed,
            deep_strategy,
            (SearchCandidate(document=doc, score=0.45, tool=RetrievalTool.HYBRID),),
        )
        self.assertFalse(below_deep.confident)
        self.assertIn("top score 0.42 below deep threshold", below_deep.reason)
        self.assertTrue(at_deep.confident)

    def test_14_per_iteration_trace_evidence_preserved_across_passes(self) -> None:
        multi_toolbox = MultiPassDistinctEvidenceToolbox(self.root, self.now)
        router = ComplexityRoutingAgent(multi_toolbox, max_escalations=2)  # type: ignore[arg-type]
        parsed = self.parser.parse('"Pricing_v3.pptx"')

        result = router.run(parsed)

        self.assertEqual(len(result.trace), 3)

        # Pass 0 evidence (FAST) retained even though Pass 1 and Pass 2 return different candidates
        ev0 = result.trace[0].candidate_evidence
        self.assertEqual(len(ev0), 1)
        self.assertEqual(
            ev0[0],
            TraceCandidateEvidence(
                document_id="doc-pass-0",
                score=0.30,
                tools=(RetrievalTool.FILENAME,),
                matched_terms=("hint0",),
            ),
        )

        # Pass 1 evidence (STANDARD) records both contributing tools (KEYWORD and SEMANTIC)
        ev1 = result.trace[1].candidate_evidence
        self.assertEqual(len(ev1), 1)
        self.assertEqual(ev1[0].document_id, "doc-pass-1")
        self.assertAlmostEqual(ev1[0].score, 0.45, places=4)
        self.assertEqual(ev1[0].tools, (RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC))
        self.assertEqual(ev1[0].matched_terms, ("kw1", "sem1"))

        # Pass 2 evidence (DEEP) records final pass candidate
        ev2 = result.trace[2].candidate_evidence
        self.assertEqual(len(ev2), 1)
        self.assertEqual(ev2[0].document_id, "doc-pass-2")
        self.assertEqual(ev2[0].tools, (RetrievalTool.HYBRID,))
        self.assertEqual(ev2[0].matched_terms, ("hyb2",))
        self.assertEqual(result.candidates[0].document.id, "doc-pass-2")

    def test_15_reasoner_validation_rejects_malformed_invented_dropped_and_preserves_provenance(
        self,
    ) -> None:
        parsed = self.parser.parse("that thing about numbers from a while ago")
        doc_a = self.corpus.documents[0]
        doc_b = self.corpus.documents[1]
        cand_a = SearchCandidate(
            document=doc_a,
            score=0.72,
            tool=RetrievalTool.HYBRID,
            matched_terms=("pricing",),
            reason="hybrid: keyword, semantic",
        )
        cand_b = SearchCandidate(
            document=doc_b,
            score=0.58,
            tool=RetrievalTool.HYBRID,
            matched_terms=("numbers",),
            reason="hybrid: metadata, semantic",
        )
        outcomes = {
            RetrievalTool.HYBRID: ToolExecutionOutcome(
                tool=RetrievalTool.HYBRID,
                candidates=(cand_a, cand_b),
            )
        }
        deep_strategy = SearchStrategy(
            lane=SearchLane.DEEP,
            tools=(RetrievalTool.HYBRID,),
            requires_reranking=True,
            uses_llm_reasoning=True,
        )

        # Case A: Valid reordering/rescoring that attempts to tamper with FileDocument and tool/provenance
        tampered_doc_b = FileDocument(
            id=doc_b.id,
            path=Path("/tampered/fake_path.xlsx"),
            title="tampered_title",
            extension="xlsx",
            content="tampered content",
        )

        class ValidReorderingReasoner:
            def reason_candidates(self, parsed_query, candidates, *, widened=False, forced_guard=False):
                return (
                    SearchCandidate(
                        document=tampered_doc_b,
                        score=0.88,
                        tool=RetrievalTool.FILENAME,
                        matched_terms=("hallucinated_term",),
                        reason="llm boosted spreadsheet",
                    ),
                    SearchCandidate(
                        document=doc_a,
                        score=0.51,
                        tool=RetrievalTool.FILENAME,
                        matched_terms=(),
                        reason="",
                    ),
                )

        router_valid = ComplexityRoutingAgent(self.toolbox, reasoner=ValidReorderingReasoner())
        reordered, status = router_valid._observe_with_status(outcomes, deep_strategy, parsed)
        self.assertEqual(status, ReasoningStatus.EXECUTED)
        self.assertEqual([c.document.id for c in reordered], [doc_b.id, doc_a.id])
        self.assertAlmostEqual(reordered[0].score, 0.88, places=4)
        # Original document identity, tool, matched_terms, and provenance are preserved
        self.assertIs(reordered[0].document, doc_b)
        self.assertEqual(reordered[0].tool, RetrievalTool.HYBRID)
        self.assertEqual(reordered[0].matched_terms, ("numbers",))
        self.assertIn("router merged evidence from hybrid", reordered[0].reason)
        self.assertIn("llm boosted spreadsheet", reordered[0].reason)
        self.assertIn("router merged evidence from hybrid", reordered[1].reason)

        # Case B: Malformed / invalid outputs must be rejected and preserve original candidates + FAILED
        invented_doc = FileDocument(
            id="invented-id-999",
            path=self.root / "invented.txt",
            title="invented",
            extension="txt",
            content="invented",
        )
        invalid_outputs = [
            None,
            "not-a-candidate-list",
            # Mapping returned as overall reasoner result
            {doc_a.id: cand_a, doc_b.id: cand_b},
            (None, cand_b),
            ("not-a-candidate", cand_b),
            # Invented candidate ID
            (
                SearchCandidate(document=invented_doc, score=0.90, tool=RetrievalTool.HYBRID),
                cand_b,
            ),
            # Duplicate candidate ID
            (cand_a, cand_a),
            # Dropped candidate (only 1 of 2 returned)
            (cand_a,),
            # String score
            (
                SearchCandidate(document=doc_a, score="0.8", tool=RetrievalTool.HYBRID),  # type: ignore[arg-type]
                cand_b,
            ),
            # None score
            (
                SearchCandidate(document=doc_a, score=None, tool=RetrievalTool.HYBRID),  # type: ignore[arg-type]
                cand_b,
            ),
            # Boolean score
            (
                SearchCandidate(document=doc_a, score=True, tool=RetrievalTool.HYBRID),  # type: ignore[arg-type]
                cand_b,
            ),
            # Non-finite scores (NaN, inf)
            (
                SearchCandidate(document=doc_a, score=float("nan"), tool=RetrievalTool.HYBRID),
                cand_b,
            ),
            (
                SearchCandidate(document=doc_a, score=float("inf"), tool=RetrievalTool.HYBRID),
                cand_b,
            ),
            # Out-of-range scores (> 1.0 and < 0.0)
            (
                SearchCandidate(document=doc_a, score=1.25, tool=RetrievalTool.HYBRID),
                cand_b,
            ),
            (
                SearchCandidate(document=doc_a, score=-0.05, tool=RetrievalTool.HYBRID),
                cand_b,
            ),
        ]

        for bad_output in invalid_outputs:
            with self.subTest(bad_output=bad_output):
                class BadReasoner:
                    def reason_candidates(self, parsed_query, candidates, *, widened=False, forced_guard=False):
                        return bad_output

                router_bad = ComplexityRoutingAgent(self.toolbox, reasoner=BadReasoner())
                fallback, bad_status = router_bad._observe_with_status(outcomes, deep_strategy, parsed)
                self.assertEqual(bad_status, ReasoningStatus.FAILED)
                self.assertEqual([c.document.id for c in fallback], [doc_a.id, doc_b.id])
                self.assertAlmostEqual(fallback[0].score, 0.72, places=4)
                self.assertAlmostEqual(fallback[1].score, 0.58, places=4)

    def test_16_concurrent_queries_do_not_leak_reasoning_status_or_trace_state(self) -> None:
        barrier = threading.Barrier(2, timeout=5.0)
        sync_errors: list[BaseException] = []

        class SynchronizedReasoner:
            def reason_candidates(
                self,
                parsed_query: ParsedQuery,
                candidates: tuple[SearchCandidate, ...],
                *,
                widened: bool = False,
                forced_guard: bool = False,
            ) -> tuple[SearchCandidate, ...]:
                # Wait for the concurrent STANDARD query to run while this DEEP query is inside observe()
                try:
                    barrier.wait()
                except BaseException as exc:
                    sync_errors.append(exc)
                    raise
                return candidates

        class SynchronizedToolbox:
            def __init__(self, inner: RetrievalToolbox) -> None:
                self.inner = inner

            def search(
                self,
                tool: RetrievalTool,
                parsed_query: ParsedQuery,
                limit: int = 10,
                *,
                widened: bool = False,
            ) -> tuple[SearchCandidate, ...]:
                res = self.inner.search(tool, parsed_query, limit=limit, widened=widened)
                if tool == RetrievalTool.SEMANTIC and "pricing" in parsed_query.raw:
                    try:
                        barrier.wait()
                    except BaseException as exc:
                        sync_errors.append(exc)
                        raise
                return res

        shared_router = ComplexityRoutingAgent(
            SynchronizedToolbox(self.toolbox),  # type: ignore[arg-type]
            max_escalations=0,
            reasoner=SynchronizedReasoner(),
        )
        standard_query = self.parser.parse("the pricing deck I was working on last week")
        deep_query = self.parser.parse("that thing about numbers from a while ago")

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_deep = executor.submit(shared_router.run, deep_query)
            fut_standard = executor.submit(shared_router.run, standard_query)
            res_deep = fut_deep.result(timeout=5.0)
            res_standard = fut_standard.result(timeout=5.0)

        self.assertEqual(sync_errors, [], f"Barrier synchronization failed: {sync_errors!r}")
        self.assertFalse(barrier.broken)

        self.assertEqual(len(res_deep.trace), 1)
        self.assertEqual(res_deep.trace[0].strategy.lane, SearchLane.DEEP)
        self.assertEqual(res_deep.trace[0].reasoning_status, ReasoningStatus.EXECUTED)

        self.assertEqual(len(res_standard.trace), 1)
        self.assertEqual(res_standard.trace[0].strategy.lane, SearchLane.STANDARD)
        self.assertEqual(res_standard.trace[0].reasoning_status, ReasoningStatus.NOT_REQUESTED)
        self.assertEqual(res_standard.trace[0].tool_errors, {})


if __name__ == "__main__":
    unittest.main()
