"""Deterministic unit and local loopback HTTP integration tests for Slice 3 LLM contracts."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scoutx import cli
from scoutx.downstream import (
    DEGRADED_FALLBACK_PREFIX,
    FORCED_EXIT_SUBSTRING,
    ExplanationModule,
)
from scoutx.engine import ScoutXEngine
from scoutx.llm import (
    GeminiNativeLLMProvider,
    GroundedExplanationItem,
    LLMConfig,
    LLMError,
    LLMGenerationResult,
    LLMProvider,
    LLMResponseFormatError,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAICompatibleLLMProvider,
    create_llm_provider,
)
from scoutx.models import (
    DateWindow,
    FileDocument,
    ParsedQuery,
    ReasoningStatus,
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


def _make_doc(
    doc_id: str = "doc-1",
    name: str = "q3_budget.xlsx",
    title: str = "Q3 Marketing Budget",
    extension: str = "xlsx",
    content: str = "Quarterly marketing budget numbers and revenue forecast for Q3.",
    modified_at: datetime | None = datetime(2026, 5, 18, 12, 0, 0),
    metadata: dict[str, str] | None = None,
    root_dir: str = "/secret/host/directory",
) -> FileDocument:
    return FileDocument(
        id=doc_id,
        path=Path(root_dir) / name,
        title=title,
        extension=extension,
        content=content,
        modified_at=modified_at,
        metadata=metadata or {},
    )


def _make_router_result(
    candidates: tuple[SearchCandidate, ...],
    *,
    lane: SearchLane = SearchLane.STANDARD,
    forced_exit: bool = False,
    date_window: DateWindow | None = None,
) -> RouterResult:
    pq = ParsedQuery(
        raw="find q3 budget",
        intent="find",
        keywords=("budget",),
        date_window=date_window,
    )
    strategy = SearchStrategy(
        lane=lane,
        tools=(RetrievalTool.HYBRID,) if lane == SearchLane.DEEP else (RetrievalTool.KEYWORD,),
        requires_reranking=(lane == SearchLane.DEEP),
        uses_llm_reasoning=(lane == SearchLane.DEEP),
    )
    reflection = Reflection(
        confident=not forced_exit,
        coverage=len(candidates),
        top_score=candidates[0].score if candidates else 0.0,
        relevance_to_intent=0.35 if forced_exit else 0.80,
        reason="iteration guard forced Deep Lane exit" if forced_exit else "confident",
    )
    step = RouterTraceStep(
        iteration=2 if forced_exit else 0,
        complexity_score=0.75 if lane == SearchLane.DEEP else 0.45,
        strategy=strategy,
        candidates_seen=len(candidates),
        reflection=reflection,
        forced_exit=forced_exit,
    )
    return RouterResult(
        parsed_query=pq,
        candidates=candidates,
        trace=(step,),
        forced_exit=forced_exit,
    )


class _StubLLMProvider:
    def __init__(
        self,
        *,
        model_id: str = "stub-model-v1",
        config: LLMConfig | None = None,
        parse_fn: Any = None,
        reason_fn: Any = None,
        explain_fn: Any = None,
    ) -> None:
        self._model_id = model_id
        self._config = config or LLMConfig(provider="ollama", model=model_id)
        self._parse_fn = parse_fn
        self._reason_fn = reason_fn
        self._explain_fn = explain_fn
        self.parse_calls = 0
        self.reason_calls = 0
        self.explain_calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def config(self) -> LLMConfig:
        return self._config

    def parse_ambiguous_query(self, raw_query: str, rule_draft: ParsedQuery) -> ParsedQuery:
        self.parse_calls += 1
        if self._parse_fn is not None:
            return self._parse_fn(raw_query, rule_draft)
        return rule_draft

    def reason_candidates(
        self,
        parsed_query: ParsedQuery,
        candidates: tuple[SearchCandidate, ...],
        *,
        widened: bool = False,
        forced_guard: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        self.reason_calls += 1
        if self._reason_fn is not None:
            return self._reason_fn(parsed_query, candidates, widened=widened, forced_guard=forced_guard)
        return candidates

    def generate_explanations(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
    ) -> LLMGenerationResult:
        self.explain_calls += 1
        if self._explain_fn is not None:
            return self._explain_fn(candidates, router_result, profile)
        return LLMGenerationResult(
            explanations=tuple(
                GroundedExplanationItem(
                    document_id=c.document.id,
                    explanation=f"Matches {c.document.title} using {c.tool.value} retrieval.",
                    cited_evidence=("title", "retrieval_tool"),
                )
                for c in candidates
            ),
            model_id=self._model_id,
        )


class _LoopbackRecord:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.mode: str = "ok"
        self.delay_seconds: float = 0.0

    def record(self, path: str, headers: dict[str, str], raw_body: str) -> None:
        with self.lock:
            self.requests.append({"path": path, "headers": headers, "body": raw_body})


class Slice3LLMTests(unittest.TestCase):
    def test_01_llm_config_validation_and_provider_defaults(self) -> None:
        ollama_cfg = LLMConfig(provider="ollama", model="qwen2.5:3b")
        self.assertEqual(ollama_cfg.resolved_base_url, "http://127.0.0.1:11434/v1")
        self.assertIsNone(ollama_cfg.resolved_api_key_env)
        self.assertEqual(ollama_cfg.max_input_chars, 2400)

        openai_cfg = LLMConfig(provider="openai_compatible", model="gpt-4o-mini")
        self.assertEqual(openai_cfg.resolved_base_url, "https://api.openai.com/v1")
        self.assertEqual(openai_cfg.resolved_api_key_env, "OPENAI_API_KEY")

        gemini_cfg = LLMConfig(provider="gemini", model="gemini-2.5-flash")
        self.assertEqual(gemini_cfg.resolved_base_url, "https://generativelanguage.googleapis.com/v1beta")
        self.assertEqual(gemini_cfg.resolved_api_key_env, "GEMINI_API_KEY")

        self.assertIsInstance(create_llm_provider(ollama_cfg), OpenAICompatibleLLMProvider)
        self.assertIsInstance(create_llm_provider(openai_cfg), OpenAICompatibleLLMProvider)
        self.assertIsInstance(create_llm_provider(gemini_cfg), GeminiNativeLLMProvider)
        self.assertIsInstance(create_llm_provider(ollama_cfg), LLMProvider)

        invalid_configs = [
            {"provider": "unknown", "model": "m"},
            {"provider": "ollama", "model": ""},
            {"provider": "ollama", "model": "   "},
            {"provider": "ollama", "model": "m", "base_url": "ftp://localhost:11434"},
            {"provider": "ollama", "model": "m", "timeout_seconds": 0.0},
            {"provider": "ollama", "model": "m", "timeout_seconds": -2.0},
            {"provider": "ollama", "model": "m", "timeout_seconds": float("nan")},
            {"provider": "ollama", "model": "m", "timeout_seconds": float("inf")},
            {"provider": "ollama", "model": "m", "timeout_seconds": True},
            {"provider": "ollama", "model": "m", "max_input_chars": 0},
            {"provider": "ollama", "model": "m", "max_input_chars": 100},
            {"provider": "ollama", "model": "m", "max_input_chars": 100000},
            {"provider": "ollama", "model": "m", "max_input_chars": True},
            {"provider": "ollama", "model": "m", "temperature": -0.1},
            {"provider": "ollama", "model": "m", "temperature": 2.5},
            {"provider": "ollama", "model": "m", "temperature": False},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LLMConfig(**kwargs)  # type: ignore[arg-type]

    def test_02_api_key_env_resolution_and_secret_redaction_on_errors(self) -> None:
        env_var = "SCOUTX_TEST_SECRET_KEY"
        secret_value = "sk-super-secret-token-987654321"

        # 1. Missing environment variable fails fast with LLMUnavailableError
        os.environ.pop(env_var, None)
        openai_provider = OpenAICompatibleLLMProvider(
            LLMConfig(
                provider="openai_compatible",
                model="gpt-4o-mini",
                base_url="http://127.0.0.1:9/v1",
                api_key_env=env_var,
                timeout_seconds=1.0,
            )
        )
        with self.assertRaises(LLMUnavailableError) as ctx:
            openai_provider.parse_ambiguous_query("that thing", ParsedQuery(raw="that thing", intent="find"))
        self.assertIn(env_var, str(ctx.exception))

        # 2. When HTTP error echoes the secret or headers, the raised exception redacts the secret
        record = _LoopbackRecord()

        class _ErrorEchoHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                record.record(self.path, dict(self.headers), raw_body)
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_payload = f'{{"error": "Invalid key {secret_value} on path {self.path}"}}'
                self.wfile.write(err_payload.encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _ErrorEchoHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {env_var: secret_value}):
                oai = OpenAICompatibleLLMProvider(
                    LLMConfig(
                        provider="openai_compatible",
                        model="gpt-4o-mini",
                        base_url=f"http://127.0.0.1:{port}/v1",
                        api_key_env=env_var,
                    )
                )
                with self.assertRaises(LLMUnavailableError) as oai_err:
                    oai.parse_ambiguous_query("that thing", ParsedQuery(raw="that thing", intent="find"))
                self.assertNotIn(secret_value, str(oai_err.exception))
                self.assertIn("[REDACTED]", str(oai_err.exception))

                gem = GeminiNativeLLMProvider(
                    LLMConfig(
                        provider="gemini",
                        model="gemini-2.5-flash",
                        base_url=f"http://127.0.0.1:{port}/v1beta",
                        api_key_env=env_var,
                    )
                )
                with self.assertRaises(LLMUnavailableError) as gem_err:
                    gem.parse_ambiguous_query("that thing", ParsedQuery(raw="that thing", intent="find"))
                self.assertNotIn(secret_value, str(gem_err.exception))
                self.assertIn("[REDACTED]", str(gem_err.exception))

                # Confirm Gemini sent key in x-goog-api-key header, NEVER in URL query string
                self.assertEqual(len(record.requests), 2)
                gemini_req = record.requests[1]
                self.assertNotIn(secret_value, gemini_req["path"])
                self.assertNotIn("key=", gemini_req["path"])
                header_keys = {k.lower(): v for k, v in gemini_req["headers"].items()}
                self.assertEqual(header_keys.get("x-goog-api-key"), secret_value)
        finally:
            server.shutdown()
            server.server_close()

    def test_03_explanation_grounding_validates_candidate_ids_and_available_evidence_fields(self) -> None:
        doc = _make_doc(
            doc_id="doc-budget",
            title="Q3 Marketing Budget",
            extension="xlsx",
            modified_at=None,  # modified_date is NOT available on this candidate
        )
        candidate = SearchCandidate(
            document=doc,
            score=0.88,
            tool=RetrievalTool.KEYWORD,
            matched_terms=(),  # matched_terms is NOT available on this candidate
            reason="keyword match",
        )
        router_result = _make_router_result((candidate,))
        profile = UserProfile(file_opens={})  # open_count is NOT available on this candidate

        # Valid grounded explanation citing available fields ("title", "file_type") with lexical overlap
        valid_provider = _StubLLMProvider(
            explain_fn=lambda cands, rr, prof: LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because the Q3 Marketing Budget spreadsheet matches your request.",
                        cited_evidence=("title", "file_type"),
                    ),
                ),
                model_id="valid-llm-v1",
            )
        )
        explainer = ExplanationModule(llm_provider=valid_provider)
        recs = explainer.explain((candidate,), router_result, profile)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].explanation_source, "llm")
        self.assertEqual(recs[0].explanation_model, "valid-llm-v1")
        self.assertEqual(recs[0].cited_evidence, ("title", "file_type"))

        # Invalid explanation cases must all fail validation and raise LLMResponseFormatError
        invalid_results = [
            # 1. Wrong document_id
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="invented-doc",
                        explanation="Recommended because Q3 Marketing Budget spreadsheet matches.",
                        cited_evidence=("title",),
                    ),
                ),
                model_id="m",
            ),
            # 2. Count mismatch (empty)
            LLMGenerationResult(explanations=(), model_id="m"),
            # 3. Unknown evidence key ("full_path" is not in ALLOWED_EVIDENCE_FIELDS)
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because Q3 Marketing Budget matches full path.",
                        cited_evidence=("title", "full_path"),
                    ),
                ),
                model_id="m",
            ),
            # 4. Unavailable evidence key ("matched_terms" is empty on candidate)
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because Q3 Marketing Budget matched terms.",
                        cited_evidence=("title", "matched_terms"),
                    ),
                ),
                model_id="m",
            ),
            # 5. Unavailable evidence key ("modified_date" when modified_at is None)
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because Q3 Marketing Budget was modified recently.",
                        cited_evidence=("title", "modified_date"),
                    ),
                ),
                model_id="m",
            ),
            # 6. Unavailable evidence key ("open_count" when opens == 0)
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because Q3 Marketing Budget was opened 5 times.",
                        cited_evidence=("title", "open_count"),
                    ),
                ),
                model_id="m",
            ),
            # 7. Zero lexical overlap with cited evidence ("title")
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="This item appears relevant to your general workflow.",
                        cited_evidence=("title",),
                    ),
                ),
                model_id="m",
            ),
            # 8. Duplicate cited_evidence keys
            LLMGenerationResult(
                explanations=(
                    GroundedExplanationItem(
                        document_id="doc-budget",
                        explanation="Recommended because Q3 Marketing Budget matches title.",
                        cited_evidence=("title", "title"),
                    ),
                ),
                model_id="m",
            ),
        ]

        for idx, bad_res in enumerate(invalid_results):
            with self.subTest(case_index=idx):
                bad_provider = _StubLLMProvider(explain_fn=lambda c, r, p, res=bad_res: res)
                mod = ExplanationModule(llm_provider=bad_provider, allow_degraded_fallback=False)
                with self.assertRaises(LLMResponseFormatError):
                    mod.explain((candidate,), router_result, profile)

    def test_04_forced_exit_disclosure_enforced_deterministically_across_varied_contradictory_paraphrases(
        self,
    ) -> None:
        doc = _make_doc(doc_id="doc-deep", title="Architecture Notes", extension="md")
        candidate = SearchCandidate(
            document=doc,
            score=0.42,
            tool=RetrievalTool.HYBRID,
            matched_terms=("architecture",),
            reason="hybrid search",
        )
        forced_router_result = _make_router_result(
            (candidate,),
            lane=SearchLane.DEEP,
            forced_exit=True,
        )

        # Multiple differently worded contradictory confidence claims that a simple regex blacklist would miss:
        contradictory_paraphrases = [
            "Architecture Notes is a high-confidence match because router was confident.",
            "Architecture Notes is definitely the exact file you asked for with 100% certainty.",
            "We found an indisputable, rock-solid match in Architecture Notes for your query.",
            "Architecture Notes is a guaranteed, verified hit with strong assurance from the search agent.",
            "Without a doubt, Architecture Notes satisfies every requirement with complete certainty.",
        ]

        for idx, misleading_prose in enumerate(contradictory_paraphrases):
            with self.subTest(paraphrase_index=idx):
                misleading_provider = _StubLLMProvider(
                    explain_fn=lambda cands, rr, prof, text=misleading_prose: LLMGenerationResult(
                        explanations=(
                            GroundedExplanationItem(
                                document_id="doc-deep",
                                explanation=text,
                                cited_evidence=("title", "matched_terms"),
                            ),
                        ),
                        model_id="test-llm",
                    )
                )
                explainer = ExplanationModule(llm_provider=misleading_provider)
                recs = explainer.explain((candidate,), forced_router_result, UserProfile())
                self.assertEqual(len(recs), 1)
                # 1. The LLM call was still executed and validated
                self.assertEqual(misleading_provider.explain_calls, 1)
                # 2. Because forced_exit is True in Deep Lane, the returned explanation is built
                #    deterministically from validated cited_evidence and never contains the unverified prose
                self.assertEqual(recs[0].explanation_source, "deterministic")
                self.assertIsNone(recs[0].explanation_model)
                self.assertEqual(recs[0].cited_evidence, ("title", "matched_terms"))
                self.assertIn(FORCED_EXIT_SUBSTRING, recs[0].explanation)
                self.assertIn("titled 'Architecture Notes'", recs[0].explanation)
                self.assertIn("matches 'architecture'", recs[0].explanation)
                self.assertNotIn("100% certainty", recs[0].explanation)
                self.assertNotIn("indisputable", recs[0].explanation)
                self.assertNotIn("guaranteed", recs[0].explanation)
                self.assertNotIn("Without a doubt", recs[0].explanation)

    def test_05_explanation_module_mandatory_llm_vs_degraded_and_compatible_modes(self) -> None:
        doc = _make_doc(doc_id="doc-1", title="Roadmap Deck", extension="pptx")
        candidate = SearchCandidate(
            document=doc,
            score=0.80,
            tool=RetrievalTool.KEYWORD,
            matched_terms=("roadmap",),
            reason="keyword match",
        )
        router_result = _make_router_result(
            (candidate,),
            lane=SearchLane.DEEP,
            forced_exit=True,
        )
        profile = UserProfile(file_opens={"doc-1": 2})

        # 1. Default compatibility mode (no provider, require_llm=False)
        compat_explainer = ExplanationModule()
        compat_recs = compat_explainer.explain((candidate,), router_result, profile)
        self.assertEqual(compat_recs[0].explanation_source, "deterministic")
        self.assertIsNone(compat_recs[0].explanation_model)
        self.assertFalse(compat_recs[0].explanation.startswith(DEGRADED_FALLBACK_PREFIX))
        self.assertIn(FORCED_EXIT_SUBSTRING, compat_recs[0].explanation)

        # 2. LLM required, no provider configured, degraded fallback disabled -> raises LLMUnavailableError
        strict_missing = ExplanationModule(llm_provider=None, require_llm=True, allow_degraded_fallback=False)
        with self.assertRaises(LLMUnavailableError):
            strict_missing.explain((candidate,), router_result, profile)

        # 3. LLM required, no provider configured, degraded fallback enabled -> explicit degraded_fallback
        degraded_missing = ExplanationModule(llm_provider=None, require_llm=True, allow_degraded_fallback=True)
        deg_recs = degraded_missing.explain((candidate,), router_result, profile)
        self.assertEqual(deg_recs[0].explanation_source, "degraded_fallback")
        self.assertIsNone(deg_recs[0].explanation_model)
        self.assertTrue(deg_recs[0].explanation.startswith(DEGRADED_FALLBACK_PREFIX))
        self.assertIn(FORCED_EXIT_SUBSTRING, deg_recs[0].explanation)

        # 4. Provider configured, fails at runtime, degraded fallback disabled -> raises LLMError
        failing_provider = _StubLLMProvider(
            explain_fn=lambda c, r, p: (_ for _ in ()).throw(LLMTimeoutError("timed out"))
        )
        strict_failing = ExplanationModule(llm_provider=failing_provider, allow_degraded_fallback=False)
        with self.assertRaises(LLMTimeoutError):
            strict_failing.explain((candidate,), router_result, profile)

        # 5. Provider configured, fails at runtime, degraded fallback enabled -> explicit degraded_fallback
        degraded_failing = ExplanationModule(llm_provider=failing_provider, allow_degraded_fallback=True)
        deg_fail_recs = degraded_failing.explain((candidate,), router_result, profile)
        self.assertEqual(deg_fail_recs[0].explanation_source, "degraded_fallback")
        self.assertIsNone(deg_fail_recs[0].explanation_model)
        self.assertTrue(deg_fail_recs[0].explanation.startswith(DEGRADED_FALLBACK_PREFIX))
        self.assertIn(FORCED_EXIT_SUBSTRING, deg_fail_recs[0].explanation)

    def test_06_query_refinement_preserves_explicit_filename_type_date_and_keyword_constraints(self) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0)
        ambiguous_raw = "find that thing 'budget_v2.xlsx' from last week maybe"

        # Valid refinement: clarifies intent="recommend" and adds keyword "finance" while keeping all hard constraints
        valid_provider = _StubLLMProvider(
            parse_fn=lambda raw, draft: ParsedQuery(
                raw=draft.raw,
                intent="recommend",
                keywords=draft.keywords + ("finance", "forecast"),
                file_types=draft.file_types,
                date_window=draft.date_window,
                filename_hints=draft.filename_hints,
                ambiguity_score=0.25,
                parser_confidence=0.88,
            )
        )
        qu_valid = QueryUnderstandingModule(now=now, llm_provider=valid_provider, ambiguity_llm_threshold=0.30)
        parsed = qu_valid.parse(ambiguous_raw)
        self.assertEqual(valid_provider.parse_calls, 1)
        self.assertEqual(parsed.intent, "recommend")
        self.assertIn("finance", parsed.keywords)
        self.assertEqual(parsed.filename_hints, ("budget_v2.xlsx",))
        self.assertEqual(parsed.file_types, ("csv", "tsv", "xls", "xlsx"))
        self.assertIsNotNone(parsed.date_window)
        self.assertEqual(parsed.date_window.label if parsed.date_window else "", "last week")

        # Unambiguous query below threshold makes ZERO LLM calls
        unambiguous_provider = _StubLLMProvider()
        qu_unambiguous = QueryUnderstandingModule(
            now=now,
            llm_provider=unambiguous_provider,
            ambiguity_llm_threshold=0.50,
        )
        qu_unambiguous.parse("find 'q3_budget.xlsx' marketing forecast report from last week")
        self.assertEqual(unambiguous_provider.parse_calls, 0)

        # Invalid refinements must all be rejected and fall back to rule_draft unchanged
        baseline_draft = QueryUnderstandingModule(now=now).parse(ambiguous_raw)
        bad_transforms = [
            # 1. Drops filename_hints
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=d.file_types,
                date_window=d.date_window,
                filename_hints=(),
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
            # 2. Invents filename_hint not in raw_query
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=d.file_types,
                date_window=d.date_window,
                filename_hints=d.filename_hints + ("secret.pdf",),
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
            # 3. Drops or alters file_types
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=("pdf",),
                date_window=d.date_window,
                filename_hints=d.filename_hints,
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
            # 4. Invents additional file_type filter not in raw_query
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=d.file_types + ("pdf",),
                date_window=d.date_window,
                filename_hints=d.filename_hints,
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
            # 5. Drops date_window
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=d.file_types,
                date_window=None,
                filename_hints=d.filename_hints,
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
            # 6. Invents a date_window when rule_draft had None
            lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=d.keywords,
                file_types=d.file_types,
                date_window=DateWindow(now - timedelta(days=1), now, "yesterday"),
                filename_hints=d.filename_hints,
                ambiguity_score=0.2,
                parser_confidence=0.9,
            ),
        ]

        for idx, fn in enumerate(bad_transforms):
            with self.subTest(bad_transform=idx):
                bad_provider = _StubLLMProvider(parse_fn=fn)
                qu_bad = QueryUnderstandingModule(now=now, llm_provider=bad_provider, ambiguity_llm_threshold=0.30)
                result = qu_bad.parse(ambiguous_raw)
                self.assertEqual(bad_provider.parse_calls, 1)
                self.assertEqual(result, baseline_draft)

        # Also verify when raw query has NO date/type constraints, LLM cannot invent them
        vague_no_date = "find that thing with numbers maybe"
        draft_no_date = QueryUnderstandingModule(now=now).parse(vague_no_date)
        invent_date_provider = _StubLLMProvider(
            parse_fn=lambda raw, d: ParsedQuery(
                raw=d.raw,
                intent=d.intent,
                keywords=("budget",),
                file_types=("xlsx",),  # invented file_type!
                date_window=d.date_window,
                filename_hints=d.filename_hints,
                ambiguity_score=0.2,
                parser_confidence=0.8,
            )
        )
        qu_no_invent = QueryUnderstandingModule(now=now, llm_provider=invent_date_provider)
        self.assertEqual(qu_no_invent.parse(vague_no_date), draft_no_date)

    def test_07_request_payload_respects_max_input_chars_across_all_three_operations_and_both_providers(
        self,
    ) -> None:
        record = _LoopbackRecord()
        long_tail_marker = "TAIL_MARKER_NEVER_TRANSMITTED_99999"
        long_content = ("Quarterly budget forecast numbers and revenue analysis. " * 200) + long_tail_marker
        secret_host_dir = "/Users/private_user/ConfidentialVault/DeepSubfolder"

        # 6 candidates with long titles, metadata, matched terms, and 10,000+ char content
        candidates = tuple(
            SearchCandidate(
                document=_make_doc(
                    doc_id=f"doc-{i}",
                    name=f"budget_2026_{i}.xlsx",
                    title=f"Budget 2026 Comprehensive Departmental Forecast Report Part {i} " + ("ExtraTitle " * 8),
                    extension="xlsx",
                    content=long_content,
                    root_dir=secret_host_dir,
                    metadata={"author": "Alice Finance " * 6, "project": "EnterpriseBudgetPlanning " * 5},
                ),
                score=0.85 - i * 0.05,
                tool=RetrievalTool.HYBRID,
                matched_terms=("budget", "forecast", "quarterly", "revenue", "analysis", "departmental"),
                reason="hybrid match",
            )
            for i in range(6)
        )
        # Excessively long raw query and many keywords/hints on rule_draft
        huge_raw_query = (
            "find that thing about quarterly departmental budget forecast numbers " * 80
        ) + long_tail_marker
        huge_draft = ParsedQuery(
            raw=huge_raw_query,
            intent="find",
            keywords=tuple(f"keyword_{i}_long_token_value" for i in range(25)),
            file_types=tuple(f"ext{i}" for i in range(15)),
            filename_hints=tuple(f"hint_file_number_{i}.xlsx" for i in range(15)),
            date_window=DateWindow(datetime(2026, 5, 1), datetime(2026, 5, 20), "last week"),
            ambiguity_score=0.75,
            parser_confidence=0.40,
        )
        router_result = RouterResult(
            parsed_query=huge_draft,
            candidates=candidates,
            trace=_make_router_result(candidates, lane=SearchLane.DEEP).trace,
            forced_exit=False,
        )

        class _BudgetInspectHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                record.record(self.path, dict(self.headers), raw_body)

                outer = json.loads(raw_body)
                if "messages" in outer:
                    sys_text = outer["messages"][0]["content"]
                    user_text = outer["messages"][1]["content"]
                else:
                    sys_text = outer["systemInstruction"]["parts"][0]["text"]
                    user_text = outer["contents"][0]["parts"][0]["text"]

                if "Refine the ambiguous search query" in sys_text:
                    model_json = {
                        "intent": "find",
                        "keywords": ["budget", "forecast"],
                        "file_types": [],
                        "filename_hints": [],
                        "ambiguity_score": 0.40,
                        "parser_confidence": 0.80,
                    }
                elif "Evaluate candidates for the query" in sys_text:
                    start = user_text.find("<untrusted_candidate_data>") + len("<untrusted_candidate_data>")
                    end = user_text.find("</untrusted_candidate_data>")
                    inner = json.loads(user_text[start:end])
                    cands = inner["untrusted_candidates"]
                    model_json = {
                        "candidates": [
                            {"document_id": c["document_id"], "score": 0.8, "reason": "ok"}
                            for c in cands
                        ]
                    }
                else:
                    start = user_text.find("<untrusted_candidate_data>") + len("<untrusted_candidate_data>")
                    end = user_text.find("</untrusted_candidate_data>")
                    inner = json.loads(user_text[start:end])
                    cands = inner["untrusted_candidates"]
                    model_json = {
                        "explanations": [
                            {
                                "document_id": c["document_id"],
                                "explanation": f"Matches {c['title']} via {c['retrieval_tool']} retrieval.",
                                "cited_evidence": ["title", "retrieval_tool"],
                            }
                            for c in cands
                        ]
                    }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if ":generateContent" in self.path:
                    envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(model_json)}]}}]}
                else:
                    envelope = {"choices": [{"message": {"content": json.dumps(model_json)}}]}
                self.wfile.write(json.dumps(envelope).encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _BudgetInspectHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            max_chars = 850
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gem-key"}):
                for provider_name, base_url in (
                    ("ollama", f"http://127.0.0.1:{port}/v1"),
                    ("gemini", f"http://127.0.0.1:{port}/v1beta"),
                ):
                    with self.subTest(provider=provider_name):
                        record.requests.clear()
                        provider = create_llm_provider(
                            LLMConfig(
                                provider=provider_name,
                                model="budget-test-model",
                                base_url=base_url,
                                max_input_chars=max_chars,
                            )
                        )
                        # Execute all 3 operations with oversized inputs
                        provider.parse_ambiguous_query(huge_raw_query, huge_draft)
                        provider.reason_candidates(huge_draft, candidates, widened=True)
                        provider.generate_explanations(candidates, router_result, UserProfile())

                        self.assertEqual(len(record.requests), 3)
                        for op_idx, req in enumerate(record.requests):
                            sent_json = json.loads(req["body"])
                            if provider_name == "ollama":
                                sys_prompt = sent_json["messages"][0]["content"]
                                user_prompt = sent_json["messages"][1]["content"]
                            else:
                                sys_prompt = sent_json["systemInstruction"]["parts"][0]["text"]
                                user_prompt = sent_json["contents"][0]["parts"][0]["text"]

                            total_chars = len(sys_prompt) + len(user_prompt)
                            self.assertLessEqual(
                                total_chars,
                                max_chars,
                                f"Operation {op_idx} on {provider_name} exceeded max_input_chars: {total_chars} > {max_chars}",
                            )
                            self.assertNotIn(long_tail_marker, user_prompt)
                            self.assertNotIn(secret_host_dir, user_prompt)
                            self.assertNotIn("ConfidentialVault", user_prompt)
                            # Verify inner JSON inside the untrusted wrapper is intact (never mid-JSON sliced)
                            if op_idx == 0:
                                self.assertTrue(user_prompt.startswith("<untrusted_query_data>"))
                                self.assertTrue(user_prompt.endswith("</untrusted_query_data>"))
                                inner_str = user_prompt[
                                    len("<untrusted_query_data>") : -len("</untrusted_query_data>")
                                ]
                                self.assertIsInstance(json.loads(inner_str), dict)
                            else:
                                self.assertTrue(user_prompt.startswith("<untrusted_candidate_data>"))
                                self.assertTrue(user_prompt.endswith("</untrusted_candidate_data>"))
                                inner_str = user_prompt[
                                    len("<untrusted_candidate_data>") : -len("</untrusted_candidate_data>")
                                ]
                                self.assertIsInstance(json.loads(inner_str), dict)

                        # Also verify that if a caller configures an impossibly small budget for 6 candidates
                        # (e.g. max_input_chars=400), the provider refuses to transmit an oversized request
                        # and raises LLMUnavailableError without making an HTTP call.
                        record.requests.clear()
                        tiny_provider = create_llm_provider(
                            LLMConfig(
                                provider=provider_name,
                                model="budget-test-model",
                                base_url=base_url,
                                max_input_chars=400,
                            )
                        )
                        with self.assertRaises(LLMUnavailableError):
                            tiny_provider.reason_candidates(huge_draft, candidates, widened=True)
                        self.assertEqual(len(record.requests), 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_08_loopback_openai_compatible_and_gemini_native_providers_end_to_end(self) -> None:
        record = _LoopbackRecord()

        class _MultiProviderHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                record.record(self.path, dict(self.headers), raw_body)

                outer = json.loads(raw_body)
                if "messages" in outer:
                    sys_text = outer["messages"][0]["content"]
                    user_text = outer["messages"][1]["content"]
                else:
                    sys_text = outer["systemInstruction"]["parts"][0]["text"]
                    user_text = outer["contents"][0]["parts"][0]["text"]

                if "Refine the ambiguous search query" in sys_text:
                    model_json = {
                        "intent": "recommend",
                        "keywords": ["budget", "planning"],
                        "file_types": [],
                        "filename_hints": [],
                        "ambiguity_score": 0.65,
                        "parser_confidence": 0.60,
                    }
                elif "Evaluate candidates for the query" in sys_text:
                    start = user_text.find("<untrusted_candidate_data>") + len("<untrusted_candidate_data>")
                    end = user_text.find("</untrusted_candidate_data>")
                    inner = json.loads(user_text[start:end])
                    cands = inner["untrusted_candidates"]
                    model_json = {
                        "candidates": [
                            {
                                "document_id": c["document_id"],
                                "score": 0.91 if idx == 0 else 0.62,
                                "reason": "loopback deep reasoning match",
                            }
                            for idx, c in enumerate(cands)
                        ]
                    }
                else:
                    start = user_text.find("<untrusted_candidate_data>") + len("<untrusted_candidate_data>")
                    end = user_text.find("</untrusted_candidate_data>")
                    inner = json.loads(user_text[start:end])
                    cands = inner["untrusted_candidates"]
                    model_json = {
                        "explanations": [
                            {
                                "document_id": c["document_id"],
                                "explanation": f"Recommended because {c['title']} matched via {c['retrieval_tool']} search.",
                                "cited_evidence": ["title", "retrieval_tool"],
                            }
                            for c in cands
                        ]
                    }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if ":generateContent" in self.path:
                    envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(model_json)}]}}]}
                else:
                    envelope = {"choices": [{"message": {"content": json.dumps(model_json)}}]}
                self.wfile.write(json.dumps(envelope).encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _MultiProviderHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            docs = (
                _make_doc(
                    doc_id="doc-1",
                    name="budget_plan.md",
                    title="Budget Planning Notes",
                    extension="md",
                    content="Notes on budget planning and quarterly numbers.",
                ),
            )
            corpus = DocumentCorpus(docs)

            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gem-key"}):
                for provider_cfg in (
                    LLMConfig(
                        provider="ollama",
                        model="qwen2.5:3b",
                        base_url=f"http://127.0.0.1:{port}/v1",
                    ),
                    LLMConfig(
                        provider="gemini",
                        model="gemini-2.5-flash",
                        base_url=f"http://127.0.0.1:{port}/v1beta",
                    ),
                ):
                    with self.subTest(provider=provider_cfg.provider):
                        provider = create_llm_provider(provider_cfg)
                        engine = ScoutXEngine(corpus, llm_provider=provider)
                        # Vague query triggers parse_ambiguous_query, Deep Lane reason_candidates, and generate_explanations
                        resp = engine.search("find that thing with numbers maybe", limit=3)
                        self.assertEqual(len(resp.recommendations), 1)
                        rec = resp.recommendations[0]
                        self.assertEqual(rec.explanation_source, "llm")
                        self.assertEqual(rec.explanation_model, provider_cfg.model)
                        self.assertEqual(rec.cited_evidence, ("title", "retrieval_tool"))
                        deep_steps = [
                            s for s in resp.router_result.trace if s.strategy.lane == SearchLane.DEEP
                        ]
                        self.assertTrue(deep_steps)
                        self.assertEqual(deep_steps[-1].reasoning_status, ReasoningStatus.EXECUTED)
        finally:
            server.shutdown()
            server.server_close()

    def test_09_loopback_provider_timeout_http_500_and_malformed_json_handling(self) -> None:
        record = _LoopbackRecord()

        class _FaultyHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 0:
                    self.rfile.read(length)
                if record.delay_seconds > 0:
                    time.sleep(record.delay_seconds)
                if record.mode == "http_500":
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "internal server error"}')
                    return
                if record.mode == "malformed_json":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"choices": [{"message": {"content": "{not_valid_json"}}]}')
                    return

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _FaultyHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            doc = _make_doc(doc_id="doc-fault", title="Fault Test Doc")
            corpus = DocumentCorpus((doc,))
            provider = OpenAICompatibleLLMProvider(
                LLMConfig(
                    provider="ollama",
                    model="local-model",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    timeout_seconds=0.1,
                )
            )

            # 1. Timeout -> LLMTimeoutError
            record.delay_seconds = 0.35
            record.mode = "http_500"
            with self.assertRaises(LLMTimeoutError):
                provider.parse_ambiguous_query("that thing", ParsedQuery(raw="that thing", intent="find"))

            # 2. HTTP 500 -> LLMUnavailableError, and router records ReasoningStatus.FAILED while preserving candidates
            record.delay_seconds = 0.0
            record.mode = "http_500"
            router = ComplexityRoutingAgent(RetrievalToolbox(corpus), reasoner=provider)
            deep_query = ParsedQuery(
                raw="that thing maybe",
                intent="find",
                keywords=("fault",),
                ambiguity_score=0.85,
                parser_confidence=0.25,
            )
            r_res = router.run(deep_query)
            self.assertEqual(r_res.trace[-1].reasoning_status, ReasoningStatus.FAILED)
            self.assertTrue(r_res.candidates)

            # 3. Malformed JSON -> LLMResponseFormatError
            record.mode = "malformed_json"
            with self.assertRaises(LLMResponseFormatError):
                provider.generate_explanations(r_res.candidates, r_res, UserProfile())
        finally:
            server.shutdown()
            server.server_close()

    def test_10_cli_llm_flags_offline_fallback_and_missing_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "q3_budget.md").write_text("Q3 marketing budget numbers", encoding="utf-8")

            # 1. --llm-provider without --llm-model exits with code 2
            stderr_buf = io.StringIO()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SCOUTX_LLM_MODEL", None)
                with contextlib.redirect_stderr(stderr_buf):
                    code = cli.main(
                        ["search", "budget", "--root", str(root), "--llm-provider", "ollama"]
                    )
            self.assertEqual(code, 2)
            self.assertIn("--llm-model", stderr_buf.getvalue())

            # 2. --require-llm without provider and without --offline-fallback exits with code 2
            stderr_buf = io.StringIO()
            with contextlib.redirect_stderr(stderr_buf):
                code = cli.main(["search", "budget", "--root", str(root), "--require-llm"])
            self.assertEqual(code, 2)
            self.assertIn("LLM provider is required", stderr_buf.getvalue())

            # 3. --require-llm with --offline-fallback succeeds with code 0 and prints degraded notice
            stdout_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                code = cli.main(
                    ["search", "budget", "--root", str(root), "--require-llm", "--offline-fallback"]
                )
            self.assertEqual(code, 0)
            self.assertIn(DEGRADED_FALLBACK_PREFIX, stdout_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
