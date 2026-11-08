from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from app.agents.evaluator import evaluate_candidates
from app.agents.graph import build_agent_graph
from app.agents.planner import deterministic_understanding, plan_retrieval, understand_query
from app.agents.schemas import QueryUnderstanding
from app.agents.tools import make_retrieval_tools
from app.api.search import SearchRequest, search_files
from app.retrieval.common import SearchCandidate

USER = UUID("00000000-0000-4000-8000-000000000001")


@pytest.mark.parametrize(
    ("query", "intent", "tool_name"),
    [
        ("Find main.py", "exact_filename", "filename"),
        ("Find PDFs modified yesterday", "metadata", "metadata"),
        ("Show files in /authorized/Research", "metadata", "metadata"),
        ("Find code containing omp parallel for", "code_identifier", "code"),
        ("Find identifier max_flow in C++", "code_identifier", "code"),
        ("Find max_flow", "code_identifier", "code"),
        ("Find document explaining FlashAttention IO-awareness", "topic", "semantic"),
        ("Find implementation of maximum flow", "code_identifier", "code"),
        ("Find the recording where matrix multiplication was discussed", "video", "media"),
        ("What was that thing I worked on recently?", "ambiguous", "hybrid"),
        ("Find PPTX presentations modified today", "metadata", "metadata"),
        ("Find the image explaining a transformer block", "image", "media"),
    ],
)
def test_query_patterns_route_to_required_tools(query, intent, tool_name):
    understood = deterministic_understanding(query, now=datetime(2026, 11, 8, 12, tzinfo=UTC))
    plan = plan_retrieval(understood)
    assert understood.intent == intent
    assert tool_name in [call.tool for call in plan.tools]
    if intent in {"exact_filename", "metadata"} or (
        intent == "ambiguous" and tool_name == "hybrid"
    ):
        assert [call.tool for call in plan.tools] == [tool_name]
    if intent == "image":
        assert [call.tool for call in plan.tools] == ["metadata", "media"]


def test_exact_filename_remains_filename_only_even_with_manual_mode():
    parsed = deterministic_understanding("Find main.py")
    assert [x.tool for x in plan_retrieval(parsed, request_mode="hybrid").tools] == ["filename"]


@pytest.mark.parametrize(
    ("query", "initial", "retry"),
    [
        ("Find max_flow implementation", "code", "hybrid"),
        ("What was that thing I worked on recently?", "hybrid", "keyword"),
        ("Find the recording about matrix multiplication", "media", "media"),
    ],
)
def test_retry_changes_strategy_within_query_family(query, initial, retry):
    parsed = deterministic_understanding(query)
    first = plan_retrieval(parsed, retry_count=0)
    second = plan_retrieval(parsed, retry_count=1)
    assert first.tools[0].tool == initial
    assert second.tools[0].tool == retry


def test_model_structured_output_is_validated_for_ambiguous_query():
    expected = QueryUnderstanding(
        intent="ambiguous",
        normalized_query="GPU kernel project",
        keywords=["GPU", "kernel"],
        complexity="complex",
        ambiguous=True,
    )

    class Structured:
        async def ainvoke(self, _prompt):
            return expected

    class FakeModel:
        def with_structured_output(self, schema, *, method):
            assert schema is QueryUnderstanding and method == "json_schema"
            return Structured()

    assert (
        asyncio.run(understand_query("What was that thing I worked on recently?", FakeModel()))
        == expected
    )


def test_search_request_rejects_blank_query():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


def test_simple_filename_fast_path_does_not_call_llm():
    class UnexpectedModel:
        def with_structured_output(self, *_args, **_kwargs):
            raise AssertionError("simple queries must use deterministic routing")

    result = asyncio.run(understand_query("Find main.py", UnexpectedModel()))
    assert result.intent == "exact_filename"


def test_tools_have_fixed_identity_and_reject_extra_authority_args():
    tools = make_retrieval_tools(USER, lambda: None)
    assert {x.name for x in tools} == {
        "filename",
        "metadata",
        "keyword",
        "semantic",
        "hybrid",
        "code",
        "media",
    }
    for item in tools:
        assert "user_id" not in item.args_schema.model_fields
        assert "root_id" not in item.args_schema.model_fields


def test_evaluator_uses_terminal_fastpath_for_metadata_and_retries_topic():
    metadata = QueryUnderstanding(intent="metadata", normalized_query="pdf from yesterday")
    topic = QueryUnderstanding(intent="topic", normalized_query="flash attention")
    assert evaluate_candidates("pdf from yesterday", metadata, []).sufficient
    assert not evaluate_candidates("flash attention", topic, []).sufficient


def test_api_returns_trace_sources_and_conversation_id(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.rows = {}

        def add(self, row):
            row.id = uuid4()
            self.rows[row.id] = row

        def flush(self):
            pass

        def commit(self):
            pass

        def get(self, _model, run_id):
            return self.rows.get(run_id)

    candidate = SearchCandidate(
        file_id=uuid4(),
        chunk_id=uuid4(),
        file_name="paper.pdf",
        file_path="/authorized/paper.pdf",
        content="FlashAttention",
        modality="text",
        extension=".pdf",
        relevance_score=0.8,
        page_number=3,
        retrieval_method="semantic",
    )

    async def fake_run(**kwargs):
        assert kwargs["user_id"] == USER
        return {
            "results": [candidate.model_dump(mode="json")],
            "intent": {"intent": "topic"},
            "plan": {"tools": [{"tool": "semantic"}]},
            "tools_invoked": ["semantic"],
            "candidate_counts": {"semantic": 1},
            "retry_count": 0,
            "execution_trace": ["understand_query", "record_run"],
            "errors": [],
            "answer": None,
        }

    monkeypatch.setattr("app.api.search.run_supervisor", fake_run)
    response = asyncio.run(
        search_files(
            SearchRequest(query="Find FlashAttention", conversation_id=uuid4()), USER, FakeSession()
        )
    )
    assert response.results[0].source_reference == "page 3"
    assert response.tools_invoked == ["semantic"]
    assert response.execution_trace[-1] == "record_run"


def test_graph_runs_semantic_then_adapts_to_hybrid_and_exposes_trace():
    async def exercise():
        hit = {
            "file_id": str(uuid4()),
            "chunk_id": str(uuid4()),
            "file_name": "attention.pdf",
            "file_path": "/authorized/attention.pdf",
            "content": "FlashAttention IO-aware tiling",
            "modality": "text",
            "extension": ".pdf",
            "relevance_score": 0.8,
            "page_number": 4,
            "retrieval_method": "hybrid_rrf",
        }
        calls = {"semantic": 0, "hybrid": 0}

        @tool
        def semantic(query: str, extensions: list[str] | None = None) -> str:
            """Search semantic file chunks."""
            calls["semantic"] += 1
            return "[]"

        @tool
        def hybrid(query: str, extensions: list[str] | None = None) -> str:
            """Search dense and BM25 file chunks."""
            calls["hybrid"] += 1
            return json.dumps([hit])

        run_id = uuid4()
        graph = build_agent_graph(
            user_id=USER,
            run_id=run_id,
            retrieval_tools=[semantic, hybrid],
            checkpointer=MemorySaver(),
            session_factory=lambda: nullcontext(object()),
            context_loader=lambda _session, _user: {"preferences": {}, "recent_files": []},
            run_recorder=lambda _state: None,
        )
        initial = {
            "original_query": "Find document explaining FlashAttention IO-awareness",
            "query": "Find document explaining FlashAttention IO-awareness",
            "request_mode": "auto",
            "run_id": str(run_id),
            "retry_count": 0,
            "max_retries": 1,
            "tools_invoked": [],
            "candidate_counts": {},
            "results": [],
            "errors": [],
            "execution_trace": [],
        }
        config = {"configurable": {"thread_id": f"{USER}:{uuid4()}"}}
        nodes = []
        async for update in graph.astream(initial, config, stream_mode="updates"):
            nodes.extend(update)
        final = (await graph.aget_state(config)).values
        assert calls == {"semantic": 1, "hybrid": 1}
        assert "expand_query" in nodes and "record_run" in nodes
        assert final["tools_invoked"] == ["semantic", "hybrid"]
        assert final["retry_count"] == 1
        assert final["results"][0]["page_number"] == 4
        assert final["execution_trace"].count("execute_retrieval") == 2

    asyncio.run(exercise())


def test_graph_applies_metadata_prefilter_to_parallel_content_results():
    async def exercise():
        eligible = str(uuid4())
        unrelated = str(uuid4())
        chunk = str(uuid4())
        metadata_candidate = {
            "file_id": eligible,
            "chunk_id": str(uuid4()),
            "file_name": "flashattention.pdf",
            "file_path": "/authorized/flashattention.pdf",
            "content": "",
            "modality": "text",
            "extension": ".pdf",
            "relevance_score": 0.1,
            "retrieval_method": "metadata",
        }
        semantic_hits = [
            {
                **metadata_candidate,
                "chunk_id": chunk,
                "content": "FlashAttention is IO-aware",
                "relevance_score": 0.9,
                "retrieval_method": "semantic",
            },
            {
                **metadata_candidate,
                "file_id": unrelated,
                "chunk_id": str(uuid4()),
                "file_name": "other.pdf",
                "content": "FlashAttention mention in another file",
                "relevance_score": 0.8,
                "retrieval_method": "semantic",
            },
        ]
        called = set()

        @tool
        def metadata(
            extensions: list[str],
            mime_types: list[str],
            modified_after: str | None,
            modified_before: str | None,
            filename_contains: str | None,
            path_contains: str | None,
        ) -> str:
            """Filter user files by metadata."""
            called.add("metadata")
            assert extensions == [".pdf"]
            return json.dumps([metadata_candidate])

        @tool
        def semantic(query: str, extensions: list[str] | None = None) -> str:
            """Search semantic chunks."""
            called.add("semantic")
            return json.dumps(semantic_hits)

        run_id = uuid4()
        graph = build_agent_graph(
            user_id=USER,
            run_id=run_id,
            retrieval_tools=[metadata, semantic],
            checkpointer=MemorySaver(),
            session_factory=lambda: nullcontext(object()),
            context_loader=lambda _session, _user: {},
            run_recorder=lambda _state: None,
        )
        initial = {
            "original_query": "Find PDF explaining FlashAttention",
            "query": "Find PDF explaining FlashAttention",
            "request_mode": "auto",
            "run_id": str(run_id),
            "retry_count": 0,
            "max_retries": 0,
            "tools_invoked": [],
            "candidate_counts": {},
            "results": [],
            "errors": [],
            "execution_trace": [],
        }
        config = {"configurable": {"thread_id": str(uuid4())}}
        final = None
        async for state in graph.astream(initial, config, stream_mode="values"):
            final = state
        assert called == {"metadata", "semantic"}
        assert final is not None
        assert [result["file_id"] for result in final["results"]] == [eligible]
        assert final["candidate_counts"] == {"metadata": 1, "semantic": 2}

    asyncio.run(exercise())
