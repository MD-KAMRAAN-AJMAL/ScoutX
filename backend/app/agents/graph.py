"""LangGraph StateGraph for query understanding through run recording."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.evaluator import evaluate_candidates
from app.agents.planner import expand_query, plan_retrieval, understand_query
from app.agents.schemas import QueryUnderstanding, RetrievalPlan, ToolInvocation
from app.agents.state import AgentState
from app.agents.tools import make_retrieval_tools
from app.database.engine import SessionLocal
from app.database.models import AgentRun, File, FileActivity
from app.database.repositories import PreferenceRepository
from app.reranking.context import aggregate_files, build_context
from app.reranking.generation import generate_grounded_answer
from app.reranking.schemas import GroundedContext, RerankedCandidate
from app.reranking.service import rerank_candidates
from app.retrieval.common import SearchCandidate, deduplicate, reciprocal_rank_fusion

logger = logging.getLogger(__name__)
MAX_RETRIES = 1


def load_user_context(session: Session, user_id: UUID) -> dict[str, Any]:
    preference = PreferenceRepository(session).get(user_id)
    recent = session.execute(
        select(FileActivity.event_type, File.filename, File.extension)
        .outerjoin(File, File.id == FileActivity.file_id)
        .where(FileActivity.user_id == user_id)
        .order_by(FileActivity.created_at.desc())
        .limit(12)
    ).all()
    return {
        "preferences": dict(preference.preferences or {}) if preference else {},
        "recent_files": [{"event": e, "filename": n, "extension": x} for e, n, x in recent],
    }


def _trace(state: AgentState, node: str) -> list[str]:
    return [*state.get("execution_trace", []), node]


def _decode_tool_messages(messages: list[ToolMessage], calls: list[ToolInvocation]):
    ranked: list[list[SearchCandidate]] = []
    counts: dict[str, int] = {}
    errors: list[str] = []
    for call, message in zip(calls, messages, strict=True):
        content = message.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            data = json.loads(str(content))
            if not isinstance(data, list):
                raise TypeError("tool output must be a result list")
            found = [SearchCandidate.model_validate(item) for item in data]
            ranked.append(found)
            counts[call.tool] = counts.get(call.tool, 0) + len(found)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            ranked.append([])
            counts[call.tool] = 0
            errors.append(f"{call.tool}: invalid result ({type(exc).__name__})")
    return ranked, counts, errors


def build_agent_graph(
    *,
    user_id: UUID,
    run_id: UUID,
    request_mode: str = "auto",
    session_factory: Callable[[], Session] = SessionLocal,
    llm: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    retrieval_tools: list[BaseTool] | None = None,
    context_loader: Callable[[Session, UUID], dict[str, Any]] | None = None,
    run_recorder: Callable[[AgentState], None] | None = None,
    reranker: Callable[..., Any] | None = None,
    answer_generator: Callable[..., Any] | None = None,
    generate_answer: bool = True,
    search_filters: dict[str, Any] | None = None,
):
    """Build a graph with a fixed user identity and a closed retrieval-tool allowlist."""
    tools = retrieval_tools or make_retrieval_tools(user_id, session_factory)
    tool_node = ToolNode(tools, handle_tool_errors=True)
    allowlist = {tool.name for tool in tools}
    context_loader = context_loader or load_user_context
    reranker = reranker or rerank_candidates
    answer_generator = answer_generator or generate_grounded_answer
    search_filters = search_filters or {}

    async def understand_node(state: AgentState):
        parsed = await understand_query(state["original_query"], llm)
        return {
            "query": parsed.normalized_query,
            "intent": parsed.model_dump(mode="json"),
            "execution_trace": _trace(state, "understand_query"),
        }

    async def context_node(state: AgentState):
        def load():
            with session_factory() as session:
                return context_loader(session, user_id)

        try:
            context = await asyncio.to_thread(load)
        except Exception:
            logger.exception("Unable to load search context")
            context = {"preferences": {}, "recent_files": []}
        return {"user_context": context, "execution_trace": _trace(state, "load_user_context")}

    def plan_node(state: AgentState):
        parsed = QueryUnderstanding.model_validate(state["intent"])
        plan = plan_retrieval(
            parsed,
            request_mode=state.get("request_mode", request_mode),
            query=state.get("query"),
            retry_count=state.get("retry_count", 0),
        )
        if any(call.tool not in allowlist for call in plan.tools):
            raise ValueError("Planner selected a tool outside the registered allowlist")
        return {
            "plan": plan.model_dump(mode="json"),
            "execution_trace": _trace(state, "plan_retrieval"),
        }

    async def execute_node(state: AgentState):
        started = time.perf_counter()
        logger.info(
            "Retrieval started",
            extra={"run_id": str(run_id), "query_length": len(state.get("query", ""))},
        )
        plan = RetrievalPlan.model_validate(state["plan"])
        calls = plan.tools
        tool_calls = []
        for index, call in enumerate(calls):
            if call.tool not in allowlist:
                raise ValueError("Tool call is outside the registered allowlist")
            tool_calls.append(
                {
                    "name": call.tool,
                    "args": call.arguments,
                    "id": f"{run_id}-{state.get('retry_count', 0)}-{index}",
                    "type": "tool_call",
                }
            )
        # ToolNode dispatches independent LangChain tools concurrently and validates their schemas.
        output = await tool_node.ainvoke(
            {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
        )
        messages = [x for x in output.get("messages", []) if isinstance(x, ToolMessage)]
        lists, counts, errors = _decode_tool_messages(messages, calls)
        metadata_indices = [i for i, call in enumerate(calls) if call.tool == "metadata"]
        content_indices = [i for i, call in enumerate(calls) if call.tool != "metadata"]
        if metadata_indices and content_indices:
            eligible_file_ids = {
                candidate.file_id for index in metadata_indices for candidate in lists[index]
            }
            # Metadata is a filter here, not a competing relevance source.
            lists = [
                [candidate for candidate in group if candidate.file_id in eligible_file_ids]
                if index in content_indices
                else []
                for index, group in enumerate(lists)
            ]
        nonempty = [group for group in lists if group]
        merged = (
            reciprocal_rank_fusion(*nonempty, limit=100, method="supervisor_rrf")
            if len(nonempty) > 1
            else deduplicate(nonempty[0] if nonempty else [])
        )
        if search_filters:
            allowed_extensions = search_filters.get("extensions") or []
            merged = [
                item
                for item in merged
                if not allowed_extensions or (item.extension or "").lower() in allowed_extensions
            ]
            if any(
                search_filters.get(key) is not None
                for key in ("root_id", "modified_after", "modified_before")
            ):

                def scoped_file_ids():
                    statement = select(File.id).where(File.user_id == user_id)
                    if search_filters.get("root_id") is not None:
                        statement = statement.where(File.root_id == search_filters["root_id"])
                    if search_filters.get("modified_after") is not None:
                        statement = statement.where(
                            File.modified_at >= search_filters["modified_after"]
                        )
                    if search_filters.get("modified_before") is not None:
                        statement = statement.where(
                            File.modified_at <= search_filters["modified_before"]
                        )
                    if allowed_extensions:
                        statement = statement.where(
                            func.lower(File.extension).in_(allowed_extensions)
                        )
                    with session_factory() as session:
                        return set(session.scalars(statement).all())

                permitted_files = await asyncio.to_thread(scoped_file_ids)
                merged = [item for item in merged if item.file_id in permitted_files]
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Retrieval completed",
            extra={
                "run_id": str(run_id),
                "candidate_count": len(merged),
                "duration_ms": duration_ms,
            },
        )
        return {
            "results": [x.model_dump(mode="json") for x in merged],
            "tools_invoked": [*state.get("tools_invoked", []), *(x.tool for x in calls)],
            "candidate_counts": {**state.get("candidate_counts", {}), **counts},
            "errors": [*state.get("errors", []), *errors],
            "execution_result": "candidates" if merged else "empty",
            "latency_ms": {**state.get("latency_ms", {}), "retrieval": duration_ms},
            "execution_trace": _trace(state, "execute_retrieval"),
        }

    def rerank_node(state: AgentState):
        started = time.perf_counter()
        ranked, status = reranker(
            state.get("query", state["original_query"]), state.get("results", [])
        )
        ranked = [RerankedCandidate.model_validate(item) for item in ranked]
        results = []
        for item in ranked:
            candidate = item.candidate.model_copy(update={"relevance_score": item.final_score})
            results.append(candidate.model_dump(mode="json"))
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "results": results,
            "reranked_candidates": [item.model_dump(mode="json") for item in ranked],
            "reranking_status": status,
            "latency_ms": {**state.get("latency_ms", {}), "reranking": duration_ms},
            "execution_trace": _trace(state, "rerank_results"),
        }

    def evaluate_node(state: AgentState):
        parsed = QueryUnderstanding.model_validate(state["intent"])
        evaluation = evaluate_candidates(
            state.get("query", state["original_query"]), parsed, state.get("results", [])
        )
        return {
            "evaluation": evaluation.model_dump(mode="json"),
            "execution_trace": _trace(state, "evaluate_results"),
        }

    def route_after_evaluation(state: AgentState) -> str:
        if state.get("evaluation", {}).get("sufficient", False):
            return "build_context"
        budget = min(max(state.get("max_retries", MAX_RETRIES), 0), MAX_RETRIES)
        return "expand_query" if state.get("retry_count", 0) < budget else "build_context"

    def expand_node(state: AgentState):
        parsed = QueryUnderstanding.model_validate(state["intent"])
        expanded, _terms = expand_query(state.get("query", state["original_query"]), parsed)
        return {
            "query": expanded,
            "retry_count": state.get("retry_count", 0) + 1,
            "execution_trace": _trace(state, "expand_query"),
        }

    def context_build_node(state: AgentState):
        started = time.perf_counter()
        ranked = [
            RerankedCandidate.model_validate(item) for item in state.get("reranked_candidates", [])
        ]
        files = aggregate_files(ranked)
        context = build_context(ranked)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "files": [item.model_dump(mode="json") for item in files],
            "context": context.model_dump(mode="json"),
            "latency_ms": {**state.get("latency_ms", {}), "context": duration_ms},
            "execution_trace": _trace(state, "build_context"),
        }

    async def answer_node(state: AgentState):
        if not state.get("generate_answer", generate_answer):
            return {
                "answer": None,
                "citations": [],
                "generation_status": "retrieval_only",
                "execution_trace": _trace(state, "generate_answer"),
            }
        context = GroundedContext.model_validate(state.get("context", {}))
        started = time.perf_counter()
        try:
            answer, citations, status = await answer_generator(
                state.get("original_query", state.get("query", "")), context, llm
            )
        except Exception as exc:
            logger.exception("Grounded answer generator raised")
            answer = "I found relevant files, but couldn't generate a grounded summary. Please review the source material."
            citations = []
            status = "llm_error"
            errors = [*state.get("errors", []), f"answer generation failed ({type(exc).__name__})"]
        else:
            errors = state.get("errors", [])
        return {
            "answer": answer,
            "citations": citations,
            "generation_status": status,
            "latency_ms": {
                **state.get("latency_ms", {}),
                "generation": round((time.perf_counter() - started) * 1000, 2),
            },
            "errors": errors,
            "execution_trace": _trace(state, "generate_answer"),
        }

    async def record_node(state: AgentState):
        final_state = {**state, "execution_trace": _trace(state, "record_run")}
        recorder = run_recorder or (
            lambda value: _persist_run(session_factory, user_id, run_id, value)
        )
        await asyncio.to_thread(recorder, final_state)
        return {"execution_trace": final_state["execution_trace"]}

    graph = StateGraph(AgentState)
    nodes = {
        "understand_query": understand_node,
        "load_user_context": context_node,
        "plan_retrieval": plan_node,
        "execute_retrieval": execute_node,
        "rerank_results": rerank_node,
        "evaluate_results": evaluate_node,
        "expand_query": expand_node,
        "build_context": context_build_node,
        "generate_answer": answer_node,
        "record_run": record_node,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "load_user_context")
    graph.add_edge("load_user_context", "plan_retrieval")
    graph.add_edge("plan_retrieval", "execute_retrieval")
    graph.add_edge("execute_retrieval", "rerank_results")
    graph.add_edge("rerank_results", "evaluate_results")
    graph.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {"expand_query": "expand_query", "build_context": "build_context"},
    )
    graph.add_edge("expand_query", "plan_retrieval")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", "record_run")
    graph.add_edge("record_run", END)
    return graph.compile(checkpointer=checkpointer)


def _persist_run(
    session_factory: Callable[[], Session], user_id: UUID, run_id: UUID, state: AgentState
) -> None:
    with session_factory() as session:
        run = session.scalar(
            select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id)
        )
        if run is None:
            return
        run.status = "failed" if state.get("errors") else "succeeded"
        run.plan = state.get("plan", {})
        run.outcome = {
            key: state.get(key)
            for key in (
                "intent",
                "tools_invoked",
                "candidate_counts",
                "retry_count",
                "execution_result",
                "evaluation",
                "reranking_status",
                "generation_status",
                "latency_ms",
                "answer",
                "execution_trace",
                "errors",
            )
        }
        run.finished_at = datetime.now(UTC)
        session.commit()
