"""Ollama model runtime and persistent PostgreSQL-backed graph execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from opentelemetry import trace
from sqlalchemy.orm import Session

from app.agents.graph import MAX_RETRIES, build_agent_graph
from app.config import get_settings
from app.database.engine import SessionLocal

_SETUP_LOCK = asyncio.Lock()
_CHECKPOINTER_READY = False


def build_chat_model() -> ChatOllama:
    cfg = get_settings()
    return ChatOllama(
        model=cfg.ollama_model,
        base_url=str(cfg.ollama_base_url),
        temperature=0,
        client_kwargs={"timeout": cfg.ollama_timeout_seconds},
        async_client_kwargs={"timeout": cfg.ollama_timeout_seconds},
    )


def _postgres_uri(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _open_checkpointer():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    connection = await AsyncConnection.connect(
        _postgres_uri(str(get_settings().database_url)),
        autocommit=True,
        row_factory=dict_row,
        prepare_threshold=0,
    )
    saver = AsyncPostgresSaver(
        connection, serde=JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=[])
    )
    return connection, saver


async def run_supervisor(
    *,
    query: str,
    user_id: UUID,
    run_id: UUID,
    mode: str = "auto",
    conversation_id: UUID | None = None,
    session_factory: Callable[[], Session] = SessionLocal,
    llm: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    retrieval_tools=None,
    context_loader=None,
    run_recorder=None,
    reranker=None,
    answer_generator=None,
    generate_answer: bool = True,
    search_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_args = {
        "user_id": user_id,
        "run_id": run_id,
        "request_mode": mode,
        "session_factory": session_factory,
        "llm": llm or build_chat_model(),
        "retrieval_tools": retrieval_tools,
        "context_loader": context_loader,
        "run_recorder": run_recorder,
        "reranker": reranker,
        "answer_generator": answer_generator,
        "generate_answer": generate_answer,
        "search_filters": search_filters,
    }
    state = {
        "original_query": query,
        "query": query,
        "request_mode": mode,
        "run_id": str(run_id),
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "tools_invoked": [],
        "candidate_counts": {},
        "results": [],
        "errors": [],
        "execution_trace": [],
        "generate_answer": generate_answer,
        "latency_ms": {},
    }
    config = {
        "configurable": {"thread_id": f"{user_id}:{conversation_id or run_id}"},
        "recursion_limit": 40,
        "metadata": {"agent_run_id": str(run_id)},
    }
    if checkpointer is not None:
        graph = build_agent_graph(**graph_args, checkpointer=checkpointer)
        return await _run_graph(graph, state, config, mode=mode)

    global _CHECKPOINTER_READY
    connection, saver = await _open_checkpointer()
    try:
        if not _CHECKPOINTER_READY:
            async with _SETUP_LOCK:
                if not _CHECKPOINTER_READY:
                    await saver.setup()
                    _CHECKPOINTER_READY = True
        graph = build_agent_graph(**graph_args, checkpointer=saver)
        return await _run_graph(graph, state, config, mode=mode)
    finally:
        await connection.close()


async def _execute_graph(graph, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    final = None
    trace: list[str] = []
    async for current in graph.astream(state, config, stream_mode="values"):
        final = current
        if len(current.get("execution_trace", [])) > len(trace):
            trace = list(current["execution_trace"])
    if final is None:
        raise RuntimeError("agent graph returned no state")
    final["execution_trace"] = trace
    return final


async def _run_graph(graph, state, config, *, mode: str) -> dict[str, Any]:
    tracer = trace.get_tracer("scoutx.agent")
    with tracer.start_as_current_span("scoutx.agent.invoke") as span:
        span.set_attribute("scoutx.search.mode", mode)
        span.set_attribute("scoutx.search.answer_enabled", bool(state.get("generate_answer")))
        final = await _execute_graph(graph, state, config)
        span.set_attribute("scoutx.agent.retry_count", int(final.get("retry_count", 0)))
        span.set_attribute("scoutx.agent.candidate_count", len(final.get("results", [])))
        span.set_attribute("scoutx.agent.status", str(final.get("execution_result", "unknown")))
        return final
