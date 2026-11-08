"""Serializable typed state passed between supervisor graph nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    original_query: str
    query: str
    request_mode: str
    run_id: str
    intent: dict[str, Any]
    user_context: dict[str, Any]
    plan: dict[str, Any]
    results: list[dict[str, Any]]
    tools_invoked: list[str]
    candidate_counts: dict[str, int]
    retry_count: int
    max_retries: int
    evaluation: dict[str, Any]
    context: list[dict[str, Any]]
    reranked_candidates: list[dict[str, Any]]
    files: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    reranking_status: str
    generation_status: str
    generate_answer: bool
    latency_ms: dict[str, float]
    answer: str | None
    errors: list[str]
    execution_trace: list[str]
    execution_result: str
