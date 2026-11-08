"""Authenticated file search endpoint backed by the LangGraph supervisor."""

import logging
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from app.agents.supervisor import run_supervisor
from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.models import AgentRun
from app.database.repositories import AgentRunRepository
from app.reranking.schemas import AnswerCitation, FileRecommendation
from app.retrieval.common import SearchCandidate

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=20, ge=1, le=100)
    mode: Literal["auto", "keyword", "semantic", "hybrid"] = "auto"
    conversation_id: UUID | None = None
    generate_answer: bool = True

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchResult(APIModel):
    file_id: str
    chunk_id: str | None
    name: str
    path: str
    score: float = Field(ge=0)
    excerpt: str | None = None
    source_reference: str | None = None
    modality: str
    extension: str | None = None
    retrieval_method: str


class SearchResponse(APIModel):
    query: str
    results: list[SearchResult]
    answer: str | None = None
    run_id: UUID
    conversation_id: UUID
    intent: dict
    retrieval_plan: dict
    tools_invoked: list[str]
    candidate_counts: dict[str, int]
    retry_count: int
    execution_trace: list[str]
    warnings: list[str] = Field(default_factory=list)
    files: list[FileRecommendation] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    reranking_status: str = "unknown"
    generation_status: str = "unknown"
    latency_ms: dict[str, float] = Field(default_factory=dict)


@router.post("", response_model=SearchResponse, responses={503: {"model": ErrorResponse}})
async def search_files(
    request: SearchRequest,
    user_id: UUID = Depends(get_local_user_id),
    session: Session = Depends(get_db),
) -> SearchResponse:
    conversation_id = request.conversation_id or uuid4()
    run = AgentRunRepository(session).create(user_id, request.query)
    session.commit()
    try:
        state = await run_supervisor(
            query=request.query,
            user_id=user_id,
            run_id=run.id,
            mode=request.mode,
            conversation_id=conversation_id,
            generate_answer=request.generate_answer,
        )
    except Exception as exc:
        logger.exception("Search supervisor failed for run %s", run.id)
        stored = session.get(AgentRun, run.id)
        if stored is not None and stored.user_id == user_id:
            stored.status = "failed"
            stored.finished_at = datetime.now(UTC)
            stored.outcome = {"execution_result": "failed", "error_type": type(exc).__name__}
            session.commit()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "search_unavailable",
                "message": "The local search supervisor could not complete this request.",
                "details": {"run_id": str(run.id)},
            },
        ) from exc

    candidates = [SearchCandidate.model_validate(value) for value in state.get("results", [])]
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                file_id=str(x.file_id),
                chunk_id=str(x.chunk_id) if x.chunk_id else None,
                name=x.file_name,
                path=x.file_path,
                score=x.relevance_score,
                excerpt=x.content[:600],
                source_reference=_source_reference(x),
                modality=x.modality,
                extension=x.extension,
                retrieval_method=x.retrieval_method,
            )
            for x in candidates[: request.limit]
        ],
        answer=state.get("answer"),
        run_id=run.id,
        conversation_id=conversation_id,
        intent=state.get("intent", {}),
        retrieval_plan=state.get("plan", {}),
        tools_invoked=state.get("tools_invoked", []),
        candidate_counts=state.get("candidate_counts", {}),
        retry_count=state.get("retry_count", 0),
        execution_trace=state.get("execution_trace", []),
        warnings=state.get("errors", []),
        files=state.get("files", []),
        citations=state.get("citations", []),
        reranking_status=state.get("reranking_status", "unknown"),
        generation_status=state.get("generation_status", "unknown"),
        latency_ms=state.get("latency_ms", {}),
    )


def _source_reference(candidate: SearchCandidate) -> str | None:
    parts = []
    if candidate.page_number is not None:
        parts.append(f"page {candidate.page_number}")
    if candidate.line_start is not None:
        end = (
            f"–{candidate.line_end}"
            if candidate.line_end and candidate.line_end != candidate.line_start
            else ""
        )
        parts.append(f"lines {candidate.line_start}{end}")
    if candidate.start_time_ms is not None:
        start = candidate.start_time_ms // 1000
        end = (candidate.end_time_ms or candidate.start_time_ms) // 1000
        parts.append(f"{start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}")
    return ", ".join(parts) or None
