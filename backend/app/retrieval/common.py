"""Shared result contract and Qdrant/PostgreSQL result translation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from functools import wraps
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client.http.exceptions import ApiException
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File
from app.security.permissions import require_readable_file


class RetrievalBackendError(RuntimeError):
    """Stable error for database, vector-store, or embedding service failures."""


def normalize_backend_errors[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Give tool callers one predictable exception for external retrieval failures."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (SQLAlchemyError, ApiException, httpx.HTTPError, RuntimeError, OSError) as exc:
            raise RetrievalBackendError(f"{function.__name__} backend failed") from exc

    return wrapped


class SearchInput(BaseModel):
    """Validated tool arguments; deliberately excludes authenticated identity."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=20, ge=1, le=1000)


class AuthenticatedSearchContext(BaseModel):
    """Backend-only identity context; never use this model as an LLM tool schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: UUID


def authenticated_user_id(value: UUID) -> UUID:
    """Validate backend identity separately from model-controlled tool arguments."""
    return AuthenticatedSearchContext.model_validate({"user_id": value}).user_id


class QuerySearchInput(SearchInput):
    query: str = Field(min_length=1, max_length=2000)
    extensions: list[str] | None = Field(default=None, max_length=64)
    modalities: list[str] | None = Field(default=None, max_length=16)
    languages: list[str] | None = Field(default=None, max_length=32)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("extensions", "modalities", "languages")
    @classmethod
    def normalize_facets(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [value.strip().lower() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("filter values must not be blank")
        return sorted(set(cleaned))


class FilenameSearchInput(SearchInput):
    filename: str = Field(min_length=1, max_length=1024)
    minimum_similarity: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("filename must not be blank")
        return value


class MetadataSearchInput(SearchInput):
    extensions: list[str] | None = Field(default=None, max_length=64)
    mime_types: list[str] | None = Field(default=None, max_length=64)
    root_id: UUID | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    filename_contains: str | None = Field(default=None, max_length=1024)

    @field_validator("modified_before")
    @classmethod
    def validate_date_range(cls, value: datetime | None, info) -> datetime | None:
        modified_after = info.data.get("modified_after")
        if value is not None and modified_after is not None and value < modified_after:
            raise ValueError("modified_before must be at or after modified_after")
        return value

    @field_validator("extensions", "mime_types")
    @classmethod
    def normalize_metadata_facets(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [value.strip().lower() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("filter values must not be blank")
        return sorted(set(cleaned))


class CodeSearchInput(QuerySearchInput):
    languages: list[str] | None = Field(default=None, max_length=32)


class MediaSearchInput(QuerySearchInput):
    media_types: list[Literal["image", "audio", "video"]] = Field(
        default_factory=lambda: ["image", "audio", "video"], max_length=3
    )
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"


class HybridSearchInput(QuerySearchInput):
    candidate_limit: int | None = Field(default=None, ge=1, le=1000)


class SearchCandidate(BaseModel):
    """A ranked, provenance-preserving result suitable for agent tool output."""

    model_config = ConfigDict(extra="forbid")

    file_id: UUID
    chunk_id: UUID | None
    file_name: str
    file_path: str
    content: str
    modality: str
    extension: str | None
    relevance_score: float = Field(ge=0)
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    retrieval_method: str
    source_references: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def stable_key(self) -> tuple[str, str]:
        """Chunk identity for deduplication and file identity for aggregation."""
        return (str(self.file_id), str(self.chunk_id) if self.chunk_id else "")


def deduplicate(candidates: Iterable[SearchCandidate]) -> list[SearchCandidate]:
    """Keep the highest scoring occurrence of each chunk, with stable tie order."""
    best: dict[tuple[str, str], tuple[int, SearchCandidate]] = {}
    for index, candidate in enumerate(candidates):
        try:
            require_readable_file(candidate.file_path)
        except OSError:
            continue
        key = candidate.stable_key
        current = best.get(key)
        if current is None or candidate.relevance_score > current[1].relevance_score:
            best[key] = (index, candidate)
    return [
        item[1]
        for item in sorted(best.values(), key=lambda pair: (-pair[1].relevance_score, pair[0]))
    ]


def reciprocal_rank_fusion(
    *ranked_lists: Iterable[SearchCandidate],
    limit: int = 20,
    rank_constant: int = 60,
    method: str = "rrf",
) -> list[SearchCandidate]:
    """Fuse rankings using ranks only; dense and BM25 raw scores stay incomparable."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    scores: dict[tuple[str, str], float] = {}
    exemplar: dict[tuple[str, str], SearchCandidate] = {}
    for ranked in ranked_lists:
        seen: set[tuple[str, str]] = set()
        for rank, candidate in enumerate(ranked, start=1):
            key = candidate.stable_key
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
            exemplar.setdefault(key, candidate)
    results = [
        candidate.model_copy(update={"relevance_score": score, "retrieval_method": method})
        for key, score in scores.items()
        for candidate in [exemplar[key]]
    ]
    return sorted(
        results,
        key=lambda result: (
            -result.relevance_score,
            result.file_path,
            result.chunk_id.hex if result.chunk_id else "",
        ),
    )[:limit]


def qdrant_filter(
    user_id: UUID,
    *,
    modalities: list[str] | None = None,
    languages: list[str] | None = None,
    extensions: list[str] | None = None,
    file_ids: list[UUID] | None = None,
):
    """Build mandatory user/active-version filters plus optional validated facets."""
    from qdrant_client import models

    user_id = authenticated_user_id(user_id)
    must = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id))),
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
    ]
    if modalities:
        must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=modalities)))
    if languages:
        must.append(models.FieldCondition(key="language", match=models.MatchAny(any=languages)))
    if extensions:
        must.append(models.FieldCondition(key="extension", match=models.MatchAny(any=extensions)))
    if file_ids is not None:
        must.append(
            models.FieldCondition(
                key="file_id",
                match=models.MatchAny(any=[str(i) for i in file_ids] or [str(UUID(int=0))]),
            )
        )
    return models.Filter(must=must)


def point_candidates(
    session: Session, user_id: UUID, points: Iterable[Any], *, retrieval_method: str
) -> list[SearchCandidate]:
    """Resolve returned Qdrant point IDs to canonical PostgreSQL chunk identities."""
    points = list(points)
    point_ids = []
    for point in points:
        try:
            point_ids.append(UUID(str(point.id)))
        except (ValueError, TypeError, AttributeError):
            continue
    if not point_ids:
        return []
    rows = session.execute(
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        # The Qdrant result is already restricted to active points. Resolve its
        # relational owner without requiring File.version: during a failed
        # replacement that pointer intentionally still names an older version.
        .where(DocumentChunk.qdrant_point_id.in_(point_ids), File.user_id == user_id)
    ).all()
    by_point = {chunk.qdrant_point_id: (chunk, file) for chunk, file in rows}
    candidates = []
    for point in points:
        row = by_point.get(UUID(str(point.id)))
        if row is None:
            continue
        chunk, file = row
        candidates.append(
            SearchCandidate(
                file_id=file.id,
                chunk_id=chunk.id,
                file_name=file.filename,
                file_path=file.canonical_path,
                content=chunk.content,
                modality=chunk.modality,
                extension=file.extension,
                relevance_score=max(0.0, float(point.score or 0.0)),
                page_number=chunk.page_number,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                start_time_ms=chunk.start_time_ms,
                end_time_ms=chunk.end_time_ms,
                retrieval_method=retrieval_method,
                source_references=list((chunk.chunk_metadata or {}).get("source_references", [])),
            )
        )
    return deduplicate(candidates)


def sql_chunk_candidates(
    rows: Iterable[tuple[DocumentChunk, File]],
    *,
    retrieval_method: str,
    scores: dict[UUID, float] | None = None,
) -> list[SearchCandidate]:
    scores = scores or {}
    return deduplicate(
        SearchCandidate(
            file_id=file.id,
            chunk_id=chunk.id,
            file_name=file.filename,
            file_path=file.canonical_path,
            content=chunk.content,
            modality=chunk.modality,
            extension=file.extension,
            relevance_score=max(0.0, float(scores.get(chunk.id, 1.0))),
            page_number=chunk.page_number,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            start_time_ms=chunk.start_time_ms,
            end_time_ms=chunk.end_time_ms,
            retrieval_method=retrieval_method,
            source_references=list((chunk.chunk_metadata or {}).get("source_references", [])),
        )
        for chunk, file in rows
    )


def canonical_filter(
    session: Session,
    user_id: UUID,
    *,
    extensions: list[str] | None = None,
    mime_types: list[str] | None = None,
    root_id: UUID | None = None,
    modified_after=None,
    modified_before=None,
):
    """Return the active chunk generation joined to files owned by this user."""
    statement = (
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        .where(File.user_id == user_id, active_chunk_condition())
    )
    if extensions:
        statement = statement.where(File.extension.in_([e.lower() for e in extensions]))
    if mime_types:
        statement = statement.where(File.mime_type.in_(mime_types))
    if root_id:
        statement = statement.where(File.root_id == root_id)
    if modified_after:
        statement = statement.where(File.modified_at >= modified_after)
    if modified_before:
        statement = statement.where(File.modified_at <= modified_before)
    return statement


def active_chunk_condition():
    """Match the active generation while retaining the previous generation on failure."""
    active_version = File.extra_metadata["active_content_version"].as_integer()
    return or_(
        DocumentChunk.content_version == active_version,
        and_(
            active_version.is_(None),
            File.indexing_status == "indexed",
            DocumentChunk.content_version == File.version,
        ),
    )
