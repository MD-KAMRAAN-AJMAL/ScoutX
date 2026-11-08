"""Exact and fuzzy filename retrieval over user-owned indexed files."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File
from app.retrieval.common import (
    SearchCandidate,
    active_chunk_condition,
    deduplicate,
    sql_chunk_candidates,
)


def search_exact_filename(
    session: Session, user_id: UUID, filename: str, *, limit: int = 20
) -> list[SearchCandidate]:
    """Find a complete filename match (case insensitive) with current chunks."""
    if not filename.strip():
        return []
    rows = session.execute(
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            File.user_id == user_id,
            active_chunk_condition(),
            func.lower(File.filename) == filename.strip().lower(),
        )
        .order_by(File.canonical_path, DocumentChunk.chunk_index)
        .limit(limit)
    ).all()
    # File-level matches have a unit score; they do not pretend to be content matches.
    return sql_chunk_candidates(rows, retrieval_method="filename_exact")


def search_fuzzy_filename(
    session: Session, user_id: UUID, query: str, *, limit: int = 20, minimum_similarity: float = 0.2
) -> list[SearchCandidate]:
    """Use PostgreSQL pg_trgm similarity and substring matching for typo tolerance."""
    query = query.strip()
    if not query:
        return []
    similarity = func.similarity(File.filename, query)
    rows = session.execute(
        select(DocumentChunk, File, similarity.label("similarity"))
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            File.user_id == user_id,
            active_chunk_condition(),
            (similarity >= minimum_similarity) | (File.filename.ilike(f"%{query}%")),
        )
        .order_by(similarity.desc(), File.canonical_path, DocumentChunk.chunk_index)
        .limit(limit)
    ).all()
    candidates = []
    for chunk, file, score in rows:
        candidates.append(
            SearchCandidate(
                file_id=file.id,
                chunk_id=chunk.id,
                file_name=file.filename,
                file_path=file.canonical_path,
                content=chunk.content,
                modality=chunk.modality,
                extension=file.extension,
                relevance_score=max(0.0, float(score or 0.0)),
                page_number=chunk.page_number,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                start_time_ms=chunk.start_time_ms,
                end_time_ms=chunk.end_time_ms,
                retrieval_method="filename_fuzzy",
            )
        )
    return deduplicate(candidates)
