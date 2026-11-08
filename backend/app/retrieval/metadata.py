"""Structured filtering against indexed file metadata."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File
from app.retrieval.common import SearchCandidate, active_chunk_condition, deduplicate


def search_metadata(
    session: Session,
    user_id: UUID,
    *,
    extensions: list[str] | None = None,
    mime_types: list[str] | None = None,
    root_id: UUID | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    filename_contains: str | None = None,
    path_contains: str | None = None,
    limit: int = 100,
) -> list[SearchCandidate]:
    """Search file attributes and return current chunks as provenance-bearing results."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    stmt = (
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            File.user_id == user_id,
            active_chunk_condition(),
        )
    )
    if extensions:
        stmt = stmt.where(
            File.extension.in_(
                [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]
            )
        )
    if mime_types:
        stmt = stmt.where(File.mime_type.in_(mime_types))
    if root_id:
        stmt = stmt.where(File.root_id == root_id)
    if modified_after:
        stmt = stmt.where(File.modified_at >= modified_after)
    if modified_before:
        stmt = stmt.where(File.modified_at <= modified_before)
    if filename_contains:
        stmt = stmt.where(File.filename.ilike(f"%{filename_contains.strip()}%"))
    if path_contains:
        stmt = stmt.where(File.canonical_path.ilike(f"%{path_contains.strip()}%"))
    rows = session.execute(
        stmt.order_by(
            File.modified_at.desc().nullslast(), File.canonical_path, DocumentChunk.chunk_index
        ).limit(limit)
    ).all()
    return deduplicate(
        SearchCandidate(
            file_id=file.id,
            chunk_id=chunk.id,
            file_name=file.filename,
            file_path=file.canonical_path,
            content=chunk.content,
            modality=chunk.modality,
            extension=file.extension,
            relevance_score=1.0,
            page_number=chunk.page_number,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            start_time_ms=chunk.start_time_ms,
            end_time_ms=chunk.end_time_ms,
            retrieval_method="metadata",
        )
        for chunk, file in rows
    )
