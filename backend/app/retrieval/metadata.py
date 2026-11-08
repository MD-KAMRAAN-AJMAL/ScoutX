"""Structured filtering against indexed file metadata."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File
from app.retrieval.common import (
    MetadataSearchInput,
    SearchCandidate,
    active_chunk_condition,
    authenticated_user_id,
    deduplicate,
    normalize_backend_errors,
)


@normalize_backend_errors
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
    limit: int = 100,
) -> list[SearchCandidate]:
    """Search file attributes and return current chunks as provenance-bearing results."""
    args = MetadataSearchInput.model_validate(
        {
            "extensions": extensions,
            "mime_types": mime_types,
            "root_id": root_id,
            "modified_after": modified_after,
            "modified_before": modified_before,
            "filename_contains": filename_contains,
            "limit": limit,
        }
    )
    user_id = authenticated_user_id(user_id)
    stmt = (
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            File.user_id == user_id,
            active_chunk_condition(),
        )
    )
    if args.extensions:
        stmt = stmt.where(
            File.extension.in_([e if e.startswith(".") else f".{e}" for e in args.extensions])
        )
    if args.mime_types:
        stmt = stmt.where(File.mime_type.in_(args.mime_types))
    if args.root_id:
        stmt = stmt.where(File.root_id == args.root_id)
    if args.modified_after:
        stmt = stmt.where(File.modified_at >= args.modified_after)
    if args.modified_before:
        stmt = stmt.where(File.modified_at <= args.modified_before)
    if args.filename_contains:
        stmt = stmt.where(File.filename.ilike(f"%{args.filename_contains.strip()}%"))
    rows = session.execute(
        stmt.order_by(
            File.modified_at.desc().nullslast(), File.canonical_path, DocumentChunk.chunk_index
        ).limit(args.limit)
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
            source_references=list((chunk.chunk_metadata or {}).get("source_references", [])),
        )
        for chunk, file in rows
    )
