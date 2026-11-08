"""Qdrant point construction and consistency checks for persisted chunks."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.database.models import DocumentChunk, File
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import ContentChunk


def point_id_for(chunk_id: UUID, content_fingerprint: str) -> UUID:
    """A model/content-specific deterministic ID keeps a prior index intact."""
    from uuid import uuid5

    return uuid5(chunk_id, content_fingerprint)


def prepare_point(
    chunk: DocumentChunk | ContentChunk,
    file: File,
    service: EmbeddingService,
) -> tuple[UUID, str, dict[str, Any]]:
    content_fingerprint = service.content_fingerprint(chunk.content)
    point_id = point_id_for(chunk.id, content_fingerprint)
    metadata = dict(chunk.chunk_metadata if isinstance(chunk, DocumentChunk) else chunk.metadata)
    source_refs = metadata.get("source_references", [])
    slide_number = next(
        (ref.get("slide_number") for ref in source_refs if ref.get("slide_number") is not None),
        None,
    )
    cell_range = next(
        (ref.get("cell_range") for ref in source_refs if ref.get("cell_range") is not None), None
    )
    payload = {
        "file_id": str(file.id),
        "user_id": str(file.user_id),
        "root_id": str(file.root_id),
        "canonical_path": file.canonical_path,
        "filename": file.filename,
        "extension": file.extension,
        "mime_type": file.mime_type,
        "content_version": chunk.content_version,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "modality": chunk.modality,
        "language": chunk.language,
        "page_number": chunk.page_number,
        "slide_number": slide_number,
        "sheet_name": metadata.get("sheet_name")
        or next((ref.get("sheet_name") for ref in source_refs if ref.get("sheet_name")), None),
        "cell_range": cell_range or metadata.get("cell_range"),
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "start_time_ms": chunk.start_time_ms,
        "end_time_ms": chunk.end_time_ms,
        "section_title": chunk.section_title,
        "source_references": source_refs,
        "embedding_fingerprint": content_fingerprint,
        "embedding_model_fingerprint": service.fingerprint,
    }
    # Qdrant rejects null payload values in some versions and nulls cannot be
    # meaningfully filtered, so only serialize populated metadata fields.
    return (
        point_id,
        content_fingerprint,
        {key: value for key, value in payload.items() if value is not None},
    )


def verify_point_batch(
    stored: dict[UUID, str],
    expected: dict[UUID, str],
) -> bool:
    """Return true only if every expected point exists with the right content hash."""
    return all(stored.get(point_id) == fingerprint for point_id, fingerprint in expected.items())
