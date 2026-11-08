"""Qdrant collection names, vector schema and indexed payload fields."""

from __future__ import annotations

from app.embeddings.qwen_embedding import EMBEDDING_DIMENSIONS

COLLECTION_NAME = "file_content_chunks"
DENSE_VECTOR_NAME = "dense"
BM25_TEXT_FIELD = "content"

PAYLOAD_INDEXES = {
    "file_id": "keyword",
    "user_id": "keyword",
    "root_id": "keyword",
    "filename": "keyword",
    "canonical_path": "keyword",
    "extension": "keyword",
    "mime_type": "keyword",
    "content_version": "integer",
    "chunk_index": "integer",
    "is_active": "bool",
    "modality": "keyword",
    "language": "keyword",
    "page_number": "integer",
    "slide_number": "integer",
    "sheet_name": "keyword",
    "cell_range": "keyword",
    "line_start": "integer",
    "line_end": "integer",
    "start_time_ms": "integer",
    "end_time_ms": "integer",
    "section_title": "keyword",
}


def ensure_collection(client, collection_name: str = COLLECTION_NAME) -> None:
    """Create the collection/indexes once, rejecting incompatible existing schemas."""
    from qdrant_client import models

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                )
            },
        )
    else:
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if dense is None or dense.size != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} must have named vector "
                f"{DENSE_VECTOR_NAME!r} with {EMBEDDING_DIMENSIONS} dimensions"
            )

    client.create_payload_index(
        collection_name=collection_name,
        field_name=BM25_TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            scoring=True,
            lowercase=True,
        ),
        wait=True,
    )
    schema_types = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "bool": models.PayloadSchemaType.BOOL,
    }
    for field_name, field_type in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_types[field_type],
            wait=True,
        )
