"""Dense semantic retrieval over active Qdrant vectors."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.embeddings.service import EmbeddingService
from app.retrieval.common import SearchCandidate, deduplicate, point_candidates, qdrant_filter
from app.vectorstore.qdrant_client import QdrantStore, get_qdrant_store


def search_semantic(
    session: Session,
    user_id: UUID,
    query: str,
    *,
    limit: int = 20,
    extensions: list[str] | None = None,
    modalities: list[str] | None = None,
    languages: list[str] | None = None,
    embedding_service: EmbeddingService | None = None,
    store: QdrantStore | None = None,
) -> list[SearchCandidate]:
    """Embed one query with Person A's provider and retrieve scoped active vectors."""
    query = query.strip()
    if not query:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    embedding_service = embedding_service or EmbeddingService()
    vectors = embedding_service.embed_queries([query])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an invalid query batch")
    store = store or get_qdrant_store()
    store.ensure_ready()
    from app.vectorstore.collections import DENSE_VECTOR_NAME

    response = store.client.query_points(
        collection_name=store.collection_name,
        query=vectors[0],
        using=DENSE_VECTOR_NAME,
        query_filter=qdrant_filter(
            user_id, extensions=extensions, modalities=modalities, languages=languages
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return deduplicate(
        point_candidates(session, user_id, response.points, retrieval_method="semantic")
    )
