"""Dense semantic retrieval over active Qdrant vectors."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.embeddings.service import EmbeddingService
from app.retrieval.common import (
    QuerySearchInput,
    SearchCandidate,
    authenticated_user_id,
    deduplicate,
    normalize_backend_errors,
    point_candidates,
    qdrant_filter,
)
from app.vectorstore.qdrant_client import QdrantStore, get_qdrant_store


@normalize_backend_errors
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
    args = QuerySearchInput.model_validate(
        {
            "query": query,
            "limit": limit,
            "extensions": extensions,
            "modalities": modalities,
            "languages": languages,
        }
    )
    user_id = authenticated_user_id(user_id)
    embedding_service = embedding_service or EmbeddingService()
    vectors = embedding_service.embed_queries([args.query])
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
            user_id,
            extensions=args.extensions,
            modalities=args.modalities,
            languages=args.languages,
        ),
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    return deduplicate(
        point_candidates(session, user_id, response.points, retrieval_method="semantic")
    )
