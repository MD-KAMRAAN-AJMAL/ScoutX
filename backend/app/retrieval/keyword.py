"""Qdrant full-text/BM25 retrieval using its indexed content payload."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

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
def search_keyword(
    session: Session,
    user_id: UUID,
    query: str,
    *,
    limit: int = 20,
    extensions: list[str] | None = None,
    modalities: list[str] | None = None,
    languages: list[str] | None = None,
    store: QdrantStore | None = None,
) -> list[SearchCandidate]:
    """Run Qdrant's BM25 full-text scorer over the indexed ``content`` payload."""
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
    store = store or get_qdrant_store()
    store.ensure_ready()
    from qdrant_client import models

    response = store.client.query_points(
        collection_name=store.collection_name,
        query=models.TextQuery(text=models.TextQueryInput(query=args.query)),
        using="content",
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
        point_candidates(session, user_id, response.points, retrieval_method="keyword")
    )
