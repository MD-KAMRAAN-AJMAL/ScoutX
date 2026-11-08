"""Qdrant full-text/BM25 retrieval using its indexed content payload."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.retrieval.common import SearchCandidate, deduplicate, point_candidates, qdrant_filter
from app.vectorstore.qdrant_client import QdrantStore, get_qdrant_store


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
    query = query.strip()
    if not query:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    store = store or get_qdrant_store()
    store.ensure_ready()
    from qdrant_client import models

    response = store.client.query_points(
        collection_name=store.collection_name,
        query=models.TextQuery(text=models.TextQueryInput(query=query)),
        using="content",
        query_filter=qdrant_filter(
            user_id, extensions=extensions, modalities=modalities, languages=languages
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return deduplicate(
        point_candidates(session, user_id, response.points, retrieval_method="keyword")
    )
