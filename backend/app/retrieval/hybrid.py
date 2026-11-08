"""Hybrid dense plus BM25 retrieval with Qdrant server-side RRF fusion."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.embeddings.service import EmbeddingService
from app.retrieval.common import (
    HybridSearchInput,
    SearchCandidate,
    authenticated_user_id,
    normalize_backend_errors,
    point_candidates,
    qdrant_filter,
    reciprocal_rank_fusion,
)
from app.vectorstore.collections import DENSE_VECTOR_NAME
from app.vectorstore.qdrant_client import QdrantStore, get_qdrant_store


@normalize_backend_errors
def search_hybrid(
    session: Session,
    user_id: UUID,
    query: str,
    *,
    limit: int = 20,
    candidate_limit: int | None = None,
    extensions: list[str] | None = None,
    modalities: list[str] | None = None,
    languages: list[str] | None = None,
    embedding_service: EmbeddingService | None = None,
    store: QdrantStore | None = None,
) -> list[SearchCandidate]:
    """Fuse dense and BM25 rankings natively in Qdrant using reciprocal rank fusion."""
    args = HybridSearchInput.model_validate(
        {
            "query": query,
            "limit": limit,
            "candidate_limit": candidate_limit,
            "extensions": extensions,
            "modalities": modalities,
            "languages": languages,
        }
    )
    user_id = authenticated_user_id(user_id)
    candidates = args.candidate_limit or min(max(args.limit * 4, 40), 1000)
    if candidates < args.limit:
        raise ValueError("candidate_limit must be at least limit")
    embedding_service = embedding_service or EmbeddingService()
    vectors = embedding_service.embed_queries([args.query])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an invalid query batch")
    store = store or get_qdrant_store()
    store.ensure_ready()
    from qdrant_client import models

    q_filter = qdrant_filter(
        user_id,
        extensions=args.extensions,
        modalities=args.modalities,
        languages=args.languages,
    )
    text_query = models.TextQuery(text=models.TextQueryInput(query=args.query))
    response = store.client.query_points(
        collection_name=store.collection_name,
        prefetch=[
            models.Prefetch(
                query=vectors[0], using=DENSE_VECTOR_NAME, filter=q_filter, limit=candidates
            ),
            models.Prefetch(query=text_query, using="content", filter=q_filter, limit=candidates),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=q_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    return point_candidates(session, user_id, response.points, retrieval_method="hybrid_rrf")


def fuse_ranked_candidates(
    dense: list[SearchCandidate],
    keyword: list[SearchCandidate],
    *,
    limit: int = 20,
    rank_constant: int = 60,
) -> list[SearchCandidate]:
    """Application fallback/helper for cases where native fusion is unavailable."""
    return reciprocal_rank_fusion(
        dense, keyword, limit=limit, rank_constant=rank_constant, method="hybrid_rrf"
    )
