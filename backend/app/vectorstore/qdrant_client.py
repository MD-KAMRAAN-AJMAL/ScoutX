"""Qdrant adapter for versioned file content points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.embeddings.qwen_embedding import EMBEDDING_DIMENSIONS
from app.vectorstore.collections import (
    BM25_SPARSE_VECTOR_NAME,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    ensure_collection,
)


@dataclass(frozen=True)
class VectorSearchHit:
    """A Qdrant match. Hydrate its point ID through DocumentChunkRepository."""

    point_id: UUID
    score: float


class QdrantStore:
    def __init__(self, client=None, *, collection_name: str | None = None):
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection or COLLECTION_NAME
        if client is None:
            from qdrant_client import QdrantClient as Client

            client = Client(url=str(settings.qdrant_url), api_key=settings.qdrant_api_key)
        self.client = client
        self._ready = False

    def ensure_ready(self) -> None:
        if not self._ready:
            ensure_collection(self.client, self.collection_name)
            self._ready = True

    def existing_fingerprints(self, user_id: UUID, point_ids: list[UUID]) -> dict[UUID, str]:
        if not point_ids:
            return {}
        self.ensure_ready()
        found = {}
        for batch in _batches([str(point_id) for point_id in point_ids], 256):
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                fingerprint = payload.get("embedding_fingerprint")
                if (
                    fingerprint
                    and payload.get("user_id") == str(user_id)
                    and payload.get("bm25_index_version") == 1
                ):
                    found[UUID(str(point.id))] = str(fingerprint)
        return found

    def update_point_payload(self, user_id: UUID, point_id: UUID, payload: dict[str, Any]) -> None:
        """Refresh relational metadata without replacing an unchanged vector."""
        if payload.get("user_id") != str(user_id):
            raise ValueError("Updated Qdrant metadata must retain the authenticated user")
        self.ensure_ready()
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[str(point_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not existing or (existing[0].payload or {}).get("user_id") != str(user_id):
            raise LookupError("Qdrant point does not exist for this user")
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={key: value for key, value in payload.items() if value is not None},
            points=[str(point_id)],
            wait=True,
        )

    def existing_active_flags(self, user_id: UUID, point_ids: list[UUID]) -> dict[UUID, bool]:
        if not point_ids:
            return {}
        self.ensure_ready()
        flags: dict[UUID, bool] = {}
        for batch in _batches([str(point_id) for point_id in point_ids], 256):
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            flags.update(
                {
                    UUID(str(point.id)): bool((point.payload or {}).get("is_active", False))
                    for point in points
                    if (point.payload or {}).get("user_id") == str(user_id)
                }
            )
        return flags

    def upsert_chunks(
        self,
        user_id: UUID,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        *,
        active_flags: dict[UUID, bool] | None = None,
    ) -> list[UUID]:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding batch lengths do not match")
        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSIONS or not all(
                math.isfinite(float(value)) for value in vector
            ):
                raise ValueError(
                    f"Each dense vector must contain {EMBEDDING_DIMENSIONS} finite values"
                )
        self.ensure_ready()
        existing_points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[str(chunk["id"]) for chunk in chunks],
            with_payload=True,
            with_vectors=False,
        )
        if any((point.payload or {}).get("user_id") != str(user_id) for point in existing_points):
            raise PermissionError("Cannot replace a Qdrant point owned by another user")
        points = []
        ids = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = UUID(str(chunk["id"]))
            ids.append(point_id)
            payload = dict(chunk["payload"])
            if payload.get("user_id") != str(user_id):
                raise ValueError("Every Qdrant point must belong to the authenticated user")
            payload["is_active"] = bool((active_flags or {}).get(point_id, False))
            payload["bm25_index_version"] = 1
            points.append(
                models.PointStruct(
                    id=str(point_id),
                    vector={
                        DENSE_VECTOR_NAME: vector,
                        BM25_SPARSE_VECTOR_NAME: models.Document(
                            text=str(payload.get("content", "")), model="Qdrant/bm25"
                        ),
                    },
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            verified = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(point_id) for point_id in ids],
                with_payload=True,
                with_vectors=False,
            )
            fingerprints = {
                str(point.id): (
                    (point.payload or {}).get("embedding_fingerprint"),
                    (point.payload or {}).get("bm25_index_version"),
                )
                for point in verified
            }
            if len(fingerprints) != len(ids) or any(
                fingerprints.get(str(point_id))
                != (
                    chunk["payload"].get("embedding_fingerprint"),
                    1,
                )
                for point_id, chunk in zip(ids, chunks, strict=True)
            ):
                raise RuntimeError("Qdrant upsert verification failed")
        return ids

    def dense_search(
        self,
        query_vector: list[float],
        *,
        user_id: UUID,
        limit: int = 20,
        project_id: UUID | None = None,
        authorized_file_ids: list[UUID] | None = None,
        extensions: list[str] | None = None,
        mime_types: list[str] | None = None,
    ) -> list[VectorSearchHit]:
        """Search active dense vectors in authenticated user and optional project scope."""
        from qdrant_client import models

        if len(query_vector) != EMBEDDING_DIMENSIONS or not all(
            math.isfinite(float(value)) for value in query_vector
        ):
            raise ValueError(f"query_vector must have {EMBEDDING_DIMENSIONS} finite values")
        _validate_search_scope(user_id, limit, project_id, authorized_file_ids)
        if project_id is not None and not authorized_file_ids:
            return []
        self.ensure_ready()
        conditions = _search_conditions(
            models,
            user_id=user_id,
            project_id=project_id,
            authorized_file_ids=authorized_file_ids,
            extensions=extensions,
            mime_types=mime_types,
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )
        return [
            VectorSearchHit(point_id=UUID(str(point.id)), score=float(point.score))
            for point in response.points
        ]

    def bm25_search(
        self,
        query: str,
        *,
        user_id: UUID,
        limit: int = 20,
        project_id: UUID | None = None,
        authorized_file_ids: list[UUID] | None = None,
        extensions: list[str] | None = None,
        mime_types: list[str] | None = None,
    ) -> list[VectorSearchHit]:
        """Run Qdrant's indexed BM25 sparse search in authenticated user scope."""
        from qdrant_client import models

        if not query.strip():
            return []
        _validate_search_scope(user_id, limit, project_id, authorized_file_ids)
        if project_id is not None and not authorized_file_ids:
            return []
        self.ensure_ready()
        conditions = _search_conditions(
            models,
            user_id=user_id,
            project_id=project_id,
            authorized_file_ids=authorized_file_ids,
            extensions=extensions,
            mime_types=mime_types,
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model="Qdrant/bm25"),
            using=BM25_SPARSE_VECTOR_NAME,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )
        return [
            VectorSearchHit(point_id=UUID(str(point.id)), score=float(point.score))
            for point in response.points
        ]

    def activate_version(
        self,
        user_id: UUID,
        file_id: UUID,
        content_version: int,
        expected_point_ids: list[UUID],
    ) -> None:
        """Activate the exact verified point set and deactivate older generations."""
        self.ensure_ready()
        expected_ids = [str(point_id) for point_id in expected_point_ids]
        if not expected_ids:
            raise RuntimeError("Cannot activate a file version with no verified Qdrant points")
        for batch in _batches(expected_ids, 256):
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            if len(points) != len(batch) or any(
                (point.payload or {}).get("file_id") != str(file_id)
                or (point.payload or {}).get("user_id") != str(user_id)
                or (point.payload or {}).get("content_version") != content_version
                for point in points
            ):
                raise RuntimeError("Cannot activate a file version with unverified Qdrant points")
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": True},
                points=batch,
                wait=True,
            )
        keep = {str(point_id) for point_id in expected_point_ids}
        for points in self._scroll_file(file_id, user_id=user_id):
            stale = [str(point.id) for point in points if str(point.id) not in keep]
            for batch in _batches(stale, 256):
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"is_active": False},
                    points=batch,
                    wait=True,
                )

    def _delete_points(self, point_ids: list[UUID]) -> None:
        from qdrant_client import models

        if not point_ids:
            return
        self.ensure_ready()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[str(point_id) for point_id in point_ids]),
            wait=True,
        )

    def delete_obsolete_file_points(
        self, user_id: UUID, file_id: UUID, keep_point_ids: list[UUID]
    ) -> None:
        """Delete superseded and orphaned vectors after a new point set is active."""
        self.ensure_ready()
        keep = {str(point_id) for point_id in keep_point_ids}
        for points in self._scroll_file(file_id, user_id=user_id):
            obsolete_ids = [UUID(str(point.id)) for point in points if str(point.id) not in keep]
            self._delete_points(obsolete_ids)

    def delete_file_points(self, user_id: UUID, file_id: UUID) -> None:
        from qdrant_client import models

        self.ensure_ready()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id", match=models.MatchValue(value=str(file_id))
                        ),
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=str(user_id))
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def scroll_file_points(self, user_id: UUID, file_id: UUID) -> list[Any]:
        return [point for page in self._scroll_file(file_id, user_id=user_id) for point in page]

    def _scroll_file(self, file_id: UUID, *, user_id: UUID):
        from qdrant_client import models

        self.ensure_ready()
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="file_id", match=models.MatchValue(value=str(file_id))),
                models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id))),
            ]
        )
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if page:
                yield page
            if offset is None:
                break


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    return QdrantStore()


def _batches(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _validate_search_scope(
    user_id: UUID,
    limit: int,
    project_id: UUID | None,
    authorized_file_ids: list[UUID] | None,
) -> None:
    if not isinstance(user_id, UUID):
        raise TypeError("user_id must be the authenticated user's UUID")
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    if project_id is not None and authorized_file_ids is None:
        raise ValueError("project searches require PostgreSQL-authorized file IDs")


def _search_conditions(
    models,
    *,
    user_id: UUID,
    project_id: UUID | None,
    authorized_file_ids: list[UUID] | None,
    extensions: list[str] | None,
    mime_types: list[str] | None,
):
    conditions = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id))),
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
    ]
    if project_id is not None:
        # Project membership is resolved against PostgreSQL, not a stale vector payload.
        conditions.append(
            models.FieldCondition(
                key="file_id",
                match=models.MatchAny(any=[str(file_id) for file_id in authorized_file_ids or []]),
            )
        )
    if extensions:
        conditions.append(
            models.FieldCondition(
                key="extension",
                match=models.MatchAny(any=[extension.lower() for extension in extensions]),
            )
        )
    if mime_types:
        conditions.append(
            models.FieldCondition(key="mime_type", match=models.MatchAny(any=mime_types))
        )
    return conditions
