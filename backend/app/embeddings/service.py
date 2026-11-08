"""Batching, normalization and fingerprint helpers for embedding providers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from functools import lru_cache

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.qwen_embedding import Qwen3Embedding


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None, *, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else get_settings().embedding_batch_size
        if batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.provider = provider or Qwen3Embedding()
        self.batch_size = batch_size

    @property
    def fingerprint(self) -> str:
        return self.provider.fingerprint

    def embed_documents(
        self, texts: Iterable[str]
    ) -> Iterator[tuple[list[str], list[list[float]]]]:
        """Yield bounded text/vector batches, retaining stable row alignment."""
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) == self.batch_size:
                yield batch, self.provider.embed_documents(batch, batch_size=self.batch_size)
                batch = []
        if batch:
            yield batch, self.provider.embed_documents(batch, batch_size=self.batch_size)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_queries(texts, batch_size=self.batch_size)

    def content_fingerprint(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.fingerprint.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Reuse the embedding model loaded by this process across tasks."""
    return EmbeddingService()
