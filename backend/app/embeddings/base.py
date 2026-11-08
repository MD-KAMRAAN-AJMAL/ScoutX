"""Embedding provider interfaces shared by indexing and search code."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str
    model_revision: str
    dimensions: int

    @property
    def fingerprint(self) -> str:
        return f"{self.model_name}@{self.model_revision}:dim={self.dimensions}:v1"

    @abstractmethod
    def embed_documents(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed passages without query-only instructions."""

    @abstractmethod
    def embed_queries(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed queries with the provider's retrieval instruction."""
