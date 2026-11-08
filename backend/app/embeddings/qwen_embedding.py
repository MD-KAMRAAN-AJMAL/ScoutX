"""Qwen3-Embedding-0.6B provider with explicit dimension and device checks."""

from __future__ import annotations

import threading
from typing import Any

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# Pin the source revision so reindex fingerprints stay stable across upstream updates.
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_DIMENSIONS = 1024
QUERY_PROMPT = (
    "Instruct: Given a natural-language query, retrieve file passages that are useful "
    "for answering the query.\nQuery:"
)


class Qwen3Embedding(EmbeddingProvider):
    model_name = MODEL_NAME
    model_revision = MODEL_REVISION
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, device: str | None = None, model: Any | None = None):
        settings = get_settings()
        self.device = device or settings.embedding_device
        self.cpu_threads = settings.embedding_cpu_threads
        self._provided_model = model
        self._model = model
        self._lock = threading.RLock()

    def embed_documents(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        return self._encode(texts, query=False, batch_size=batch_size)

    def embed_queries(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        return self._encode(texts, query=True, batch_size=batch_size)

    def _encode(self, texts: list[str], *, query: bool, batch_size: int) -> list[list[float]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input texts must be non-empty strings")
        with self._lock:
            model = self._get_model()
            kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "truncate_dim": self.dimensions,
            }
            if query:
                kwargs["prompt"] = QUERY_PROMPT
            output = model.encode(texts, **kwargs)
        vectors = output.tolist() if hasattr(output, "tolist") else output
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        normalized: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != self.dimensions:
                raise RuntimeError(
                    f"{self.model_name} returned {len(values)} dimensions; expected {self.dimensions}"
                )
            normalized.append(values)
        return normalized

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen embeddings require the optional 'embedding' dependency group "
                    "(sentence-transformers and PyTorch)"
                ) from exc

            selected_device = self._select_device(torch, self.device)
            if selected_device == "cpu":
                torch.set_num_threads(self.cpu_threads)
            model = SentenceTransformer(
                self.model_name,
                revision=self.model_revision,
                device=selected_device,
            )
            actual_dimensions = model.get_sentence_embedding_dimension()
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    f"Configured Qwen revision produces {actual_dimensions} dimensions; "
                    f"Qdrant collection expects {self.dimensions}"
                )
            self.device = selected_device
            self._model = model
            return model

    @staticmethod
    def _select_device(torch, requested: str) -> str:
        requested = requested.lower()
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("SCOUTX_EMBEDDING_DEVICE must be auto, cpu, cuda, or mps")
        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(
            getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        )
        if requested == "auto":
            return "cuda" if cuda_available else "mps" if mps_available else "cpu"
        if requested == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
