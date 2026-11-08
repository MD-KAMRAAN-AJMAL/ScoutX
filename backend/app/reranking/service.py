"""Lazy, batched Qwen cross-encoder reranking with deterministic fallback."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from app.config import get_settings
from app.reranking.schemas import RerankedCandidate
from app.retrieval.common import SearchCandidate

logger = logging.getLogger(__name__)
_MODEL_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[str, str, int, bool], Any] = {}
_INFERENCE_LOCK = threading.RLock()


def _load_cross_encoder(model_name: str, device: str, max_length: int, local_only: bool):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranking requires the optional embedding dependencies; install the embedding extra."
        ) from exc
    return CrossEncoder(
        model_name,
        max_length=max_length,
        device=None if device == "auto" else device,
        automodel_args={"local_files_only": local_only},
        tokenizer_args={"local_files_only": local_only},
    )


def _get_model() -> Any:
    cfg = get_settings()
    key = (
        cfg.reranker_model,
        cfg.embedding_device,
        cfg.reranker_max_length,
        cfg.reranker_local_files_only,
    )
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = _load_cross_encoder(*key)
        return _MODEL_CACHE[key]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700)))
    exp_value = math.exp(max(value, -700))
    return exp_value / (1.0 + exp_value)


def _score(model: Any, query: str, candidates: Sequence[SearchCandidate]) -> list[float]:
    cfg = get_settings()
    scored: list[float] = []
    # CrossEncoder.predict is batched; the lock avoids unsafe concurrent use of one model.
    with _INFERENCE_LOCK:
        for start in range(0, len(candidates), cfg.reranker_batch_size):
            batch = candidates[start : start + cfg.reranker_batch_size]
            pairs = [(query, item.content[:24000]) for item in batch]
            values = model.predict(
                pairs, batch_size=cfg.reranker_batch_size, show_progress_bar=False
            )
            for value in values:
                raw = float(value)
                # Qwen's CrossEncoder returns a logit difference; sigmoid maps it to [0, 1].
                scored.append(_sigmoid(raw))
    return scored


def _rank_fallback(candidates: Sequence[SearchCandidate]) -> list[float]:
    count = len(candidates)
    if count <= 1:
        return [1.0] * count
    # Rank normalization is deliberately transparent and bounded.
    return [1.0 - index / count for index in range(count)]


def rerank_candidates(
    query: str,
    candidates: Sequence[SearchCandidate | dict[str, Any]],
    *,
    model_loader: Callable[[], Any] | None = None,
) -> tuple[list[RerankedCandidate], str]:
    """Score at most the configured candidate limit; return ranked, provenance-safe items."""
    cfg = get_settings()
    parsed = [SearchCandidate.model_validate(x) for x in candidates]
    if not parsed:
        return [], "empty"
    limited = parsed[: cfg.reranker_candidate_limit]
    status = "qwen3"
    started = time.perf_counter()
    if cfg.reranker_enabled:
        try:
            model = (model_loader or _get_model)()
            scores = _score(model, query, limited)
            if len(scores) != len(limited):
                raise ValueError("reranker returned an unexpected number of scores")
        except Exception:
            logger.exception("Qwen reranker failed; applying configured fallback")
            if not cfg.reranker_fallback_enabled:
                scores = [0.0] * len(limited)
                status = "failed"
            else:
                scores = _rank_fallback(limited)
                status = "fallback"
    else:
        scores = _rank_fallback(limited)
        status = "disabled_fallback"
    result = [
        RerankedCandidate(
            candidate=candidate,
            original_retrieval_score=candidate.relevance_score,
            reranker_score=score,
            final_score=score,
        )
        for candidate, score in zip(limited, scores, strict=True)
    ]
    result.sort(key=lambda item: item.final_score, reverse=True)
    logger.info(
        "Reranking completed",
        extra={
            "status": status,
            "candidate_count": len(limited),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return result[: cfg.reranker_top_n], status
