"""OCR and audio/video transcript retrieval with preserved source timing."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.embeddings.service import EmbeddingService
from app.retrieval.common import SearchCandidate, reciprocal_rank_fusion
from app.retrieval.keyword import search_keyword
from app.retrieval.semantic import search_semantic
from app.vectorstore.qdrant_client import QdrantStore

MEDIA_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mpeg",
    ".mpg",
]


def search_media(
    session: Session,
    user_id: UUID,
    query: str,
    *,
    media_types: list[str] | None = None,
    mode: str = "hybrid",
    limit: int = 20,
    embedding_service: EmbeddingService | None = None,
    store: QdrantStore | None = None,
) -> list[SearchCandidate]:
    """Search image OCR and audio/video transcript chunks, retaining timestamps."""
    query = query.strip()
    if not query:
        return []
    if mode not in {"keyword", "semantic", "hybrid"}:
        raise ValueError("mode must be keyword, semantic or hybrid")
    if limit < 1:
        raise ValueError("limit must be positive")
    types = [value.lower() for value in (media_types or ["image", "audio", "video"])]
    if not set(types) <= {"image", "audio", "video"}:
        raise ValueError("media_types must contain image, audio or video")
    # Video transcripts are normalized with audio modality; file extension identifies video.
    modalities = sorted({"image" if item == "image" else "audio" for item in types})
    extensions = [
        extension
        for extension in MEDIA_EXTENSIONS
        if (
            (
                extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
                and "image" in types
            )
            or (
                extension not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
                and ("audio" in types or "video" in types)
            )
        )
    ]
    keyword = search_keyword(
        session,
        user_id,
        query,
        limit=limit,
        extensions=extensions,
        modalities=modalities,
        store=store,
    )
    keyword = [item for item in keyword if item.extension in extensions]
    if mode == "keyword":
        return keyword[:limit]
    semantic = search_semantic(
        session,
        user_id,
        query,
        limit=limit,
        extensions=extensions,
        modalities=modalities,
        embedding_service=embedding_service,
        store=store,
    )
    semantic = [item for item in semantic if item.extension in extensions]
    if mode == "semantic":
        return semantic[:limit]
    return reciprocal_rank_fusion(semantic, keyword, limit=limit, method="media_rrf")
