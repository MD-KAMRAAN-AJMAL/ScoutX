"""Controlled LangChain tools. Authenticated identity is bound in closures, not arguments."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.embeddings.service import EmbeddingService
from app.retrieval.code import SOURCE_EXTENSIONS, search_code
from app.retrieval.filename import search_exact_filename, search_fuzzy_filename
from app.retrieval.hybrid import search_hybrid
from app.retrieval.keyword import search_keyword
from app.retrieval.media import search_media
from app.retrieval.metadata import search_metadata
from app.retrieval.semantic import search_semantic
from app.vectorstore.qdrant_client import QdrantStore


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilenameArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=1024)


class ContentArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=2000)
    extensions: list[str] | None = Field(default=None, max_length=20)


class MetadataArgs(StrictArgs):
    extensions: list[str] = Field(default_factory=list, max_length=20)
    mime_types: list[str] = Field(default_factory=list, max_length=20)
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    filename_contains: str | None = Field(default=None, max_length=255)
    path_contains: str | None = Field(default=None, max_length=1024)


class CodeArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=2000)
    languages: list[str] = Field(default_factory=list, max_length=12)


class MediaArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=2000)
    media_types: list[str] = Field(default_factory=lambda: ["audio", "video"], max_length=3)
    mode: str = Field(default="keyword", pattern="^(keyword|semantic|hybrid)$")


def make_retrieval_tools(
    user_id: UUID,
    session_factory: Callable[[], Session],
    *,
    embedding_service: EmbeddingService | None = None,
    store: QdrantStore | None = None,
) -> list[StructuredTool]:
    """Build the fixed seven-tool allowlist with server-controlled user scoping."""
    embedder = embedding_service or EmbeddingService()

    def serialize(items) -> str:
        return json.dumps([item.model_dump(mode="json") for item in items])

    def run(operation):
        with session_factory() as session:
            return operation(session)

    def filename(query: str) -> str:
        def search(session):
            found = search_exact_filename(session, user_id, query, limit=20)
            if not found:
                found = search_fuzzy_filename(session, user_id, query, limit=20)
            return serialize(found)

        return run(search)

    def metadata(
        extensions: list[str],
        mime_types: list[str],
        modified_after: datetime | None,
        modified_before: datetime | None,
        filename_contains: str | None,
        path_contains: str | None,
    ) -> str:
        return run(
            lambda session: serialize(
                search_metadata(
                    session,
                    user_id,
                    extensions=extensions or None,
                    mime_types=mime_types or None,
                    modified_after=modified_after,
                    modified_before=modified_before,
                    filename_contains=filename_contains,
                    path_contains=path_contains,
                    limit=1000,
                )
            )
        )

    def keyword(query: str, extensions: list[str] | None = None) -> str:
        return run(
            lambda session: serialize(
                search_keyword(
                    session, user_id, query, limit=30, extensions=extensions, store=store
                )
            )
        )

    def semantic(query: str, extensions: list[str] | None = None) -> str:
        return run(
            lambda session: serialize(
                search_semantic(
                    session,
                    user_id,
                    query,
                    limit=30,
                    extensions=extensions,
                    embedding_service=embedder,
                    store=store,
                )
            )
        )

    def hybrid(query: str, extensions: list[str] | None = None) -> str:
        return run(
            lambda session: serialize(
                search_hybrid(
                    session,
                    user_id,
                    query,
                    limit=30,
                    extensions=extensions,
                    embedding_service=embedder,
                    store=store,
                )
            )
        )

    def code(query: str, languages: list[str] | None = None) -> str:
        return run(
            lambda session: serialize(
                search_code(
                    session,
                    user_id,
                    query,
                    languages=languages,
                    limit=30,
                    embedding_service=embedder,
                    store=store,
                )
            )
        )

    def media(query: str, media_types: list[str] | None = None, mode: str = "keyword") -> str:
        types = media_types or ["audio", "video"]
        if not set(types) <= {"image", "audio", "video"}:
            raise ValueError("invalid media type")
        return run(
            lambda session: serialize(
                search_media(
                    session,
                    user_id,
                    query,
                    media_types=types,
                    mode=mode,
                    limit=30,
                    embedding_service=embedder,
                    store=store,
                )
            )
        )

    return [
        StructuredTool.from_function(
            func=filename,
            name="filename",
            description="Exact/fuzzy authorized filename search.",
            args_schema=FilenameArgs,
        ),
        StructuredTool.from_function(
            func=metadata,
            name="metadata",
            description="Filter indexed files by metadata and path text.",
            args_schema=MetadataArgs,
        ),
        StructuredTool.from_function(
            func=keyword,
            name="keyword",
            description="BM25 keyword search over indexed content.",
            args_schema=ContentArgs,
        ),
        StructuredTool.from_function(
            func=semantic,
            name="semantic",
            description="Dense semantic file-content search.",
            args_schema=ContentArgs,
        ),
        StructuredTool.from_function(
            func=hybrid,
            name="hybrid",
            description="Dense and BM25 search fused by reciprocal rank.",
            args_schema=ContentArgs,
        ),
        StructuredTool.from_function(
            func=code,
            name="code",
            description=f"Search Tree-sitter source chunks: {', '.join(SOURCE_EXTENSIONS)}.",
            args_schema=CodeArgs,
        ),
        StructuredTool.from_function(
            func=media,
            name="media",
            description="Search OCR, audio transcript and video transcript chunks.",
            args_schema=MediaArgs,
        ),
    ]
