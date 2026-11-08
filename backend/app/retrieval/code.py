"""Language-aware exact identifier and semantic source-code retrieval."""

from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import DocumentChunk, File
from app.embeddings.service import EmbeddingService
from app.retrieval.common import (
    CodeSearchInput,
    SearchCandidate,
    active_chunk_condition,
    authenticated_user_id,
    deduplicate,
    normalize_backend_errors,
    reciprocal_rank_fusion,
)
from app.retrieval.keyword import search_keyword
from app.retrieval.semantic import search_semantic
from app.vectorstore.qdrant_client import QdrantStore

SOURCE_EXTENSIONS = [
    ".py",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
]
_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "c#": "c_sharp",
    "javascript": "javascript",
    "typescript": "typescript",
    "python": "python",
    "c": "c",
    "java": "java",
    "rust": "rust",
    "go": "go",
}


@normalize_backend_errors
def search_code(
    session: Session,
    user_id: UUID,
    query: str,
    *,
    languages: list[str] | None = None,
    limit: int = 20,
    embedding_service: EmbeddingService | None = None,
    store: QdrantStore | None = None,
) -> list[SearchCandidate]:
    """Search source chunks by exact identifier, BM25 terms and code semantics."""
    args = CodeSearchInput.model_validate({"query": query, "limit": limit, "languages": languages})
    user_id = authenticated_user_id(user_id)
    query = args.query
    normalized_languages = [
        _LANGUAGE_ALIASES.get(value.strip().lower(), value.strip().lower())
        for value in (args.languages or [])
    ]
    filters = [
        File.extension.in_(SOURCE_EXTENSIONS),
        DocumentChunk.modality == "code",
        DocumentChunk.chunk_metadata["content_type"]
        .as_string()
        .in_(["source", "code", "code_declaration", "import", "identifier", "comment"]),
    ]
    if normalized_languages:
        filters.append(DocumentChunk.language.in_(normalized_languages))
    statement = (
        select(DocumentChunk, File)
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            File.user_id == user_id,
            active_chunk_condition(),
            *filters,
        )
    )
    # PostgreSQL regex word boundaries avoid matching an identifier as part of a longer name.
    identifier = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", query)
    exact_results: list[SearchCandidate] = []
    if identifier:
        pattern = r"\m" + re.escape(query) + r"\M"
        rows = session.execute(
            statement.where(DocumentChunk.content.op("~*")(pattern))
            .order_by(File.canonical_path, DocumentChunk.chunk_index)
            .limit(args.limit)
        ).all()
        exact_results = [
            SearchCandidate(
                file_id=file.id,
                chunk_id=chunk.id,
                file_name=file.filename,
                file_path=file.canonical_path,
                content=chunk.content,
                modality=chunk.modality,
                extension=file.extension,
                mime_type=getattr(file, "mime_type", None),
                content_version=getattr(chunk, "content_version", None),
                section_title=getattr(chunk, "section_title", None),
                relevance_score=1.0,
                page_number=chunk.page_number,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                start_time_ms=chunk.start_time_ms,
                end_time_ms=chunk.end_time_ms,
                retrieval_method="code_identifier",
                source_references=list((chunk.chunk_metadata or {}).get("source_references", [])),
            )
            for chunk, file in rows
        ]

    qdrant_langs = normalized_languages or None
    exact_terms = search_keyword(
        session,
        user_id,
        query,
        limit=args.limit,
        extensions=SOURCE_EXTENSIONS,
        modalities=["code"],
        languages=qdrant_langs,
        store=store,
    )
    # Filter away any non-code or non-function chunks if stale/incorrect payload exists.
    exact_terms = [item for item in exact_terms if item.extension in SOURCE_EXTENSIONS]
    semantic = search_semantic(
        session,
        user_id,
        query,
        limit=args.limit,
        extensions=SOURCE_EXTENSIONS,
        modalities=["code"],
        languages=qdrant_langs,
        embedding_service=embedding_service,
        store=store,
    )
    merged = reciprocal_rank_fusion(
        exact_results, exact_terms, semantic, limit=args.limit, method="code_rrf"
    )
    return deduplicate(merged)
