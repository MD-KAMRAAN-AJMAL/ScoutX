"""File grouping, budgeted evidence construction, and citation validation."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from hashlib import sha256
from typing import Any

from app.config import get_settings
from app.reranking.schemas import (
    AnswerCitation,
    CitationLocation,
    ContextEvidence,
    FileRecommendation,
    GroundedContext,
    RerankedCandidate,
    SupportingChunk,
)

logger = logging.getLogger(__name__)


def _content_key(file_id: str, chunk_id: str | None, content: str) -> str:
    normalized = " ".join(content.split()).casefold()
    if not normalized:
        return f"{file_id}:{chunk_id or ''}"
    return f"{file_id}:{sha256(normalized.encode('utf-8')).hexdigest()}"


def aggregate_files(
    ranked: list[RerankedCandidate], *, chunks_per_file: int = 3
) -> list[FileRecommendation]:
    grouped: dict[str, list[RerankedCandidate]] = defaultdict(list)
    for item in ranked:
        grouped[str(item.candidate.file_id)].append(item)
    recommendations = []
    for file_id, items in grouped.items():
        items.sort(key=lambda x: x.final_score, reverse=True)
        best = items[0]
        candidate = best.candidate
        distinct: list[RerankedCandidate] = []
        seen: set[str] = set()
        for item in items:
            key = _content_key(
                file_id,
                str(item.candidate.chunk_id) if item.candidate.chunk_id else None,
                item.candidate.content,
            )
            if key not in seen:
                distinct.append(item)
                seen.add(key)
            if len(distinct) >= chunks_per_file:
                break
        # A second independent chunk adds only a small capped confirmation bonus.
        second = distinct[1].final_score if len(distinct) > 1 else 0.0
        file_score = min(1.0, best.final_score + 0.1 * second * (1 - best.final_score))
        recommendations.append(
            FileRecommendation(
                file_id=file_id,
                file_name=candidate.file_name,
                file_path=candidate.file_path,
                extension=candidate.extension,
                mime_type=candidate.mime_type,
                score=file_score,
                retrieval_sources=sorted({item.candidate.retrieval_method for item in distinct}),
                supporting_chunks=[
                    SupportingChunk(
                        chunk_id=str(item.candidate.chunk_id) if item.candidate.chunk_id else None,
                        content=item.candidate.content,
                        snippet=item.candidate.content[:600],
                        score=item.final_score,
                        retrieval_score=item.original_retrieval_score,
                        retrieval_method=item.candidate.retrieval_method,
                        content_version=item.candidate.content_version,
                        section_title=item.candidate.section_title,
                        source_references=item.candidate.source_references,
                        page_number=item.candidate.page_number,
                        line_start=item.candidate.line_start,
                        line_end=item.candidate.line_end,
                        start_time_ms=item.candidate.start_time_ms,
                        end_time_ms=item.candidate.end_time_ms,
                    )
                    for item in distinct
                ],
            )
        )
    return sorted(recommendations, key=lambda item: item.score, reverse=True)


def _words(value: str) -> list[str]:
    return re.findall(r"\S+", value)


def build_context(
    ranked: list[RerankedCandidate], *, token_budget: int | None = None
) -> GroundedContext:
    """Build provenance-preserving evidence using a conservative whitespace token estimate."""
    budget = token_budget or get_settings().rag_context_token_budget
    started = time.perf_counter()
    chosen: list[ContextEvidence] = []
    used = 0
    seen: set[str] = set()
    for item in ranked:
        candidate = item.candidate
        key = _content_key(
            str(candidate.file_id),
            str(candidate.chunk_id) if candidate.chunk_id else None,
            candidate.content,
        )
        if key in seen:
            continue
        seen.add(key)
        tokens = _words(candidate.content)
        remaining = budget - used
        if remaining <= 0:
            break
        if len(tokens) > remaining:
            tokens = tokens[:remaining]
        content = " ".join(tokens)
        if not content:
            continue
        citation_id = f"S{len(chosen) + 1}"
        chosen.append(
            ContextEvidence(
                citation_id=citation_id,
                file_id=str(candidate.file_id),
                chunk_id=str(candidate.chunk_id) if candidate.chunk_id else None,
                file_name=candidate.file_name,
                file_path=candidate.file_path,
                content=content,
                score=item.final_score,
                retrieval_method=candidate.retrieval_method,
                content_version=candidate.content_version,
                section_title=candidate.section_title,
                source_references=candidate.source_references,
                page_number=candidate.page_number,
                line_start=candidate.line_start,
                line_end=candidate.line_end,
                start_time_ms=candidate.start_time_ms,
                end_time_ms=candidate.end_time_ms,
            )
        )
        used += len(tokens)
    context = GroundedContext(evidence=chosen, estimated_tokens=used, token_budget=budget)
    logger.info(
        "RAG context assembled",
        extra={
            "evidence_count": len(chosen),
            "estimated_tokens": used,
            "token_budget": budget,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return context


def citations_for_ids(context: GroundedContext, citation_ids: list[str]) -> list[AnswerCitation]:
    """Map valid IDs and consolidate multiple cited chunks from one file/version."""
    evidence = {entry.citation_id: entry for entry in context.evidence}
    groups: dict[tuple[str, int | None], list[ContextEvidence]] = defaultdict(list)
    seen: set[str] = set()
    for citation_id in citation_ids:
        source = evidence.get(citation_id)
        if source is None or citation_id in seen:
            continue
        seen.add(citation_id)
        groups[(source.file_id, source.content_version)].append(source)
    result = []
    for sources in groups.values():
        first = sources[0]
        locations = [
            CitationLocation(
                citation_id=source.citation_id,
                chunk_id=source.chunk_id,
                content_version=source.content_version,
                relevance_score=source.score,
                section_title=source.section_title,
                source_references=source.source_references,
                page_number=source.page_number,
                line_start=source.line_start,
                line_end=source.line_end,
                start_time_ms=source.start_time_ms,
                end_time_ms=source.end_time_ms,
            )
            for source in sources
        ]
        result.append(
            AnswerCitation(
                citation_id=first.citation_id,
                citation_ids=[source.citation_id for source in sources],
                file_id=first.file_id,
                chunk_id=first.chunk_id,
                file_name=first.file_name,
                file_path=first.file_path,
                content_version=first.content_version,
                relevance_score=max(source.score for source in sources),
                section_title=first.section_title,
                source_references=list(
                    {
                        str(item): item for source in sources for item in source.source_references
                    }.values()
                ),
                page_number=first.page_number,
                line_start=first.line_start,
                line_end=first.line_end,
                start_time_ms=first.start_time_ms,
                end_time_ms=first.end_time_ms,
                locations=locations,
            )
        )
    return result


def as_dicts(model: Any) -> list[dict[str, Any]]:
    """Small serialization helper for agent-state values."""
    return [item.model_dump(mode="json") for item in model]
