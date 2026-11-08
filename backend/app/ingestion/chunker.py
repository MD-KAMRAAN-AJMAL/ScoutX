"""Provenance-aware chunking for normalized multimodal extraction results."""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.normalized import ExtractedDocument, ExtractionStatus

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PROSE_TYPES = {"text", "document", "heading", "list", "table"}
_CODE_TYPES = {"code", "code_declaration", "source", "import", "comment", "identifier"}
_NAMESPACE = UUID("18b37a2a-94ed-4c26-a2f1-2e5f78885893")


class ContentChunk(BaseModel):
    """A stable chunk record ready for PostgreSQL persistence and embedding."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    file_id: UUID
    content_version: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    content: str
    modality: str = "text"
    language: str | None = None
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """Make approximately 500-token units with 75-token sliding overlap.

    Token counting is deliberately tokenizer-independent and approximate. Code
    declarations remain whole up to ``max_code_tokens``; very large declarations
    are split with line-range and overlap provenance rather than dropped.
    """

    def __init__(
        self, target_tokens: int = 500, overlap_tokens: int = 75, max_code_tokens: int = 2048
    ):
        if target_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("Require target_tokens > overlap_tokens >= 0")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_code_tokens = max(max_code_tokens, target_tokens)

    def chunk(self, document: ExtractedDocument) -> list[ContentChunk]:
        """Convert extracted blocks into deterministic, versioned chunk records."""
        if document.status in {ExtractionStatus.FAILED, ExtractionStatus.UNSUPPORTED}:
            return []
        headers: dict[str, str] = {}
        prepared: list[tuple[str, Any, str]] = []
        for block in document.blocks:
            if block.content_type == "spreadsheet_header":
                if block.sheet_name:
                    headers[block.sheet_name] = block.content.strip()
                continue
            content = block.content.strip()
            if block.content_type == "spreadsheet_rows" and block.sheet_name in headers:
                content = f"Columns: {headers[block.sheet_name]}\n{content}"
            if content:
                prepared.append((content, block, block.content_type))

        # Merge short neighboring prose paragraphs and transcript segments from
        # the same source location so ordinary chunks approach the target size.
        grouped = self._coalesce(prepared)
        chunks: list[ContentChunk] = []
        chunk_index = 0
        for content, block, content_type, source_blocks in grouped:
            tokens = list(_TOKEN.finditer(content))
            if not tokens:
                continue
            limit = self.max_code_tokens if content_type in _CODE_TYPES else self.target_tokens
            windows = self._windows(tokens, limit)
            # A short complete source declaration stays intact; larger ones are
            # split only at a token boundary while retaining section context.
            for start_idx, end_idx in windows:
                start_char = tokens[start_idx].start()
                end_char = tokens[end_idx - 1].end()
                piece = content[start_char:end_char].strip()
                if not piece:
                    continue
                metadata = dict(block.metadata)
                references = [self._source_reference(source) for source in source_blocks]
                block_warnings = list(
                    dict.fromkeys(
                        warning
                        for source in source_blocks
                        for warning in source.extraction_warnings
                    )
                )
                metadata.update(
                    {
                        "content_type": content_type,
                        "source_references": references,
                        "parser_name": block.parser_name,
                        "parser_version": block.parser_version,
                        "extraction_warnings": list(
                            dict.fromkeys([*document.warnings, *block_warnings])
                        ),
                        "token_count_approx": end_idx - start_idx,
                    }
                )
                line_start, line_end = self._group_line_range(
                    source_blocks, block, content, start_char, end_char
                )
                start_ms, end_ms = self._group_time_range(
                    source_blocks, block, len(tokens), start_idx, end_idx
                )
                record_id = uuid5(
                    _NAMESPACE, f"{document.file_id}:{document.content_version}:{chunk_index}"
                )
                chunks.append(
                    ContentChunk(
                        id=record_id,
                        file_id=document.file_id,
                        content_version=document.content_version,
                        chunk_index=chunk_index,
                        content=piece,
                        modality=self._modality(content_type),
                        language=block.language,
                        page_number=block.page_number,
                        line_start=line_start,
                        line_end=line_end,
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        section_title=block.section_title,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
        return chunks

    def _coalesce(self, prepared):
        result = []
        current_content = ""
        current_blocks = []
        current_type = ""
        current_key = None

        def flush():
            nonlocal current_content, current_blocks, current_type, current_key
            if current_blocks:
                result.append(
                    (current_content, current_blocks[0], current_type, current_blocks.copy())
                )
            current_content, current_blocks, current_type, current_key = "", [], "", None

        for content, block, content_type in prepared:
            can_merge = content_type in _PROSE_TYPES or content_type == "transcript"
            merge_class = (
                "transcript"
                if content_type == "transcript"
                else "table"
                if content_type == "table"
                else "prose"
            )
            key = (
                merge_class,
                block.page_number,
                block.slide_number,
                block.sheet_name,
                block.language,
            )
            proposed = (
                f"{current_content}{' ' if content_type == 'transcript' else '\n\n'}{content}"
                if current_blocks
                else content
            )
            within_target = len(_TOKEN.findall(proposed)) <= self.target_tokens
            if current_blocks and (not can_merge or key != current_key or not within_target):
                flush()
                proposed = content
            if not can_merge:
                result.append((content, block, content_type, [block]))
                continue
            current_content = proposed
            current_blocks.append(block)
            if merge_class == "transcript":
                current_type = "transcript"
            elif merge_class == "table":
                current_type = "table"
            else:
                current_type = content_type if not current_type else "document"
            current_key = key
        flush()
        return result

    def _windows(self, tokens: list[re.Match[str]], limit: int) -> list[tuple[int, int]]:
        if len(tokens) <= limit:
            return [(0, len(tokens))]
        stride = max(1, limit - self.overlap_tokens)
        result = []
        start = 0
        while start < len(tokens):
            end = min(start + limit, len(tokens))
            result.append((start, end))
            if end == len(tokens):
                break
            start += stride
        return result

    @staticmethod
    def _source_reference(block) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "page_number": block.page_number,
                "slide_number": block.slide_number,
                "sheet_name": block.sheet_name,
                "cell_range": block.cell_range,
                "line_start": block.line_start,
                "line_end": block.line_end,
                "start_time_ms": block.start_time_ms,
                "end_time_ms": block.end_time_ms,
                "section_title": block.section_title,
            }.items()
            if value is not None
        }

    @staticmethod
    def _line_range(block, content: str, start: int, end: int) -> tuple[int | None, int | None]:
        if block.line_start is None:
            return None, None
        before = content[:start].count("\n")
        within = content[start:end].count("\n")
        line_start = block.line_start + before
        line_end = min(block.line_end or (line_start + within), line_start + within)
        return line_start, max(line_start, line_end)

    def _group_line_range(self, source_blocks, block, content, start, end):
        line_blocks = [source for source in source_blocks if source.line_start is not None]
        if not line_blocks:
            return None, None
        if len(line_blocks) == 1:
            return self._line_range(block, content, start, end)
        return min(source.line_start for source in line_blocks), max(
            source.line_end or source.line_start for source in line_blocks
        )

    @staticmethod
    def _time_range(block, token_count: int, start: int, end: int) -> tuple[int | None, int | None]:
        if block.start_time_ms is None or block.end_time_ms is None or token_count == 0:
            return block.start_time_ms, block.end_time_ms
        duration = block.end_time_ms - block.start_time_ms
        return (
            block.start_time_ms + round(duration * start / token_count),
            block.start_time_ms + round(duration * end / token_count),
        )

    def _group_time_range(self, source_blocks, block, token_count, start, end):
        timed = [source for source in source_blocks if source.start_time_ms is not None]
        if len(timed) <= 1:
            return self._time_range(block, token_count, start, end)
        return min(source.start_time_ms for source in timed), max(
            source.end_time_ms for source in timed if source.end_time_ms is not None
        )

    @staticmethod
    def _modality(content_type: str) -> str:
        if content_type.startswith("image"):
            return "image"
        if content_type == "transcript":
            return "audio"
        if content_type.startswith("spreadsheet"):
            return "spreadsheet"
        if content_type in _CODE_TYPES:
            return "code"
        return "text"


def chunk_document(
    document: ExtractedDocument, *, target_tokens: int = 500, overlap_tokens: int = 75
) -> list[ContentChunk]:
    """Convenience entry point used by workers and integration code."""
    return DocumentChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens).chunk(
        document
    )
