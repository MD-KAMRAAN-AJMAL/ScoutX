"""Core data contracts shared by ScoutX modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class SearchLane(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class RetrievalTool(str, Enum):
    FILENAME = "filename"
    METADATA = "metadata"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class DateWindow:
    start: datetime | None = None
    end: datetime | None = None
    label: str = ""

    def contains(self, value: datetime | None) -> bool:
        if value is None:
            return False
        if self.start is not None and value < self.start:
            return False
        if self.end is not None and value > self.end:
            return False
        return True


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    intent: str
    keywords: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()
    date_window: DateWindow | None = None
    filename_hints: tuple[str, ...] = ()
    ambiguity_score: float = 0.0
    parser_confidence: float = 1.0

    @property
    def constraints_count(self) -> int:
        count = 0
        count += len(self.keywords)
        count += len(self.file_types)
        count += 1 if self.date_window else 0
        count += len(self.filename_hints)
        return count


@dataclass(frozen=True)
class FileDocument:
    id: str
    path: Path
    title: str
    extension: str
    content: str
    is_directory: bool = False
    modified_at: datetime | None = None
    created_at: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchCandidate:
    document: FileDocument
    score: float
    tool: RetrievalTool
    matched_terms: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class SearchStrategy:
    lane: SearchLane
    tools: tuple[RetrievalTool, ...]
    requires_reranking: bool = False
    uses_llm_reasoning: bool = False


@dataclass(frozen=True)
class Reflection:
    confident: bool
    coverage: int
    top_score: float
    relevance_to_intent: float
    reason: str


@dataclass(frozen=True)
class RouterTraceStep:
    iteration: int
    complexity_score: float
    strategy: SearchStrategy
    candidates_seen: int
    reflection: Reflection


@dataclass(frozen=True)
class RouterResult:
    parsed_query: ParsedQuery
    candidates: tuple[SearchCandidate, ...]
    trace: tuple[RouterTraceStep, ...]
    forced_exit: bool = False


@dataclass(frozen=True)
class UserProfile:
    file_opens: dict[str, int] = field(default_factory=dict)
    preferred_extensions: dict[str, int] = field(default_factory=dict)
    topic_interests: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Recommendation:
    document: FileDocument
    score: float
    explanation: str
    source_candidate: SearchCandidate


@dataclass(frozen=True)
class SearchResponse:
    recommendations: tuple[Recommendation, ...]
    router_result: RouterResult
