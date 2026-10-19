"""Core data contracts shared by ScoutX modules."""

from __future__ import annotations

import math
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


class ReasoningStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    EXECUTED = "executed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class RouterThresholds:
    """Configurable initial routing and reflection thresholds."""

    fast_max_complexity: float = 0.35
    standard_max_complexity: float = 0.70
    min_coverage: int = 1
    fast_min_top_score: float = 0.65
    standard_min_top_score: float = 0.50
    deep_min_top_score: float = 0.45
    fast_min_relevance: float = 0.55
    standard_min_relevance: float = 0.50
    deep_min_relevance: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.min_coverage, bool) or not isinstance(self.min_coverage, int) or self.min_coverage < 1:
            raise ValueError("min_coverage must be an integer >= 1")
        for name, value in (
            ("fast_max_complexity", self.fast_max_complexity),
            ("standard_max_complexity", self.standard_max_complexity),
            ("fast_min_top_score", self.fast_min_top_score),
            ("standard_min_top_score", self.standard_min_top_score),
            ("deep_min_top_score", self.deep_min_top_score),
            ("fast_min_relevance", self.fast_min_relevance),
            ("standard_min_relevance", self.standard_min_relevance),
            ("deep_min_relevance", self.deep_min_relevance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(f"{name} must be in (0.0, 1.0], got {value}")
        if not (0.0 < self.fast_max_complexity < self.standard_max_complexity <= 1.0):
            raise ValueError(
                "Complexity thresholds must satisfy 0.0 < fast_max_complexity < standard_max_complexity <= 1.0"
            )

    def required_top_score(self, lane: SearchLane) -> float:
        return {
            SearchLane.FAST: self.fast_min_top_score,
            SearchLane.STANDARD: self.standard_min_top_score,
            SearchLane.DEEP: self.deep_min_top_score,
        }[lane]

    def required_relevance(self, lane: SearchLane) -> float:
        return {
            SearchLane.FAST: self.fast_min_relevance,
            SearchLane.STANDARD: self.standard_min_relevance,
            SearchLane.DEEP: self.deep_min_relevance,
        }[lane]


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
class ToolExecutionOutcome:
    tool: RetrievalTool
    candidates: tuple[SearchCandidate, ...] = ()
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class TraceCandidateEvidence:
    document_id: str
    score: float
    tools: tuple[RetrievalTool, ...]
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchStrategy:
    lane: SearchLane
    tools: tuple[RetrievalTool, ...]
    requires_reranking: bool = False
    uses_llm_reasoning: bool = False
    widened: bool = False
    candidate_limit: int = 10


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
    candidate_evidence: tuple[TraceCandidateEvidence, ...] = ()
    tool_errors: dict[RetrievalTool, str] = field(default_factory=dict)
    forced_exit: bool = False
    reasoning_status: ReasoningStatus = ReasoningStatus.NOT_REQUESTED


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
