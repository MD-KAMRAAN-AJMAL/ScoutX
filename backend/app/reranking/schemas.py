"""Structured reranking, file aggregation, and grounded answer contracts."""

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.common import SearchCandidate


class RerankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: SearchCandidate
    original_retrieval_score: float = Field(ge=0)
    reranker_score: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)
    personalization_adjustment: float = Field(default=0.0, ge=-0.07, le=0.08)
    personalization_reasons: list[str] = Field(default_factory=list)


class SupportingChunk(BaseModel):
    chunk_id: str | None
    content: str
    snippet: str
    score: float = Field(ge=0, le=1)
    retrieval_score: float = Field(ge=0)
    retrieval_method: str
    content_version: int | None = None
    section_title: str | None = None
    source_references: list[dict] = Field(default_factory=list)
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class FileRecommendation(BaseModel):
    file_id: str
    file_name: str
    file_path: str
    extension: str | None = None
    mime_type: str | None = None
    score: float = Field(ge=0, le=1)
    scoring_method: str = "best_chunk_plus_capped_support"
    retrieval_sources: list[str] = Field(default_factory=list)
    personalization_reasons: list[str] = Field(default_factory=list)
    supporting_chunks: list[SupportingChunk]


class ContextEvidence(BaseModel):
    citation_id: str
    file_id: str
    chunk_id: str | None
    file_name: str
    file_path: str
    content: str
    score: float = Field(ge=0, le=1)
    retrieval_method: str
    content_version: int | None = None
    section_title: str | None = None
    source_references: list[dict] = Field(default_factory=list)
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class GroundedContext(BaseModel):
    evidence: list[ContextEvidence]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)


class CitationLocation(BaseModel):
    citation_id: str
    chunk_id: str | None
    content_version: int | None = None
    relevance_score: float = Field(ge=0, le=1)
    section_title: str | None = None
    source_references: list[dict] = Field(default_factory=list)
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class AnswerCitation(BaseModel):
    citation_id: str
    citation_ids: list[str]
    file_id: str
    chunk_id: str | None
    file_name: str
    file_path: str
    content_version: int | None = None
    relevance_score: float = Field(ge=0, le=1)
    section_title: str | None = None
    source_references: list[dict] = Field(default_factory=list)
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    locations: list[CitationLocation] = Field(default_factory=list)
