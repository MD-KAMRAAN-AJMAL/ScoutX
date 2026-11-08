"""Versioned dataset and run-artifact contracts for offline ScoutX evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetDocument(EvaluationModel):
    logical_file_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    filename: str = Field(min_length=1, max_length=255)
    extension: str = Field(min_length=1, max_length=16)
    mime_type: str
    content: str = ""
    modality: str = "text"
    language: str | None = None
    supported: bool = True
    synthetic: bool = True
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EvidenceReference(EvaluationModel):
    logical_file_id: str
    excerpt: str
    page: int | None = Field(default=None, ge=1)
    section: str | None = None


class RelevanceJudgment(EvaluationModel):
    logical_file_id: str
    grade: int = Field(ge=0, le=3)
    source: Literal["human_authored", "synthetic_authored", "imported"] = "synthetic_authored"
    rationale: str | None = None


class EvaluationQuery(EvaluationModel):
    query_id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=2000)
    kind: str
    judgments: list[RelevanceJudgment] = Field(default_factory=list)
    answerable: bool | None = None
    reference_answer: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)
    valid_tools: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_judgments(self) -> EvaluationQuery:
        ids = [item.logical_file_id for item in self.judgments]
        if len(ids) != len(set(ids)):
            raise ValueError("a query may contain only one judgment per logical file")
        return self


class RecommendationScenario(EvaluationModel):
    scenario_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    profile: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    category: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    relevant_file_ids: list[str]
    expected_empty: bool = False


class EvaluationDataset(EvaluationModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str
    judgment_policy: str
    documents: list[DatasetDocument]
    queries: list[EvaluationQuery]
    recommendation_scenarios: list[RecommendationScenario] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> EvaluationDataset:
        document_ids = [item.logical_file_id for item in self.documents]
        query_ids = [item.query_id for item in self.queries]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("logical_file_id values must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")
        known = set(document_ids)
        for query in self.queries:
            for item in query.judgments:
                if item.logical_file_id not in known:
                    raise ValueError(f"unknown judged logical file: {item.logical_file_id}")
            for item in query.evidence:
                if item.logical_file_id not in known:
                    raise ValueError(f"unknown evidence logical file: {item.logical_file_id}")
        for scenario in self.recommendation_scenarios:
            if not set(scenario.relevant_file_ids) <= known:
                raise ValueError(f"unknown file in recommendation scenario: {scenario.scenario_id}")
        return self


class Prediction(EvaluationModel):
    query_id: str
    result_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RankingRun(EvaluationModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_.:-]+$")
    predictions: list[Prediction]
    configuration: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_prediction_queries(self) -> RankingRun:
        ids = [item.query_id for item in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("ranking run contains duplicate query predictions")
        return self


class EvaluationArtifacts(EvaluationModel):
    """Optional observed outputs from ScoutX or another explicitly named provider."""

    dataset: str
    dataset_version: str
    rankings: list[RankingRun] = Field(default_factory=list)
    reranking: list[dict] = Field(default_factory=list)
    rag: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    agent: list[dict] = Field(default_factory=list)
    personalization: list[RankingRun] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)
    ingestion: list[dict] = Field(default_factory=list)
    performance: list[dict] = Field(default_factory=list)
    reliability: list[dict] = Field(default_factory=list)
