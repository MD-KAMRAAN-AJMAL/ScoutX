"""Validated query understanding, retrieval plan and evaluation schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Intent = Literal[
    "exact_filename", "metadata", "code_identifier", "video", "image", "topic", "ambiguous"
]
ToolName = Literal["filename", "metadata", "keyword", "semantic", "hybrid", "code", "media"]


class FileConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extensions: list[str] = Field(default_factory=list, max_length=20)
    mime_types: list[str] = Field(default_factory=list, max_length=20)
    filename_contains: str | None = Field(default=None, max_length=255)
    path_contains: str | None = Field(default=None, max_length=1024)
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    languages: list[str] = Field(default_factory=list, max_length=12)
    media_types: list[Literal["image", "audio", "video"]] = Field(default_factory=list)

    @field_validator("extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        result = []
        for value in values:
            ext = value.strip().lower()
            if not ext.startswith("."):
                ext = "." + ext
            if len(ext) > 32 or "/" in ext or "\\" in ext:
                raise ValueError("invalid extension")
            if ext not in result:
                result.append(ext)
        return result

    @model_validator(mode="after")
    def validate_dates(self):
        if (
            self.modified_after
            and self.modified_before
            and self.modified_after > self.modified_before
        ):
            raise ValueError("modified_after must precede modified_before")
        return self


class QueryUnderstanding(BaseModel):
    """LLM/deterministic interpretation without user or root authority fields."""

    model_config = ConfigDict(extra="forbid")
    intent: Intent
    normalized_query: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=24)
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    ambiguous: bool = False
    filename: str | None = Field(default=None, max_length=1024)
    constraints: FileConstraints = Field(default_factory=FileConstraints)

    @field_validator("normalized_query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ToolName
    arguments: dict[str, object] = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    intent: Intent
    tools: list[ToolInvocation] = Field(min_length=1, max_length=3)
    reason: str = Field(max_length=500)
    retry: bool = False


class RelevanceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    candidate_count: int = Field(ge=0)
    lexical_coverage: float = Field(ge=0, le=1)
    reason: str = Field(max_length=500)
