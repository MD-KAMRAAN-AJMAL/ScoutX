"""Query understanding module.

This module is intentionally deterministic by default. The architecture allows
LLM assistance for ambiguous language, but deterministic parsing keeps local
integration tests fast and repeatable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from scoutx.models import DateWindow, ParsedQuery

if TYPE_CHECKING:
    from scoutx.llm import LLMProvider


FILE_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "presentation": ("ppt", "pptx", "key"),
    "deck": ("ppt", "pptx", "key"),
    "slide": ("ppt", "pptx", "key"),
    "slides": ("ppt", "pptx", "key"),
    "spreadsheet": ("xls", "xlsx", "csv", "tsv"),
    "sheet": ("xls", "xlsx", "csv", "tsv"),
    "budget": ("xls", "xlsx", "csv", "tsv"),
    "document": ("doc", "docx", "pdf", "txt", "md"),
    "doc": ("doc", "docx"),
    "pdf": ("pdf",),
    "markdown": ("md",),
    "code": ("py", "js", "ts", "tsx", "dart", "go", "rs", "java"),
    "folder": ("folder",),
    "directory": ("folder",),
    "project": ("folder",),
    "repo": ("folder",),
    "repository": ("folder",),
}

KNOWN_FILE_EXTENSIONS = {
    "csv",
    "dart",
    "db",
    "doc",
    "docx",
    "drawio",
    "go",
    "folder",
    "html",
    "java",
    "js",
    "json",
    "key",
    "md",
    "pdf",
    "ppt",
    "pptx",
    "py",
    "rs",
    "sql",
    "sqlite",
    "ts",
    "tsx",
    "txt",
    "xls",
    "xlsx",
    "xml",
    "yaml",
    "yml",
    "tsv",
}

INTENT_TERMS = {
    "find": "find",
    "show": "find",
    "open": "find",
    "search": "find",
    "recommend": "recommend",
    "suggest": "recommend",
}

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "around",
    "at",
    "by",
    "file",
    "find",
    "for",
    "from",
    "i",
    "in",
    "it",
    "last",
    "me",
    "my",
    "of",
    "on",
    "open",
    "recommend",
    "search",
    "show",
    "suggest",
    "that",
    "the",
    "thing",
    "this",
    "to",
    "was",
    "week",
    "while",
    "with",
    "working",
}

VAGUE_TERMS = ("thing", "stuff", "something", "numbers", "while ago", "maybe", "whatever")
VAGUE_KEYWORD_TOKENS = frozenset({"thing", "stuff", "something", "numbers", "maybe", "whatever", "while", "ago"})


@dataclass(frozen=True)
class QueryUnderstandingModule:
    now: datetime | None = None
    llm_provider: LLMProvider | None = None
    ambiguity_llm_threshold: float = 0.50

    def __post_init__(self) -> None:
        if (
            isinstance(self.ambiguity_llm_threshold, bool)
            or not isinstance(self.ambiguity_llm_threshold, (int, float))
            or not math.isfinite(self.ambiguity_llm_threshold)
            or not (0.0 <= float(self.ambiguity_llm_threshold) <= 1.0)
        ):
            raise ValueError(
                f"ambiguity_llm_threshold must be a finite number in [0.0, 1.0], got {self.ambiguity_llm_threshold!r}"
            )

    def parse(self, raw_query: str) -> ParsedQuery:
        now = self.now or datetime.now()
        normalized = raw_query.strip()
        lowered = normalized.lower()

        intent = self._parse_intent(lowered)
        date_window = self._parse_date_window(lowered, now)
        file_types = self._parse_file_types(lowered)
        filename_hints = self._parse_filename_hints(normalized)
        keywords = self._parse_keywords(lowered, file_types)

        ambiguity_score = self._ambiguity_score(lowered, keywords, file_types, date_window)
        parser_confidence = self._parser_confidence(
            lowered=lowered,
            keywords=keywords,
            file_types=file_types,
            date_window=date_window,
            filename_hints=filename_hints,
        )

        rule_draft = ParsedQuery(
            raw=raw_query,
            intent=intent,
            keywords=tuple(keywords),
            file_types=tuple(file_types),
            date_window=date_window,
            filename_hints=tuple(filename_hints),
            ambiguity_score=ambiguity_score,
            parser_confidence=parser_confidence,
        )

        if self.llm_provider is not None and rule_draft.ambiguity_score >= self.ambiguity_llm_threshold:
            try:
                refined = self.llm_provider.parse_ambiguous_query(raw_query, rule_draft)
                return self._validate_refined_query(raw_query, rule_draft, refined)
            except Exception:
                return rule_draft

        return rule_draft

    def _validate_refined_query(
        self,
        raw_query: str,
        rule_draft: ParsedQuery,
        refined: object,
    ) -> ParsedQuery:
        """Validate LLM query refinement while preserving all explicit constraints and rejecting invented ones."""
        if not isinstance(refined, ParsedQuery):
            raise TypeError(f"Expected ParsedQuery from parse_ambiguous_query, got {type(refined).__name__}")
        if refined.raw != rule_draft.raw:
            raise ValueError("Refined ParsedQuery must preserve raw query text")
        if refined.intent not in {"find", "recommend"}:
            raise ValueError(f"Invalid refined intent: {refined.intent!r}")

        # 1. Validate numeric bounds
        for field_name, val in (
            ("ambiguity_score", refined.ambiguity_score),
            ("parser_confidence", refined.parser_confidence),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Refined {field_name} must be a finite number, got {val!r}")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Refined {field_name} out of bounds [0.0, 1.0]: {val}")

        # 2. Prevent dropped or invented file_types:
        #    rule_draft.file_types already captures all explicit file types/aliases in raw_query.
        if not isinstance(refined.file_types, tuple) or not all(isinstance(ft, str) for ft in refined.file_types):
            raise TypeError("Refined file_types must be a tuple of strings")
        normalized_refined_types = tuple(sorted({ft.strip().lstrip(".").lower() for ft in refined.file_types if ft.strip()}))
        if any(ft not in KNOWN_FILE_EXTENSIONS for ft in normalized_refined_types):
            raise ValueError(f"Refined file_types contains unknown extension: {normalized_refined_types!r}")
        if normalized_refined_types != tuple(sorted(set(rule_draft.file_types))):
            raise ValueError(
                f"LLM refinement must preserve exact file_types from raw query without dropping or inventing filters: "
                f"expected {rule_draft.file_types!r}, got {normalized_refined_types!r}"
            )

        # 3. Prevent dropped or invented date_window:
        if refined.date_window != rule_draft.date_window:
            raise ValueError(
                f"LLM refinement must preserve exact date_window without dropping or inventing date filters: "
                f"expected {rule_draft.date_window!r}, got {refined.date_window!r}"
            )

        # 4. Preserve all explicit filename_hints and forbid invented filename_hints not in raw_query:
        if not isinstance(refined.filename_hints, tuple) or not all(isinstance(h, str) for h in refined.filename_hints):
            raise TypeError("Refined filename_hints must be a tuple of strings")
        cleaned_hints = self._dedupe([h.strip() for h in refined.filename_hints if h.strip()])
        lower_cleaned_hints = {h.lower() for h in cleaned_hints}
        for required_hint in rule_draft.filename_hints:
            if required_hint.lower() not in lower_cleaned_hints:
                raise ValueError(f"LLM refinement dropped explicit filename_hint: {required_hint!r}")
        raw_lower = raw_query.lower()
        for hint in cleaned_hints:
            if hint.lower() not in raw_lower:
                raise ValueError(f"LLM refinement invented filename_hint not present in raw_query: {hint!r}")

        # 5. Validate topical keywords and preserve all explicit non-vague keywords from rule_draft:
        if not isinstance(refined.keywords, tuple) or not all(isinstance(k, str) for k in refined.keywords):
            raise TypeError("Refined keywords must be a tuple of strings")
        cleaned_keywords: list[str] = []
        for kw in refined.keywords:
            token = kw.strip().lower()
            if not token:
                continue
            if not re.fullmatch(r"[a-z0-9_-]{1,48}", token):
                raise ValueError(f"Invalid refined keyword token: {kw!r}")
            if len(token) <= 2 and not token.isdigit():
                continue
            if token in STOPWORDS:
                continue
            cleaned_keywords.append(token)
        deduped_keywords = self._dedupe(cleaned_keywords)
        if len(deduped_keywords) > 12:
            raise ValueError(f"Too many refined keywords ({len(deduped_keywords)} > 12)")

        explicit_non_vague = [kw for kw in rule_draft.keywords if kw not in VAGUE_KEYWORD_TOKENS]
        keyword_set = set(deduped_keywords)
        for req_kw in explicit_non_vague:
            if req_kw not in keyword_set:
                raise ValueError(f"LLM refinement dropped explicit keyword: {req_kw!r}")

        return ParsedQuery(
            raw=rule_draft.raw,
            intent=refined.intent,
            keywords=tuple(deduped_keywords),
            file_types=rule_draft.file_types,
            date_window=rule_draft.date_window,
            filename_hints=tuple(cleaned_hints),
            ambiguity_score=round(float(refined.ambiguity_score), 2),
            parser_confidence=round(float(refined.parser_confidence), 2),
        )

    def _parse_intent(self, query: str) -> str:
        for token in self._tokens(query):
            if token in INTENT_TERMS:
                return INTENT_TERMS[token]
        return "find"

    def _parse_date_window(self, query: str, now: datetime) -> DateWindow | None:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if "today" in query:
            return DateWindow(today_start, today_start + timedelta(days=1), "today")
        if "yesterday" in query:
            start = today_start - timedelta(days=1)
            return DateWindow(start, today_start, "yesterday")
        if "last week" in query:
            start = today_start - timedelta(days=7)
            return DateWindow(start, today_start, "last week")
        if "this week" in query:
            start = today_start - timedelta(days=today_start.weekday())
            return DateWindow(start, now, "this week")
        if "last month" in query:
            first_this_month = today_start.replace(day=1)
            last_month_end = first_this_month
            last_month_start = (first_this_month - timedelta(days=1)).replace(day=1)
            return DateWindow(last_month_start, last_month_end, "last month")
        if "recent" in query or "recently" in query:
            return DateWindow(today_start - timedelta(days=14), now, "recent")
        if "a while ago" in query or "from a while" in query:
            return DateWindow(None, today_start - timedelta(days=30), "a while ago")
        return None

    def _parse_file_types(self, query: str) -> list[str]:
        extensions: set[str] = set()
        tokens = set(self._tokens(query))
        for token in tokens:
            if token in FILE_TYPE_ALIASES:
                extensions.update(FILE_TYPE_ALIASES[token])
            if token in KNOWN_FILE_EXTENSIONS:
                extensions.add(token)
        for extension in re.findall(r"\b(?:[a-z0-9_-]+)\.([a-z0-9]{1,6})\b", query):
            extensions.add(extension)
        return sorted(extensions)

    def _parse_filename_hints(self, raw_query: str) -> list[str]:
        hints = re.findall(r"['\"]([^'\"]+)['\"]", raw_query)
        hints.extend(re.findall(r"\b[\w-]+\.[A-Za-z0-9]{1,6}\b", raw_query))
        return self._dedupe([hint.strip() for hint in hints if hint.strip()])

    def _parse_keywords(self, query: str, file_types: list[str]) -> list[str]:
        alias_terms = {alias for alias in FILE_TYPE_ALIASES if set(FILE_TYPE_ALIASES[alias]) & set(file_types)}
        file_type_terms = set(file_types)
        keywords: list[str] = []
        for token in self._tokens(query):
            if token in STOPWORDS or token in alias_terms or token in FILE_TYPE_ALIASES or token in file_type_terms:
                continue
            if len(token) <= 2 and not token.isdigit():
                continue
            keywords.append(token)
        return self._dedupe(keywords)

    def _ambiguity_score(
        self,
        query: str,
        keywords: list[str],
        file_types: list[str],
        date_window: DateWindow | None,
    ) -> float:
        score = 0.0
        score += 0.25 if len(keywords) == 0 else 0.0
        score += 0.20 if len(keywords) <= 1 else 0.0
        score += 0.15 if not file_types else 0.0
        score += 0.10 if date_window is None else 0.0
        score += sum(0.12 for term in VAGUE_TERMS if term in query)
        return round(min(score, 0.9), 2)

    def _parser_confidence(
        self,
        lowered: str,
        keywords: list[str],
        file_types: list[str],
        date_window: DateWindow | None,
        filename_hints: list[str],
    ) -> float:
        tokens = self._tokens(lowered)
        if not tokens:
            return 0.10
        confidence = 0.55
        if filename_hints:
            confidence += 0.20
        if keywords:
            confidence += 0.15
        if file_types or date_window is not None:
            confidence += 0.15
        vague_hits = sum(1 for term in VAGUE_TERMS if term in lowered)
        confidence -= vague_hits * 0.12
        return round(max(0.10, min(1.0, confidence)), 2)

    def _tokens(self, query: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", query.lower())

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
