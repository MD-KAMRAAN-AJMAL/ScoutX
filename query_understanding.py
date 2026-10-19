"""Query understanding module.

This module is intentionally deterministic by default. The architecture allows
LLM assistance for ambiguous language, but deterministic parsing keeps local
integration tests fast and repeatable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from scoutx.models import DateWindow, ParsedQuery


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


@dataclass(frozen=True)
class QueryUnderstandingModule:
    now: datetime | None = None

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

        return ParsedQuery(
            raw=raw_query,
            intent=intent,
            keywords=tuple(keywords),
            file_types=tuple(file_types),
            date_window=date_window,
            filename_hints=tuple(filename_hints),
            ambiguity_score=ambiguity_score,
            parser_confidence=parser_confidence,
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
