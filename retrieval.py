"""Retrieval toolbox for ScoutX."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scoutx.document_loader import DocumentLoader
from scoutx.models import FileDocument, ParsedQuery, RetrievalTool, SearchCandidate
from scoutx.text import clamp_score, cosine_similarity, expand_terms, term_counter, tokenize


class DocumentCorpus:
    def __init__(self, documents: tuple[FileDocument, ...]):
        self.documents = documents
        self._by_id = {document.id: document for document in documents}
        self._content_vectors = {
            document.id: term_counter(
                [
                    *tokenize(document.title),
                    *tokenize(document.title),
                    *tokenize(document.content),
                    document.extension,
                ]
            )
            for document in documents
        }
        self._metadata_vectors = {
            document.id: term_counter(
                [
                    *tokenize(document.path.name),
                    *tokenize(" ".join(document.metadata.values())),
                    document.extension,
                ]
            )
            for document in documents
        }

    @classmethod
    def from_directory(cls, root: str | Path) -> "DocumentCorpus":
        return cls(DocumentLoader().load_directory(Path(root)))

    def get(self, document_id: str) -> FileDocument | None:
        return self._by_id.get(document_id)

    def content_vector(self, document: FileDocument):
        return self._content_vectors[document.id]

    def metadata_vector(self, document: FileDocument):
        return self._metadata_vectors[document.id]


class RetrievalToolbox:
    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus

    def search(
        self,
        tool: RetrievalTool,
        parsed_query: ParsedQuery,
        limit: int = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        if tool == RetrievalTool.FILENAME:
            return self.filename_search(parsed_query, limit, widened=widened)
        if tool == RetrievalTool.METADATA:
            return self.metadata_search(parsed_query, limit, widened=widened)
        if tool == RetrievalTool.KEYWORD:
            return self.keyword_search(parsed_query, limit, widened=widened)
        if tool == RetrievalTool.SEMANTIC:
            return self.semantic_search(parsed_query, limit, widened=widened)
        if tool == RetrievalTool.HYBRID:
            return self.hybrid_search(parsed_query, limit, widened=widened)
        raise ValueError(f"Unsupported retrieval tool: {tool}")

    def filename_search(
        self,
        parsed_query: ParsedQuery,
        limit: int | None = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        query_text = " ".join([*parsed_query.filename_hints, *parsed_query.keywords]) or parsed_query.raw
        query_tokens = self._query_terms(parsed_query, expand=widened, widened=widened)
        candidates: list[SearchCandidate] = []

        for document in self.corpus.documents:
            name = document.path.name.lower()
            name_tokens = tokenize(name)
            token_hits = sorted(set(query_tokens) & set(name_tokens))
            prefix_hits: list[str] = []
            if widened:
                prefix_hits = sorted(
                    {
                        q_tok
                        for q_tok in query_tokens
                        if len(q_tok) >= 4
                        and any(n_tok.startswith(q_tok[:4]) or q_tok.startswith(n_tok[:4]) for n_tok in name_tokens if len(n_tok) >= 4)
                    }
                    - set(token_hits)
                )
            exact_hint = any(hint.lower() in name for hint in parsed_query.filename_hints)
            fuzzy = SequenceMatcher(None, query_text.lower(), name).ratio()
            min_fuzzy = 0.45 if widened else 0.60
            effective_fuzzy = fuzzy * 0.72 if (exact_hint or token_hits or prefix_hits or fuzzy >= min_fuzzy) else 0.0
            token_score = (len(token_hits) + 0.6 * len(prefix_hits)) / max(1, len(set(query_tokens)))
            score = clamp_score(max(effective_fuzzy, token_score, 1.0 if exact_hint else 0.0))
            if score <= 0:
                continue
            matched = tuple(sorted(set(token_hits) | set(prefix_hits)))
            reason = "filename match (widened)" if widened and prefix_hits and not token_hits else "filename match"
            candidates.append(
                SearchCandidate(
                    document=document,
                    score=score,
                    tool=RetrievalTool.FILENAME,
                    matched_terms=matched,
                    reason=reason,
                )
            )

        return self._top(candidates, limit)

    def metadata_search(
        self,
        parsed_query: ParsedQuery,
        limit: int | None = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        candidates: list[SearchCandidate, ...] = []
        query_terms = self._query_terms(parsed_query, expand=widened, widened=widened)
        query_vector = term_counter(query_terms)

        for document in self.corpus.documents:
            extension_match = document.extension in parsed_query.file_types if parsed_query.file_types else False
            date_match = (
                parsed_query.date_window.contains(document.modified_at)
                if parsed_query.date_window is not None
                else False
            )
            metadata_similarity = cosine_similarity(query_vector, self.corpus.metadata_vector(document))

            score = 0.0
            score += 0.45 if extension_match else 0.0
            score += 0.35 if date_match else 0.0
            score += metadata_similarity * (0.45 if widened else 0.35)
            score = clamp_score(score)

            if score <= 0:
                continue
            matched = []
            if extension_match:
                matched.append(document.extension)
            if date_match and parsed_query.date_window:
                matched.append(parsed_query.date_window.label)
            if widened:
                meta_hits = sorted(set(query_terms) & set(self.corpus.metadata_vector(document)))
                matched.extend(term for term in meta_hits if term not in matched)
            candidates.append(
                SearchCandidate(
                    document=document,
                    score=score,
                    tool=RetrievalTool.METADATA,
                    matched_terms=tuple(matched),
                    reason="metadata constraints matched (widened)" if widened else "metadata constraints matched",
                )
            )

        return self._top(candidates, limit)

    def keyword_search(
        self,
        parsed_query: ParsedQuery,
        limit: int | None = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        query_terms = self._query_terms(parsed_query, expand=False, widened=widened)
        query_vector = term_counter(query_terms)
        candidates: list[SearchCandidate] = []

        for document in self.corpus.documents:
            document_vector = self.corpus.content_vector(document)
            similarity = cosine_similarity(query_vector, document_vector)
            matches = sorted(set(query_terms) & set(document_vector))
            if widened and not matches:
                prefix_matches = sorted(
                    {
                        q_tok
                        for q_tok in query_terms
                        if len(q_tok) >= 4
                        and any(d_tok.startswith(q_tok[:4]) or q_tok.startswith(d_tok[:4]) for d_tok in document_vector if len(d_tok) >= 4)
                    }
                )
                if prefix_matches:
                    matches = prefix_matches
                    similarity = max(similarity, 0.28 * (len(prefix_matches) / max(1, len(query_terms))))
            if parsed_query.file_types and document.extension in parsed_query.file_types and (matches or widened):
                similarity += 0.12
            score = clamp_score(similarity)
            if score <= 0:
                continue
            candidates.append(
                SearchCandidate(
                    document=document,
                    score=score,
                    tool=RetrievalTool.KEYWORD,
                    matched_terms=tuple(matches),
                    reason="keyword content match (widened)" if widened else "keyword content match",
                )
            )

        return self._top(candidates, limit)

    def semantic_search(
        self,
        parsed_query: ParsedQuery,
        limit: int | None = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        query_terms = self._query_terms(parsed_query, expand=True, widened=widened)
        query_vector = term_counter(query_terms)
        candidates: list[SearchCandidate] = []

        for document in self.corpus.documents:
            document_vector = self.corpus.content_vector(document)
            if widened:
                document_vector = document_vector + self.corpus.metadata_vector(document)
            similarity = cosine_similarity(query_vector, document_vector)
            matches = sorted(set(query_terms) & set(document_vector))
            score = clamp_score(similarity)
            if score <= 0:
                continue
            candidates.append(
                SearchCandidate(
                    document=document,
                    score=score,
                    tool=RetrievalTool.SEMANTIC,
                    matched_terms=tuple(matches),
                    reason="semantic similarity match (widened lexical expansion)" if widened else "semantic similarity match",
                )
            )

        return self._top(candidates, limit)

    def hybrid_search(
        self,
        parsed_query: ParsedQuery,
        limit: int | None = 10,
        *,
        widened: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        weighted_sources = [
            (0.30, self.filename_search(parsed_query, limit=None, widened=widened)),
            (0.25, self.metadata_search(parsed_query, limit=None, widened=widened)),
            (0.30, self.keyword_search(parsed_query, limit=None, widened=widened)),
            (0.45, self.semantic_search(parsed_query, limit=None, widened=widened)),
        ]
        combined_scores: dict[str, float] = defaultdict(float)
        combined_terms: dict[str, set[str]] = defaultdict(set)
        reasons: dict[str, list[str]] = defaultdict(list)

        for weight, candidates in weighted_sources:
            for candidate in candidates:
                combined_scores[candidate.document.id] += candidate.score * weight
                combined_terms[candidate.document.id].update(candidate.matched_terms)
                reasons[candidate.document.id].append(candidate.tool.value)

        suffix = " (widened)" if widened else ""
        candidates = [
            SearchCandidate(
                document=document,
                score=clamp_score(score),
                tool=RetrievalTool.HYBRID,
                matched_terms=tuple(sorted(combined_terms[document_id])),
                reason=f"hybrid{suffix}: " + ", ".join(sorted(set(reasons[document_id]))),
            )
            for document_id, score in combined_scores.items()
            if (document := self.corpus.get(document_id)) is not None and score > 0
        ]
        return self._top(candidates, limit)

    def _query_terms(self, parsed_query: ParsedQuery, expand: bool, widened: bool = False) -> list[str]:
        terms = [
            *parsed_query.keywords,
            *tokenize(" ".join(parsed_query.filename_hints)),
            *parsed_query.file_types,
        ]
        if not terms or widened:
            for raw_token in tokenize(parsed_query.raw):
                if raw_token not in terms and len(raw_token) > 2:
                    terms.append(raw_token)
        if not expand:
            return terms
        expanded = expand_terms(terms)
        if widened:
            # Second-hop lexical expansion for widened retrieval passes
            expanded = expand_terms(expanded)
        return expanded

    def _top(self, candidates: list[SearchCandidate], limit: int | None) -> tuple[SearchCandidate, ...]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                len(candidate.document.path.parts),
                str(candidate.document.path).lower(),
            ),
        )
        if limit is None:
            return tuple(ordered)
        return tuple(ordered[:limit])
