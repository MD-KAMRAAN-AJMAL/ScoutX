"""Retrieval toolbox for ScoutX."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from scoutx.document_loader import DocumentLoader
from scoutx.embeddings import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    build_document_embedding_text,
    validate_and_normalize_vector,
)
from scoutx.models import FileDocument, ParsedQuery, RetrievalTool, SearchCandidate
from scoutx.storage import MetadataIndex, VectorIndex, compute_document_content_hash
from scoutx.text import clamp_score, cosine_similarity, expand_terms, term_counter, tokenize

if TYPE_CHECKING:
    from scoutx.embeddings import EmbeddingProvider


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
                    *tokenize(document.path.parent.name if document.path.parent != document.path else ""),
                    document.metadata.get("kind", ""),
                    document.extension,
                ]
            )
            for document in documents
        }

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        loader: DocumentLoader | None = None,
    ) -> "DocumentCorpus":
        doc_loader = loader or DocumentLoader()
        return cls(doc_loader.load_directory(Path(root)))

    def get(self, document_id: str) -> FileDocument | None:
        return self._by_id.get(document_id)

    def content_vector(self, document: FileDocument):
        return self._content_vectors[document.id]

    def metadata_vector(self, document: FileDocument):
        return self._metadata_vectors[document.id]


class RetrievalToolbox:
    def __init__(
        self,
        corpus: DocumentCorpus,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        metadata_index: MetadataIndex | None = None,
        strict_embeddings: bool = False,
    ):
        self.corpus = corpus
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.metadata_index = metadata_index
        self.strict_embeddings = bool(strict_embeddings)
        self.last_semantic_fallback_error: str | None = None
        self._in_memory_embeddings: dict[str, tuple[str, str, tuple[float, ...]]] = {}

        if self.metadata_index is not None:
            self.metadata_index.sync_documents(self.corpus.documents, prune_missing=True)

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
        base_query_tokens = self._query_terms(parsed_query, expand=False, widened=False)
        query_tokens = self._query_terms(parsed_query, expand=widened, widened=widened)
        candidates: list[SearchCandidate] = []

        for document in self.corpus.documents:
            name = document.path.name.lower()
            stem_name = document.path.stem.lower()
            name_tokens = tokenize(name)
            token_hits = sorted(set(query_tokens) & set(name_tokens))
            prefix_hits: list[str] = []
            if widened:
                prefix_hits = sorted(
                    {
                        q_tok
                        for q_tok in query_tokens
                        if len(q_tok) >= 4
                        and any(
                            n_tok.startswith(q_tok[:4]) or q_tok.startswith(n_tok[:4])
                            for n_tok in name_tokens
                            if len(n_tok) >= 4
                        )
                    }
                    - set(token_hits)
                )
            exact_hint = any(hint.lower() in name for hint in parsed_query.filename_hints)
            clean_q = query_text.strip().lower()
            exact_name_match = bool(clean_q and (clean_q == name or clean_q == stem_name))
            exact_synonym_match = bool(widened and (stem_name in query_tokens or name in query_tokens))
            word_boundary_match = bool(
                clean_q and len(clean_q) >= 3 and any(clean_q == n_tok for n_tok in name_tokens)
            ) or bool(
                widened and any(tok in name_tokens for tok in query_tokens if len(tok) >= 3)
            )

            fuzzy = SequenceMatcher(None, query_text.lower(), name).ratio()
            # Require actual lexical grounding (exact match, token hit, prefix hit, or hint)
            # or very high fuzzy similarity (>= 0.75 for unwidened, >= 0.65 for widened)
            # rather than admitting unrelated files at 0.45 ratio.
            min_fuzzy = 0.65 if widened else 0.75
            has_grounding = exact_name_match or exact_hint or exact_synonym_match or token_hits or prefix_hits
            effective_fuzzy = fuzzy * 0.72 if (has_grounding or fuzzy >= min_fuzzy) else 0.0

            concept_count = max(1, len(set(base_query_tokens)))
            token_score = min(1.0, (len(token_hits) + 0.6 * len(prefix_hits)) / concept_count)
            if exact_name_match:
                score = 1.0
            elif exact_hint:
                score = 1.0
            elif exact_synonym_match:
                # Exact basename or stem match on an expanded synonym (e.g. Photos folder for 'images')
                score = clamp_score(max(0.85, effective_fuzzy, token_score))
            elif word_boundary_match and token_score >= 0.5:
                # Strong exact word match in basename (direct or expanded synonym)
                score = clamp_score(
                    max(
                        0.80 if widened and not any(clean_q == n_tok for n_tok in name_tokens) else 0.85,
                        effective_fuzzy,
                        token_score,
                    )
                )
            else:
                score = clamp_score(max(effective_fuzzy, token_score))

            if score <= 0:
                continue

            matched_terms_set = set(token_hits) | set(prefix_hits)
            if exact_name_match and clean_q:
                matched_terms_set.add(clean_q)
            matched = tuple(sorted(matched_terms_set))
            reason = (
                "filename match (widened)"
                if widened and (prefix_hits or exact_synonym_match) and not exact_name_match
                else "filename match"
            )
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
        candidates: list[SearchCandidate] = []
        query_terms = self._query_terms(parsed_query, expand=widened, widened=widened)
        query_vector = term_counter(query_terms)

        for document in self.corpus.documents:
            extension_match = (
                document.extension in parsed_query.file_types if parsed_query.file_types else False
            )
            date_match = (
                parsed_query.date_window.contains(document.modified_at)
                if parsed_query.date_window is not None
                else False
            )
            metadata_similarity = cosine_similarity(
                query_vector, self.corpus.metadata_vector(document)
            )

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
                    reason=(
                        "metadata constraints matched (widened)"
                        if widened
                        else "metadata constraints matched"
                    ),
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
        query_terms = self._query_terms(parsed_query, expand=widened, widened=widened)
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
                        and any(
                            d_tok.startswith(q_tok[:4]) or q_tok.startswith(d_tok[:4])
                            for d_tok in document_vector
                            if len(d_tok) >= 4
                        )
                    }
                )
                if prefix_matches:
                    matches = prefix_matches
                    similarity = max(
                        similarity, 0.28 * (len(prefix_matches) / max(1, len(query_terms)))
                    )
            if (
                parsed_query.file_types
                and document.extension in parsed_query.file_types
                and (matches or widened)
            ):
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
        if self.embedding_provider is None:
            return self._lexical_semantic_search(parsed_query, limit, widened=widened)

        try:
            dense_scores = self._compute_dense_scores(parsed_query, widened=widened)
        except (EmbeddingError, sqlite3.Error, ValueError) as exc:
            self.last_semantic_fallback_error = str(exc)
            if self.strict_embeddings:
                raise
            return self._lexical_semantic_search(parsed_query, limit, widened=widened)

        self.last_semantic_fallback_error = None
        query_terms = self._query_terms(parsed_query, expand=True, widened=widened)
        query_vector = term_counter(query_terms)
        candidates: list[SearchCandidate] = []

        for document in self.corpus.documents:
            document_vector = self.corpus.content_vector(document)
            if widened:
                document_vector = document_vector + self.corpus.metadata_vector(document)
            lexical_sim = cosine_similarity(query_vector, document_vector)
            matches = sorted(set(query_terms) & set(document_vector))
            dense_sim = dense_scores.get(document.id, 0.0)

            if dense_sim > 0.0 and lexical_sim > 0.0:
                score = clamp_score(max(lexical_sim, dense_sim * 0.85 + lexical_sim * 0.15))
                reason = (
                    "dense semantic similarity match (widened)"
                    if widened
                    else "dense semantic similarity match"
                )
            elif dense_sim > 0.0:
                score = clamp_score(dense_sim)
                reason = (
                    "dense semantic similarity match (widened)"
                    if widened
                    else "dense semantic similarity match"
                )
            elif lexical_sim > 0.0:
                score = clamp_score(lexical_sim)
                reason = (
                    "semantic similarity match (widened lexical expansion)"
                    if widened
                    else "semantic similarity match"
                )
            else:
                continue

            if score <= 0:
                continue
            candidates.append(
                SearchCandidate(
                    document=document,
                    score=score,
                    tool=RetrievalTool.SEMANTIC,
                    matched_terms=tuple(matches),
                    reason=reason,
                )
            )

        return self._top(candidates, limit)

    def _lexical_semantic_search(
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
                    reason=(
                        "semantic similarity match (widened lexical expansion)"
                        if widened
                        else "semantic similarity match"
                    ),
                )
            )

        return self._top(candidates, limit)

    def _compute_dense_scores(
        self,
        parsed_query: ParsedQuery,
        *,
        widened: bool = False,
    ) -> dict[str, float]:
        assert self.embedding_provider is not None
        provider = self.embedding_provider
        provider_config = getattr(provider, "config", None)
        max_chars = getattr(provider_config, "max_input_chars", 4000)

        query_terms = self._query_terms(parsed_query, expand=widened, widened=widened)
        raw_query_text = " ".join(query_terms).strip() or parsed_query.raw.strip()
        if not raw_query_text or not tokenize(raw_query_text):
            return {}
        bounded_query_text = raw_query_text[:max_chars]

        if not self.corpus.documents:
            return {}

        if self.vector_index is not None:
            self.vector_index.index_documents(
                self.corpus.documents,
                provider,
                prune_missing=True,
                max_input_chars=max_chars,
            )
            query_vec = provider.embed_query(bounded_query_text)
            live_ids = {doc.id for doc in self.corpus.documents}
            ranked_pairs = self.vector_index.search(
                query_vec,
                model_id=provider.model_id,
                limit=None,
                candidate_ids=live_ids,
            )
            return {
                doc_id: score
                for doc_id, score in ranked_pairs
                if self.corpus.get(doc_id) is not None
            }

        # In-memory dense vector fallback when no persistent VectorIndex is attached
        model_id = provider.model_id
        expected_dim = getattr(provider, "dimensions", None)
        to_embed: list[tuple[FileDocument, str, str]] = []
        live_ids = {doc.id for doc in self.corpus.documents}

        # Prune deleted documents from in-memory cache
        for stale_id in list(self._in_memory_embeddings):
            if stale_id not in live_ids:
                del self._in_memory_embeddings[stale_id]

        for doc in self.corpus.documents:
            c_hash = compute_document_content_hash(doc)
            cached = self._in_memory_embeddings.get(doc.id)
            if cached is not None:
                cached_hash, cached_model, cached_vec = cached
                if cached_model != model_id:
                    raise EmbeddingDimensionMismatchError(
                        f"In-memory vector cache bound to '{cached_model}', cannot mix with '{model_id}'"
                    )
                if expected_dim is None:
                    expected_dim = len(cached_vec)
                if cached_hash == c_hash:
                    continue
            to_embed.append(
                (doc, c_hash, build_document_embedding_text(doc, max_input_chars=max_chars))
            )

        if to_embed:
            raw_vectors = provider.embed_texts([item[2] for item in to_embed])
            if len(raw_vectors) != len(to_embed):
                raise EmbeddingError(
                    f"Provider returned {len(raw_vectors)} vectors for {len(to_embed)} documents"
                )
            validated_batch: list[tuple[str, str, tuple[float, ...]]] = []
            for (doc, c_hash, _), raw_vec in zip(to_embed, raw_vectors):
                norm_vec = validate_and_normalize_vector(
                    raw_vec,
                    expected_dimensions=expected_dim,
                    normalize=True,
                )
                if expected_dim is None:
                    expected_dim = len(norm_vec)
                validated_batch.append((doc.id, c_hash, norm_vec))
            for doc_id, c_hash, norm_vec in validated_batch:
                self._in_memory_embeddings[doc_id] = (c_hash, model_id, norm_vec)

        query_vec = validate_and_normalize_vector(
            provider.embed_query(bounded_query_text),
            expected_dimensions=expected_dim,
            normalize=True,
        )
        scores: dict[str, float] = {}
        for doc in self.corpus.documents:
            entry = self._in_memory_embeddings.get(doc.id)
            if entry is None:
                continue
            _, _, doc_vec = entry
            dot = sum(q_val * d_val for q_val, d_val in zip(query_vec, doc_vec))
            if dot > 0.0:
                scores[doc.id] = clamp_score(dot)
        return scores

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

    def _query_terms(
        self, parsed_query: ParsedQuery, expand: bool, widened: bool = False
    ) -> list[str]:
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

    def _top(
        self, candidates: list[SearchCandidate], limit: int | None
    ) -> tuple[SearchCandidate, ...]:
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
