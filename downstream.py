"""Downstream modules that run after the agentic router."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scoutx.llm import (
    ALLOWED_EVIDENCE_FIELDS,
    GroundedExplanationItem,
    LLMError,
    LLMGenerationResult,
    LLMResponseFormatError,
    LLMUnavailableError,
    available_evidence_fields,
    sanitize_candidate_metadata,
)
from scoutx.models import (
    ParsedQuery,
    Recommendation,
    RouterResult,
    SearchCandidate,
    SearchLane,
    UserProfile,
)
from scoutx.text import clamp_score, tokenize

if TYPE_CHECKING:
    from scoutx.llm import LLMProvider

DEGRADED_FALLBACK_PREFIX = "[LLM unavailable — deterministic fallback] "
FORCED_EXIT_SUBSTRING = "best-effort match because the router reached its escalation guard"
FORCED_EXIT_CLAUSE = "this is a best-effort match because the router reached its escalation guard"


class RerankingModule:
    def rerank(
        self,
        candidates: tuple[SearchCandidate, ...],
        parsed_query: ParsedQuery,
    ) -> tuple[SearchCandidate, ...]:
        if not candidates:
            return candidates

        query_terms = set(parsed_query.keywords)
        reranked: list[SearchCandidate] = []
        for candidate in candidates:
            title_terms = set(tokenize(candidate.document.title))
            content_terms = set(tokenize(candidate.document.content))
            term_overlap = len(query_terms & (title_terms | content_terms)) / max(1, len(query_terms))
            score = clamp_score(candidate.score * 0.85 + term_overlap * 0.15)
            reranked.append(
                SearchCandidate(
                    document=candidate.document,
                    score=score,
                    tool=candidate.tool,
                    matched_terms=candidate.matched_terms,
                    reason=f"{candidate.reason}; reranked",
                )
            )
        return tuple(
            sorted(
                reranked,
                key=lambda candidate: (
                    -candidate.score,
                    len(candidate.document.path.parts),
                    str(candidate.document.path).lower(),
                ),
            )
        )


class ContextMemoryModule:
    def __init__(self, profile: UserProfile | None = None):
        self.profile = profile or UserProfile()

    def load_context(self) -> UserProfile:
        return self.profile


class PersonalizationModule:
    def __init__(self, relevance_floor: float = 0.25, tie_band: float = 0.15):
        self.relevance_floor = relevance_floor
        self.tie_band = tie_band

    def personalize(
        self,
        candidates: tuple[SearchCandidate, ...],
        parsed_query: ParsedQuery,
        profile: UserProfile,
    ) -> tuple[SearchCandidate, ...]:
        if not candidates:
            return candidates

        top_score = max(candidate.score for candidate in candidates)
        max_opens = max(profile.file_opens.values(), default=0)
        max_extension = max(profile.preferred_extensions.values(), default=0)
        max_topic = max(profile.topic_interests.values(), default=0)
        personalized: list[SearchCandidate] = []

        for candidate in candidates:
            score = candidate.score
            can_nudge = score >= self.relevance_floor and top_score - score <= self.tie_band
            if can_nudge:
                score += self._profile_boost(candidate, parsed_query, profile, max_opens, max_extension, max_topic)

            personalized.append(
                SearchCandidate(
                    document=candidate.document,
                    score=clamp_score(score),
                    tool=candidate.tool,
                    matched_terms=candidate.matched_terms,
                    reason=candidate.reason,
                )
            )

        return tuple(
            sorted(
                personalized,
                key=lambda candidate: (
                    -candidate.score,
                    len(candidate.document.path.parts),
                    str(candidate.document.path).lower(),
                ),
            )
        )

    def _profile_boost(
        self,
        candidate: SearchCandidate,
        parsed_query: ParsedQuery,
        profile: UserProfile,
        max_opens: int,
        max_extension: int,
        max_topic: int,
    ) -> float:
        boost = 0.0
        if max_opens:
            boost += 0.08 * (profile.file_opens.get(candidate.document.id, 0) / max_opens)
        if max_extension:
            boost += 0.04 * (profile.preferred_extensions.get(candidate.document.extension, 0) / max_extension)
        if max_topic:
            topic_hits = sum(profile.topic_interests.get(keyword, 0) for keyword in parsed_query.keywords)
            boost += 0.04 * min(1.0, topic_hits / max_topic)
        return boost


class ExplanationModule:
    """Generate final natural-language explanations with grounded LLM validation and truthful provenance."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        *,
        require_llm: bool | None = None,
        allow_degraded_fallback: bool = False,
    ) -> None:
        self.llm_provider = llm_provider
        self.require_llm = bool(llm_provider is not None) if require_llm is None else bool(require_llm)
        self.allow_degraded_fallback = bool(allow_degraded_fallback)

    def explain(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
    ) -> tuple[Recommendation, ...]:
        if not candidates:
            return ()

        final_lane = router_result.trace[-1].strategy.lane if router_result.trace else SearchLane.FAST
        is_forced_deep_exit = bool(final_lane == SearchLane.DEEP and router_result.forced_exit)

        if self.llm_provider is None:
            if not self.require_llm:
                return self._deterministic_recommendations(
                    candidates,
                    router_result,
                    profile,
                    explanation_source="deterministic",
                    prefix="",
                )
            if self.allow_degraded_fallback:
                return self._deterministic_recommendations(
                    candidates,
                    router_result,
                    profile,
                    explanation_source="degraded_fallback",
                    prefix=DEGRADED_FALLBACK_PREFIX,
                )
            raise LLMUnavailableError("LLM provider is required for ExplanationModule, but no provider was configured")

        try:
            gen_result = self.llm_provider.generate_explanations(candidates, router_result, profile)
            validated_items = self._validate_grounded_explanations(
                candidates,
                router_result,
                profile,
                gen_result,
            )
            if is_forced_deep_exit:
                # For forced low-confidence exits, arbitrary free-form LLM prose cannot be
                # mechanically proven free of paraphrased confidence claims. After executing
                # and validating the required LLM call (so cited_evidence is verified),
                # construct the user-facing text deterministically from the validated
                # cited_evidence fields and record truthful provenance ("deterministic", model=None).
                return self._forced_exit_recommendations_from_cited_evidence(
                    candidates,
                    router_result,
                    profile,
                    validated_items,
                )

            recommendations: list[Recommendation] = []
            for candidate, item in zip(candidates, validated_items):
                recommendations.append(
                    Recommendation(
                        document=candidate.document,
                        score=candidate.score,
                        explanation=item.explanation,
                        source_candidate=candidate,
                        explanation_source="llm",
                        explanation_model=gen_result.model_id,
                        cited_evidence=item.cited_evidence,
                    )
                )
            return tuple(recommendations)
        except Exception as exc:
            if not self.allow_degraded_fallback:
                if isinstance(exc, LLMError):
                    raise
                raise LLMUnavailableError(f"LLM explanation generation failed: {type(exc).__name__}: {exc}") from exc

            return self._deterministic_recommendations(
                candidates,
                router_result,
                profile,
                explanation_source="degraded_fallback",
                prefix=DEGRADED_FALLBACK_PREFIX,
            )

    def _forced_exit_recommendations_from_cited_evidence(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
        validated_items: tuple[GroundedExplanationItem, ...],
    ) -> tuple[Recommendation, ...]:
        """Build truthful, deterministic forced-exit explanations from validated cited_evidence fields."""
        parsed_query = router_result.parsed_query
        recommendations: list[Recommendation] = []

        for candidate, item in zip(candidates, validated_items):
            doc = candidate.document
            fragments: list[str] = []
            for field_name in item.cited_evidence:
                if field_name == "title":
                    fragments.append(f"titled '{doc.title}'")
                elif field_name == "matched_terms" and candidate.matched_terms:
                    fragments.append("matches " + ", ".join(f"'{term}'" for term in candidate.matched_terms[:4]))
                elif field_name == "file_type" and doc.extension:
                    fragments.append(f"has file type '{doc.extension.lstrip('.')}'")
                elif field_name == "modified_date" and doc.modified_at is not None:
                    if parsed_query.date_window and parsed_query.date_window.contains(doc.modified_at):
                        fragments.append(f"was modified during {parsed_query.date_window.label}")
                    else:
                        fragments.append(f"was modified on {doc.modified_at.date().isoformat()}")
                elif field_name == "open_count":
                    opens = profile.file_opens.get(doc.id, 0)
                    if opens > 0:
                        fragments.append(f"you opened it {opens} time{'s' if opens != 1 else ''}")
                elif field_name == "retrieval_tool":
                    fragments.append(f"surfaced by {candidate.tool.value} retrieval")
                elif field_name == "snippet" and doc.content:
                    fragments.append("contains matching snippet content")
                elif field_name == "metadata":
                    meta = sanitize_candidate_metadata(doc.metadata)
                    if meta:
                        first_key, first_val = next(iter(meta.items()))
                        fragments.append(f"has {first_key} '{first_val}'")

            fragments.append(FORCED_EXIT_CLAUSE)
            explanation = "Recommended because " + "; ".join(fragments) + "."
            recommendations.append(
                Recommendation(
                    document=doc,
                    score=candidate.score,
                    explanation=explanation,
                    source_candidate=candidate,
                    explanation_source="deterministic",
                    explanation_model=None,
                    cited_evidence=item.cited_evidence,
                )
            )

        return tuple(recommendations)

    def _validate_grounded_explanations(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
        gen_result: object,
    ) -> tuple[GroundedExplanationItem, ...]:
        """Validate candidate identity bijection, evidence field availability, and lexical anchoring."""
        if not isinstance(gen_result, LLMGenerationResult):
            raise LLMResponseFormatError(
                f"Expected LLMGenerationResult from provider, got {type(gen_result).__name__}"
            )
        if not isinstance(gen_result.model_id, str) or not gen_result.model_id.strip():
            raise LLMResponseFormatError("LLMGenerationResult must carry a non-empty model_id")

        items = gen_result.explanations
        if not isinstance(items, tuple) or len(items) != len(candidates):
            raise LLMResponseFormatError(
                f"Explanation count mismatch: expected {len(candidates)}, got {len(items) if isinstance(items, tuple) else 'non-tuple'}"
            )

        parsed_query = router_result.parsed_query
        validated: list[GroundedExplanationItem] = []

        for idx, (candidate, item) in enumerate(zip(candidates, items)):
            if not isinstance(item, GroundedExplanationItem):
                raise LLMResponseFormatError(
                    f"Expected GroundedExplanationItem at index {idx}, got {type(item).__name__}"
                )
            if item.document_id != candidate.document.id:
                raise LLMResponseFormatError(
                    f"Explanation candidate ID mismatch at index {idx}: "
                    f"expected {candidate.document.id!r}, got {item.document_id!r}"
                )

            if not isinstance(item.explanation, str):
                raise LLMResponseFormatError(f"Explanation for {item.document_id!r} must be a string")
            normalized_text = " ".join(item.explanation.split()).strip()
            if not (10 <= len(normalized_text) <= 400):
                raise LLMResponseFormatError(
                    f"Explanation length for {item.document_id!r} must be in [10, 400] chars, got {len(normalized_text)}"
                )
            if "```" in normalized_text or normalized_text.startswith("{"):
                raise LLMResponseFormatError(
                    f"Explanation for {item.document_id!r} must be natural-language prose, not raw JSON/code"
                )

            if not isinstance(item.cited_evidence, tuple) or not item.cited_evidence:
                raise LLMResponseFormatError(
                    f"Explanation for {item.document_id!r} must cite at least one evidence field"
                )
            if len(set(item.cited_evidence)) != len(item.cited_evidence):
                raise LLMResponseFormatError(
                    f"Explanation for {item.document_id!r} contains duplicate cited_evidence entries: {item.cited_evidence!r}"
                )

            available_fields = available_evidence_fields(candidate, profile)
            for field_name in item.cited_evidence:
                if not isinstance(field_name, str) or field_name not in ALLOWED_EVIDENCE_FIELDS:
                    raise LLMResponseFormatError(
                        f"Explanation for {item.document_id!r} cited unsupported evidence field {field_name!r}"
                    )
                if field_name not in available_fields:
                    raise LLMResponseFormatError(
                        f"Explanation for {item.document_id!r} cited unavailable evidence field {field_name!r} "
                        f"(available: {sorted(available_fields)})"
                    )

            anchor_tokens = self._evidence_anchor_tokens(candidate, parsed_query, profile, item.cited_evidence)
            explanation_tokens = set(tokenize(normalized_text))
            explanation_lower = normalized_text.lower()
            has_anchor = bool(anchor_tokens & explanation_tokens) or any(
                anchor in explanation_lower for anchor in anchor_tokens if len(anchor) >= 2
            )
            if not has_anchor:
                raise LLMResponseFormatError(
                    f"Explanation for {item.document_id!r} has no lexical overlap with cited evidence fields "
                    f"{item.cited_evidence!r}"
                )

            validated.append(
                GroundedExplanationItem(
                    document_id=item.document_id,
                    explanation=normalized_text,
                    cited_evidence=item.cited_evidence,
                )
            )

        return tuple(validated)

    def _evidence_anchor_tokens(
        self,
        candidate: SearchCandidate,
        parsed_query: ParsedQuery,
        profile: UserProfile,
        cited_evidence: tuple[str, ...],
    ) -> set[str]:
        """Extract concrete lexical tokens from the candidate's cited evidence fields."""
        doc = candidate.document
        anchors: set[str] = set()
        for field_name in cited_evidence:
            if field_name == "title":
                anchors.update(tokenize(doc.title))
            elif field_name == "matched_terms":
                for term in candidate.matched_terms:
                    anchors.update(tokenize(term))
                    anchors.add(term.lower())
            elif field_name == "file_type" and doc.extension:
                ext = doc.extension.lstrip(".").lower()
                anchors.add(ext)
                if ext in {"xls", "xlsx", "csv", "tsv"}:
                    anchors.update({"spreadsheet", "sheet", "csv", "excel"})
                elif ext in {"ppt", "pptx", "key"}:
                    anchors.update({"presentation", "deck", "slide", "slides"})
                elif ext in {"doc", "docx", "pdf", "txt", "md"}:
                    anchors.update({"document", "doc", "pdf", "markdown", "note", "notes"})
            elif field_name == "modified_date" and doc.modified_at is not None:
                anchors.update(tokenize(doc.modified_at.date().isoformat()))
                anchors.update({"modified", "updated", "recent", "date"})
                if parsed_query.date_window and parsed_query.date_window.label:
                    anchors.update(tokenize(parsed_query.date_window.label))
            elif field_name == "open_count":
                opens = profile.file_opens.get(doc.id, 0)
                if opens > 0:
                    anchors.update({str(opens), "opened", "open", "times", "time"})
            elif field_name == "retrieval_tool":
                anchors.update(tokenize(candidate.tool.value))
                anchors.update(tokenize(candidate.reason))
            elif field_name == "snippet" and doc.content:
                anchors.update(tokenize(doc.content[:240]))
            elif field_name == "metadata":
                meta = sanitize_candidate_metadata(doc.metadata)
                for k, v in meta.items():
                    anchors.update(tokenize(k))
                    anchors.update(tokenize(v))
        return {tok for tok in anchors if tok}

    def _deterministic_recommendations(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
        *,
        explanation_source: str,
        prefix: str,
    ) -> tuple[Recommendation, ...]:
        recommendations: list[Recommendation] = []
        parsed_query = router_result.parsed_query
        final_lane = router_result.trace[-1].strategy.lane if router_result.trace else SearchLane.FAST

        for candidate in candidates:
            fragments: list[str] = []
            cited: list[str] = []
            if candidate.matched_terms:
                fragments.append("matches " + ", ".join(f"'{term}'" for term in candidate.matched_terms[:4]))
                cited.append("matched_terms")
            if parsed_query.date_window and parsed_query.date_window.contains(candidate.document.modified_at):
                fragments.append(f"was modified during {parsed_query.date_window.label}")
                cited.append("modified_date")
            opens = profile.file_opens.get(candidate.document.id, 0)
            if opens:
                fragments.append(f"you opened it {opens} time{'s' if opens != 1 else ''}")
                cited.append("open_count")
            if final_lane == SearchLane.DEEP and router_result.forced_exit:
                fragments.append(FORCED_EXIT_CLAUSE)
            if not fragments:
                fragments.append("it is the closest match returned by the selected retrieval strategy")
                cited.append("retrieval_tool")

            explanation = f"{prefix}Recommended because " + "; ".join(fragments) + "."
            recommendations.append(
                Recommendation(
                    document=candidate.document,
                    score=candidate.score,
                    explanation=explanation,
                    source_candidate=candidate,
                    explanation_source=explanation_source,
                    explanation_model=None,
                    cited_evidence=tuple(cited),
                )
            )

        return tuple(recommendations)
