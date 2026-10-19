"""Downstream modules that run after the agentic router."""

from __future__ import annotations

from scoutx.models import (
    ParsedQuery,
    Recommendation,
    RouterResult,
    SearchCandidate,
    SearchLane,
    UserProfile,
)
from scoutx.text import clamp_score, tokenize


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
    """Generate final natural-language explanations.

    The production architecture can replace this local generator with an LLM
    provider behind the same class boundary.
    """

    def explain(
        self,
        candidates: tuple[SearchCandidate, ...],
        router_result: RouterResult,
        profile: UserProfile,
    ) -> tuple[Recommendation, ...]:
        recommendations: list[Recommendation] = []
        parsed_query = router_result.parsed_query
        final_lane = router_result.trace[-1].strategy.lane if router_result.trace else SearchLane.FAST

        for candidate in candidates:
            fragments = []
            if candidate.matched_terms:
                fragments.append("matches " + ", ".join(f"'{term}'" for term in candidate.matched_terms[:4]))
            if parsed_query.date_window and parsed_query.date_window.contains(candidate.document.modified_at):
                fragments.append(f"was modified during {parsed_query.date_window.label}")
            opens = profile.file_opens.get(candidate.document.id, 0)
            if opens:
                fragments.append(f"you opened it {opens} time{'s' if opens != 1 else ''}")
            if final_lane == SearchLane.DEEP and router_result.forced_exit:
                fragments.append("this is a best-effort match because the router reached its escalation guard")
            if not fragments:
                fragments.append("it is the closest match returned by the selected retrieval strategy")

            explanation = "Recommended because " + "; ".join(fragments) + "."
            recommendations.append(
                Recommendation(
                    document=candidate.document,
                    score=candidate.score,
                    explanation=explanation,
                    source_candidate=candidate,
                )
            )

        return tuple(recommendations)
