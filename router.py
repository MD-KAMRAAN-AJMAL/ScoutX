"""Agentic complexity and routing loop."""

from __future__ import annotations

from collections import defaultdict

from scoutx.models import (
    ParsedQuery,
    Reflection,
    RetrievalTool,
    RouterResult,
    RouterTraceStep,
    SearchCandidate,
    SearchLane,
    SearchStrategy,
)
from scoutx.retrieval import RetrievalToolbox
from scoutx.text import clamp_score


class ComplexityRoutingAgent:
    """Score, plan, act, observe, and reflect over retrieval results."""

    def __init__(
        self,
        toolbox: RetrievalToolbox,
        max_escalations: int = 2,
        search_limit: int = 10,
    ):
        self.toolbox = toolbox
        self.max_escalations = max_escalations
        self.search_limit = search_limit

    def run(self, parsed_query: ParsedQuery) -> RouterResult:
        complexity_score = self.score_complexity(parsed_query)
        trace: list[RouterTraceStep] = []
        escalation_level = 0
        candidates: tuple[SearchCandidate, ...] = ()
        forced_exit = False

        while True:
            force_guard = escalation_level >= self.max_escalations
            strategy = self.select_strategy(
                complexity_score,
                escalation_level,
                force_deep=force_guard,
                parsed_query=parsed_query,
            )
            candidates = self.act(strategy, parsed_query)
            reflection = self.reflect(parsed_query, strategy, candidates)

            if force_guard and not reflection.confident:
                forced_exit = True
                reflection = Reflection(
                    confident=True,
                    coverage=reflection.coverage,
                    top_score=reflection.top_score,
                    relevance_to_intent=reflection.relevance_to_intent,
                    reason="iteration guard forced Deep Lane exit after two escalations",
                )

            trace.append(
                RouterTraceStep(
                    iteration=escalation_level,
                    complexity_score=complexity_score,
                    strategy=strategy,
                    candidates_seen=len(candidates),
                    reflection=reflection,
                )
            )

            if reflection.confident:
                return RouterResult(
                    parsed_query=parsed_query,
                    candidates=candidates,
                    trace=tuple(trace),
                    forced_exit=forced_exit,
                )

            escalation_level += 1

    def score_complexity(self, parsed_query: ParsedQuery) -> float:
        constraints_factor = min(parsed_query.constraints_count / 6, 1.0)
        score = 0.15
        score += parsed_query.ambiguity_score * 0.65
        score += constraints_factor * 0.45
        score += (1.0 - parsed_query.parser_confidence) * 0.10
        if parsed_query.filename_hints:
            score -= 0.35
        return clamp_score(score)

    def select_strategy(
        self,
        complexity_score: float,
        escalation_level: int = 0,
        force_deep: bool = False,
        parsed_query: ParsedQuery | None = None,
    ) -> SearchStrategy:
        if force_deep:
            return SearchStrategy(
                lane=SearchLane.DEEP,
                tools=(RetrievalTool.HYBRID,),
                requires_reranking=True,
                uses_llm_reasoning=True,
            )

        if complexity_score < 0.35:
            base_lane = SearchLane.FAST
        elif complexity_score < 0.70:
            base_lane = SearchLane.STANDARD
        else:
            base_lane = SearchLane.DEEP

        lane_order = (SearchLane.FAST, SearchLane.STANDARD, SearchLane.DEEP)
        lane_index = min(lane_order.index(base_lane) + escalation_level, len(lane_order) - 1)
        lane = lane_order[lane_index]

        if lane == SearchLane.FAST:
            return SearchStrategy(
                lane=lane,
                tools=(RetrievalTool.FILENAME, RetrievalTool.METADATA),
            )
        if lane == SearchLane.STANDARD:
            tools = [RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC]
            if parsed_query and (parsed_query.file_types or parsed_query.date_window):
                tools.insert(0, RetrievalTool.METADATA)
            if parsed_query and (
                parsed_query.filename_hints
                or "folder" in parsed_query.file_types
                or any(keyword.isdigit() for keyword in parsed_query.keywords)
            ):
                tools.insert(0, RetrievalTool.FILENAME)
            return SearchStrategy(
                lane=lane,
                tools=tuple(dict.fromkeys(tools)),
            )
        return SearchStrategy(
            lane=lane,
            tools=(RetrievalTool.HYBRID,),
            requires_reranking=True,
            uses_llm_reasoning=escalation_level > 0,
        )

    def act(self, strategy: SearchStrategy, parsed_query: ParsedQuery) -> tuple[SearchCandidate, ...]:
        by_document: dict[str, list[SearchCandidate]] = defaultdict(list)
        for tool in strategy.tools:
            for candidate in self.toolbox.search(tool, parsed_query, limit=self.search_limit):
                by_document[candidate.document.id].append(candidate)

        merged: list[SearchCandidate] = []
        for document_id, candidates in by_document.items():
            best = max(candidates, key=lambda candidate: candidate.score)
            score = best.score + min(0.15, 0.05 * (len(candidates) - 1))
            matched_terms = sorted({term for candidate in candidates for term in candidate.matched_terms})
            used_tools = ", ".join(sorted({candidate.tool.value for candidate in candidates}))
            merged.append(
                SearchCandidate(
                    document=best.document,
                    score=clamp_score(score),
                    tool=best.tool,
                    matched_terms=tuple(matched_terms),
                    reason=f"router merged evidence from {used_tools}",
                )
            )

        return tuple(
            sorted(
                merged,
                key=lambda candidate: (
                    -candidate.score,
                    len(candidate.document.path.parts),
                    str(candidate.document.path).lower(),
                ),
            )
        )

    def reflect(
        self,
        parsed_query: ParsedQuery,
        strategy: SearchStrategy,
        candidates: tuple[SearchCandidate, ...],
    ) -> Reflection:
        coverage = len(candidates)
        top_score = candidates[0].score if candidates else 0.0
        relevance_to_intent = clamp_score(top_score * (1.0 - parsed_query.ambiguity_score * 0.35))

        required_top_score = {
            SearchLane.FAST: 0.70,
            SearchLane.STANDARD: 0.32,
            SearchLane.DEEP: 0.25,
        }[strategy.lane]
        required_relevance = {
            SearchLane.FAST: 0.55,
            SearchLane.STANDARD: 0.25,
            SearchLane.DEEP: 0.18,
        }[strategy.lane]

        confident = coverage > 0 and top_score >= required_top_score and relevance_to_intent >= required_relevance
        if confident:
            reason = f"{strategy.lane.value} lane produced enough coverage and relevance"
        elif coverage == 0:
            reason = "no candidates returned"
        elif top_score < required_top_score:
            reason = f"top score {top_score:.2f} below {strategy.lane.value} threshold"
        else:
            reason = "result relevance did not satisfy parsed intent"

        return Reflection(
            confident=confident,
            coverage=coverage,
            top_score=top_score,
            relevance_to_intent=relevance_to_intent,
            reason=reason,
        )
