"""Agentic complexity and routing loop."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence, runtime_checkable

from scoutx.models import (
    FileDocument,
    ParsedQuery,
    ReasoningStatus,
    Reflection,
    RetrievalTool,
    RouterResult,
    RouterThresholds,
    RouterTraceStep,
    SearchCandidate,
    SearchLane,
    SearchStrategy,
    ToolExecutionOutcome,
    TraceCandidateEvidence,
)
from scoutx.retrieval import RetrievalToolbox
from scoutx.text import clamp_score, expand_terms, tokenize

if TYPE_CHECKING:
    from scoutx.laya import LayaAdvisor

VAGUE_QUERY_TERMS = frozenset(
    {"thing", "stuff", "something", "numbers", "maybe", "whatever", "while", "ago"}
)


@runtime_checkable
class CandidateReasoner(Protocol):
    """Bounded single-turn candidate reasoning interface for Deep Lane passes."""

    def reason_candidates(
        self,
        parsed_query: ParsedQuery,
        candidates: tuple[SearchCandidate, ...],
        *,
        widened: bool = False,
        forced_guard: bool = False,
    ) -> tuple[SearchCandidate, ...]: ...


class ComplexityRoutingAgent:
    """Score, plan, act, observe, and reflect over retrieval results.

    This class is the sole autonomous decision-maker in ScoutX.
    """

    def __init__(
        self,
        toolbox: RetrievalToolbox,
        max_escalations: int = 2,
        search_limit: int = 10,
        *,
        thresholds: RouterThresholds | None = None,
        history_difficulty_fn: Callable[[ParsedQuery], float] | None = None,
        reasoner: CandidateReasoner | None = None,
        laya_advisor: LayaAdvisor | None = None,
    ):
        if isinstance(max_escalations, bool) or not isinstance(max_escalations, int) or max_escalations < 0:
            raise ValueError(f"max_escalations must be an integer >= 0, got {max_escalations!r}")
        if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit < 1:
            raise ValueError(f"search_limit must be an integer >= 1, got {search_limit!r}")

        self.toolbox = toolbox
        self.max_escalations = max_escalations
        self.search_limit = search_limit
        self.thresholds = thresholds or RouterThresholds()
        self.history_difficulty_fn = history_difficulty_fn
        self.reasoner = reasoner
        self.laya_advisor = laya_advisor

    def run(self, parsed_query: ParsedQuery) -> RouterResult:
        complexity_score = self.score_complexity(parsed_query)
        trace: list[RouterTraceStep] = []
        escalation_level = 0
        candidates: tuple[SearchCandidate, ...] = ()

        laya_suggestion = None
        if self.laya_advisor is not None:
            try:
                laya_suggestion = self.laya_advisor.advise_search(parsed_query, complexity_score)
            except Exception as exc:
                from scoutx.models import LayaAdvisorSuggestion
                laya_suggestion = LayaAdvisorSuggestion(
                    status="unavailable",
                    error_message=f"Laya advisor exception: {exc}",
                )

        while True:
            reached_escalation_cap = escalation_level >= self.max_escalations
            force_guard = self.max_escalations > 0 and reached_escalation_cap
            strategy = self.select_strategy(
                complexity_score,
                escalation_level,
                force_deep=force_guard,
                parsed_query=parsed_query,
                prior_trace=tuple(trace),
            )
            tool_outcomes = self.act(strategy, parsed_query)
            candidates, reasoning_status = self._observe_with_status(
                tool_outcomes,
                strategy,
                parsed_query,
                force_guard=force_guard,
            )
            step_evidence = self._build_step_evidence(candidates, tool_outcomes)
            tool_errors = {
                outcome.tool: outcome.error
                for outcome in tool_outcomes.values()
                if outcome.error is not None
            }
            reflection = self.reflect(
                parsed_query,
                strategy,
                candidates,
                tool_errors=tool_errors,
            )

            step_forced_exit = False
            if force_guard and not reflection.confident:
                step_forced_exit = True
                reflection = Reflection(
                    confident=False,
                    coverage=reflection.coverage,
                    top_score=reflection.top_score,
                    relevance_to_intent=reflection.relevance_to_intent,
                    reason=(
                        "iteration guard forced Deep Lane exit after two escalations "
                        f"({reflection.reason})"
                    ),
                )

            trace.append(
                RouterTraceStep(
                    iteration=escalation_level,
                    complexity_score=complexity_score,
                    strategy=strategy,
                    candidates_seen=len(candidates),
                    reflection=reflection,
                    candidate_evidence=step_evidence,
                    tool_errors=tool_errors,
                    forced_exit=step_forced_exit,
                    reasoning_status=reasoning_status,
                    laya_advisory=laya_suggestion if escalation_level == 0 else None,
                )
            )

            if reflection.confident or reached_escalation_cap:
                return RouterResult(
                    parsed_query=parsed_query,
                    candidates=candidates,
                    trace=tuple(trace),
                    forced_exit=step_forced_exit,
                )

            escalation_level += 1

    def score_complexity(self, parsed_query: ParsedQuery) -> float:
        """Compute bounded query complexity in [0.0, 1.0] without double-counting uncertainty."""
        uncertainty = max(
            clamp_score(parsed_query.ambiguity_score),
            clamp_score(1.0 - parsed_query.parser_confidence),
        )
        constraints_factor = min(parsed_query.constraints_count / 5.0, 1.0)
        multi_dimension_bonus = (
            0.12
            if (
                (parsed_query.keywords and parsed_query.file_types)
                or (parsed_query.keywords and parsed_query.date_window is not None)
                or (parsed_query.file_types and parsed_query.date_window is not None)
            )
            else 0.0
        )
        specificity_penalty = (
            0.08 if (not parsed_query.filename_hints and not parsed_query.file_types) else 0.0
        )

        historical_difficulty = 0.0
        if self.history_difficulty_fn is not None:
            try:
                historical_difficulty = clamp_score(float(self.history_difficulty_fn(parsed_query)))
            except Exception:
                historical_difficulty = 0.0

        score = (
            0.18
            + uncertainty * 0.62
            + constraints_factor * 0.20
            + multi_dimension_bonus
            + specificity_penalty
            + historical_difficulty * 0.15
        )
        if parsed_query.filename_hints:
            score -= 0.35
        return clamp_score(score)

    def select_strategy(
        self,
        complexity_score: float,
        escalation_level: int = 0,
        force_deep: bool = False,
        parsed_query: ParsedQuery | None = None,
        prior_trace: Sequence[RouterTraceStep] = (),
    ) -> SearchStrategy:
        """Select retrieval lane, tools, and widening parameters from score and prior trace."""
        widened_limit = self.search_limit * 2

        if force_deep:
            return SearchStrategy(
                lane=SearchLane.DEEP,
                tools=self._deep_tools(parsed_query, prior_trace, widened=True),
                requires_reranking=True,
                uses_llm_reasoning=True,
                widened=True,
                candidate_limit=widened_limit,
            )

        if complexity_score < self.thresholds.fast_max_complexity:
            base_lane = SearchLane.FAST
        elif complexity_score < self.thresholds.standard_max_complexity:
            base_lane = SearchLane.STANDARD
        else:
            base_lane = SearchLane.DEEP

        lane_order = (SearchLane.FAST, SearchLane.STANDARD, SearchLane.DEEP)
        base_index = lane_order.index(base_lane)
        lane_index = min(base_index + escalation_level, len(lane_order) - 1)
        lane = lane_order[lane_index]

        if lane == SearchLane.FAST:
            return SearchStrategy(
                lane=SearchLane.FAST,
                tools=(RetrievalTool.FILENAME, RetrievalTool.METADATA),
                requires_reranking=False,
                uses_llm_reasoning=False,
                widened=False,
                candidate_limit=self.search_limit,
            )

        if lane == SearchLane.STANDARD:
            widened_standard = escalation_level > 0
            tools: list[RetrievalTool] = [RetrievalTool.KEYWORD, RetrievalTool.SEMANTIC]
            if widened_standard and parsed_query is not None:
                # Adapt based on why the prior attempt failed and query constraints
                prior_reason = prior_trace[-1].reflection.reason if prior_trace else ""
                if parsed_query.file_types or parsed_query.date_window is not None or "intent" in prior_reason:
                    tools.append(RetrievalTool.METADATA)
                if parsed_query.filename_hints or "no candidates" in prior_reason:
                    tools.append(RetrievalTool.FILENAME)
            return SearchStrategy(
                lane=SearchLane.STANDARD,
                tools=tuple(dict.fromkeys(tools)),
                requires_reranking=False,
                uses_llm_reasoning=False,
                widened=widened_standard,
                candidate_limit=widened_limit if widened_standard else self.search_limit,
            )

        # DEEP lane: initial pass uses standard hybrid; escalated passes widen retrieval
        widened_deep = escalation_level > 0
        return SearchStrategy(
            lane=SearchLane.DEEP,
            tools=self._deep_tools(parsed_query, prior_trace, widened=widened_deep),
            requires_reranking=True,
            uses_llm_reasoning=True,
            widened=widened_deep,
            candidate_limit=widened_limit if widened_deep else self.search_limit,
        )

    def _deep_tools(
        self,
        parsed_query: ParsedQuery | None,
        prior_trace: Sequence[RouterTraceStep],
        *,
        widened: bool,
    ) -> tuple[RetrievalTool, ...]:
        if not widened:
            return (RetrievalTool.HYBRID,)

        tools: list[RetrievalTool] = [RetrievalTool.HYBRID]
        prior_step = prior_trace[-1] if prior_trace else None

        # If a tool failed or prior reflection lacked coverage/intent match, add explicit fallback tools
        if prior_step is not None and (
            RetrievalTool.HYBRID in prior_step.tool_errors
            or prior_step.reflection.coverage < self.thresholds.min_coverage
            or prior_step.reflection.relevance_to_intent < self.thresholds.deep_min_relevance
        ):
            tools.extend([RetrievalTool.SEMANTIC, RetrievalTool.KEYWORD])
            if parsed_query is not None and (parsed_query.file_types or parsed_query.date_window is not None):
                tools.append(RetrievalTool.METADATA)
            if parsed_query is not None and parsed_query.filename_hints:
                tools.append(RetrievalTool.FILENAME)
        elif parsed_query is not None:
            if parsed_query.file_types or parsed_query.date_window is not None:
                tools.append(RetrievalTool.METADATA)
            tools.append(RetrievalTool.SEMANTIC)

        return tuple(dict.fromkeys(tools))

    def act(
        self,
        strategy: SearchStrategy,
        parsed_query: ParsedQuery,
    ) -> dict[RetrievalTool, ToolExecutionOutcome]:
        """Invoke selected retrieval tools and capture per-tool outcomes or failures."""
        outcomes: dict[RetrievalTool, ToolExecutionOutcome] = {}
        supports_widened = self._toolbox_supports_widened()

        for tool in strategy.tools:
            try:
                if supports_widened:
                    candidates = self.toolbox.search(
                        tool,
                        parsed_query,
                        limit=strategy.candidate_limit,
                        widened=strategy.widened,
                    )
                else:
                    candidates = self.toolbox.search(
                        tool,
                        parsed_query,
                        limit=strategy.candidate_limit,
                    )
                outcomes[tool] = ToolExecutionOutcome(tool=tool, candidates=tuple(candidates))
            except Exception as exc:
                outcomes[tool] = ToolExecutionOutcome(
                    tool=tool,
                    candidates=(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        return outcomes

    def observe(
        self,
        tool_results: Mapping[RetrievalTool, ToolExecutionOutcome | Sequence[SearchCandidate]]
        | Sequence[ToolExecutionOutcome],
        strategy: SearchStrategy,
        parsed_query: ParsedQuery,
        *,
        force_guard: bool = False,
    ) -> tuple[SearchCandidate, ...]:
        """Merge and deduplicate tool outputs, apply bounded reasoning if configured, and order candidates."""
        candidates, _ = self._observe_with_status(
            tool_results,
            strategy,
            parsed_query,
            force_guard=force_guard,
        )
        return candidates

    def _observe_with_status(
        self,
        tool_results: Mapping[RetrievalTool, ToolExecutionOutcome | Sequence[SearchCandidate]]
        | Sequence[ToolExecutionOutcome],
        strategy: SearchStrategy,
        parsed_query: ParsedQuery,
        *,
        force_guard: bool = False,
    ) -> tuple[tuple[SearchCandidate, ...], ReasoningStatus]:
        """Stateless internal observe implementation returning candidates and per-pass ReasoningStatus."""
        by_document: dict[str, list[SearchCandidate]] = defaultdict(list)

        if isinstance(tool_results, Mapping):
            iterable_outcomes = list(tool_results.values())
        else:
            iterable_outcomes = list(tool_results)

        for item in iterable_outcomes:
            if isinstance(item, ToolExecutionOutcome):
                if not item.succeeded:
                    continue
                candidates_list = item.candidates
            else:
                candidates_list = tuple(item)
            for candidate in candidates_list:
                by_document[candidate.document.id].append(candidate)

        merged: list[SearchCandidate] = []
        for candidates in by_document.values():
            best = max(candidates, key=lambda candidate: candidate.score)
            distinct_tools = sorted({candidate.tool.value for candidate in candidates})
            agreement_bonus = min(0.15, 0.05 * (len(distinct_tools) - 1))
            score = clamp_score(best.score + agreement_bonus)
            matched_terms = sorted({term for candidate in candidates for term in candidate.matched_terms})
            used_tools = ", ".join(distinct_tools)
            reasons = [candidate.reason for candidate in candidates if candidate.reason]
            provenance = f"router merged evidence from {used_tools}"
            if reasons:
                provenance = f"{provenance} ({'; '.join(dict.fromkeys(reasons))})"
            merged.append(
                SearchCandidate(
                    document=best.document,
                    score=score,
                    tool=best.tool,
                    matched_terms=tuple(matched_terms),
                    reason=provenance,
                )
            )

        ordered = self._sort_candidates(merged)

        if not strategy.uses_llm_reasoning:
            return ordered, ReasoningStatus.NOT_REQUESTED

        if self.reasoner is None:
            return ordered, ReasoningStatus.UNAVAILABLE

        try:
            reasoned = self.reasoner.reason_candidates(
                parsed_query,
                ordered,
                widened=strategy.widened,
                forced_guard=force_guard,
            )
            validated = self._validate_reasoned_candidates(ordered, reasoned)
            return validated, ReasoningStatus.EXECUTED
        except Exception:
            return ordered, ReasoningStatus.FAILED

    def _validate_reasoned_candidates(
        self,
        original: tuple[SearchCandidate, ...],
        reasoned: object,
    ) -> tuple[SearchCandidate, ...]:
        """Validate single-turn reasoner output against retrieved candidates.

        Rules:
        - Must be a non-string, non-mapping sequence/iterable of SearchCandidate items.
        - Must neither invent unknown candidate IDs, duplicate IDs, nor drop retrieved candidates.
        - Each score must be a finite int/float (excluding bool) in [0.0, 1.0].
        - Reconstructs candidates from the original FileDocument, tool, matched_terms, and provenance.
        """
        if reasoned is None or isinstance(reasoned, (str, bytes, Mapping)):
            raise TypeError("Reasoner output must be a sequence of SearchCandidate objects")
        try:
            reasoned_list = list(reasoned)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("Reasoner output is not iterable") from exc

        if len(reasoned_list) != len(original):
            raise ValueError(
                f"Reasoner candidate count mismatch: expected {len(original)}, got {len(reasoned_list)}"
            )

        original_by_id = {candidate.document.id: candidate for candidate in original}
        seen_ids: set[str] = set()
        reconstructed: list[tuple[int, SearchCandidate]] = []

        for idx, item in enumerate(reasoned_list):
            if not isinstance(item, SearchCandidate):
                raise TypeError(f"Expected SearchCandidate at index {idx}, got {type(item).__name__}")
            if not isinstance(item.document, FileDocument) or not isinstance(item.document.id, str):
                raise TypeError(f"Invalid FileDocument on reasoned candidate at index {idx}")

            doc_id = item.document.id
            if doc_id not in original_by_id:
                raise ValueError(f"Reasoner invented unknown candidate document_id: {doc_id!r}")
            if doc_id in seen_ids:
                raise ValueError(f"Reasoner returned duplicate candidate document_id: {doc_id!r}")
            seen_ids.add(doc_id)

            raw_score = item.score
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
            ):
                raise ValueError(f"Invalid non-finite or boolean score for {doc_id!r}: {raw_score!r}")
            score_val = float(raw_score)
            if not (0.0 <= score_val <= 1.0):
                raise ValueError(f"Reasoner score out of bounds [0.0, 1.0] for {doc_id!r}: {score_val}")

            orig = original_by_id[doc_id]
            provenance = orig.reason
            if isinstance(item.reason, str) and item.reason.strip():
                cleaned_reason = item.reason.strip()
                if orig.reason and orig.reason not in cleaned_reason:
                    provenance = f"{orig.reason}; {cleaned_reason}"
                elif not orig.reason:
                    provenance = cleaned_reason
                else:
                    provenance = cleaned_reason

            reconstructed.append(
                (
                    idx,
                    SearchCandidate(
                        document=orig.document,
                        score=score_val,
                        tool=orig.tool,
                        matched_terms=orig.matched_terms,
                        reason=provenance,
                    ),
                )
            )

        if seen_ids != set(original_by_id.keys()):
            raise ValueError("Reasoner output did not preserve the exact input candidate ID set")

        return tuple(
            candidate
            for _, candidate in sorted(
                reconstructed,
                key=lambda pair: (
                    -pair[1].score,
                    pair[0],
                    len(pair[1].document.path.parts),
                    str(pair[1].document.path).lower(),
                ),
            )
        )

    def _build_step_evidence(
        self,
        candidates: tuple[SearchCandidate, ...],
        tool_outcomes: Mapping[RetrievalTool, ToolExecutionOutcome],
    ) -> tuple[TraceCandidateEvidence, ...]:
        """Build compact, content-free trace evidence for each candidate in a routing pass."""
        tools_by_doc: dict[str, set[RetrievalTool]] = defaultdict(set)
        for outcome in tool_outcomes.values():
            if not outcome.succeeded:
                continue
            for candidate in outcome.candidates:
                tools_by_doc[candidate.document.id].add(candidate.tool)

        evidence: list[TraceCandidateEvidence] = []
        for candidate in candidates:
            doc_id = candidate.document.id
            contributing = sorted(
                tools_by_doc.get(doc_id, {candidate.tool}),
                key=lambda tool: tool.value,
            )
            evidence.append(
                TraceCandidateEvidence(
                    document_id=doc_id,
                    score=candidate.score,
                    tools=tuple(contributing),
                    matched_terms=candidate.matched_terms,
                )
            )
        return tuple(evidence)

    def reflect(
        self,
        parsed_query: ParsedQuery,
        strategy: SearchStrategy,
        candidates: tuple[SearchCandidate, ...],
        *,
        tool_errors: Mapping[RetrievalTool, str] | None = None,
    ) -> Reflection:
        """Evaluate coverage, top score, and constraint-grounded relevance to intent."""
        error_prefix = ""
        if tool_errors:
            formatted = ", ".join(f"{tool.value}: {err}" for tool, err in tool_errors.items())
            error_prefix = f"tool failure ({formatted}); "

        coverage = len(candidates)
        top_score = candidates[0].score if candidates else 0.0
        relevance_to_intent = (
            self._evaluate_relevance_to_intent(parsed_query, candidates[0]) if candidates else 0.0
        )

        required_top_score = self.thresholds.required_top_score(strategy.lane)
        required_relevance = self.thresholds.required_relevance(strategy.lane)

        confident = (
            coverage >= self.thresholds.min_coverage
            and top_score >= required_top_score
            and relevance_to_intent >= required_relevance
        )

        if confident:
            reason = f"{error_prefix}{strategy.lane.value} lane produced enough coverage and relevance"
        elif coverage == 0:
            reason = f"{error_prefix}no candidates returned"
        elif coverage < self.thresholds.min_coverage:
            reason = (
                f"{error_prefix}coverage {coverage} below minimum threshold {self.thresholds.min_coverage}"
            )
        elif top_score < required_top_score:
            reason = f"{error_prefix}top score {top_score:.2f} below {strategy.lane.value} threshold"
        else:
            reason = (
                f"{error_prefix}result relevance {relevance_to_intent:.2f} did not satisfy parsed intent "
                f"(threshold {required_relevance:.2f})"
            )

        return Reflection(
            confident=confident,
            coverage=coverage,
            top_score=top_score,
            relevance_to_intent=relevance_to_intent,
            reason=reason,
        )

    def _evaluate_relevance_to_intent(
        self,
        parsed_query: ParsedQuery,
        candidate: SearchCandidate,
    ) -> float:
        """Score how well a candidate satisfies extracted topical, filename, type, and date constraints."""
        doc = candidate.document
        doc_tokens = (
            set(tokenize(doc.title))
            | set(tokenize(doc.content))
            | set(tokenize(doc.path.name))
            | set(tokenize(" ".join(doc.metadata.values())))
        )

        exact_hint_match = any(
            hint.lower() in doc.path.name.lower() for hint in parsed_query.filename_hints
        )

        # 1. Topical / keyword match evaluation
        keyword_ratio = 0.0
        specific_keywords = [kw for kw in parsed_query.keywords if kw not in VAGUE_QUERY_TERMS]
        has_topical_constraint = bool(parsed_query.keywords or parsed_query.filename_hints)
        if parsed_query.keywords:
            total_kw_score = 0.0
            for kw in parsed_query.keywords:
                is_vague_kw = kw in VAGUE_QUERY_TERMS
                if kw in doc_tokens:
                    total_kw_score += 0.55 if is_vague_kw else 1.0
                elif any(exp in doc_tokens for exp in expand_terms([kw])):
                    total_kw_score += 0.45 if is_vague_kw else 0.65
            keyword_ratio = total_kw_score / len(parsed_query.keywords)
        elif parsed_query.filename_hints:
            keyword_ratio = 1.0 if exact_hint_match else 0.0

        # Guard: if the query specifies topical keywords or filename hints and the candidate matches
        # neither (even via expansion), a high raw numeric score alone cannot make it relevant.
        if has_topical_constraint and keyword_ratio == 0.0 and not exact_hint_match:
            return clamp_score(min(0.15, candidate.score * 0.15))

        # 2. Aggregate weighted constraint satisfaction across active query constraints
        weighted_sum = 0.0
        total_weight = 0.0

        if parsed_query.keywords:
            weight = 0.35 if parsed_query.filename_hints else 0.45
            weighted_sum += weight * keyword_ratio
            total_weight += weight

        if parsed_query.filename_hints:
            weight = 0.45
            weighted_sum += weight * (1.0 if exact_hint_match else 0.0)
            total_weight += weight

        if parsed_query.file_types:
            weight = 0.30
            normalized_ext = doc.extension.lstrip(".").lower()
            normalized_types = {ft.lstrip(".").lower() for ft in parsed_query.file_types}
            type_match = 1.0 if normalized_ext in normalized_types else 0.0
            weighted_sum += weight * type_match
            total_weight += weight

        if parsed_query.date_window is not None:
            weight = 0.25
            date_match = 1.0 if parsed_query.date_window.contains(doc.modified_at) else 0.0
            weighted_sum += weight * date_match
            total_weight += weight

        if total_weight > 0.0:
            constraint_satisfaction = weighted_sum / total_weight
        else:
            # Query had no structured constraints at all (purely vague)
            constraint_satisfaction = 0.25

        # Penalize highly ambiguous queries that lack specific (non-vague) topical/filename grounding
        if parsed_query.ambiguity_score >= 0.50 and not exact_hint_match:
            specific_exact_hits = sum(1 for kw in specific_keywords if kw in doc_tokens)
            if specific_exact_hits == 0:
                constraint_satisfaction *= 1.0 - (parsed_query.ambiguity_score * 0.60)
            elif len(specific_keywords) <= 1:
                constraint_satisfaction *= 1.0 - (parsed_query.ambiguity_score * 0.30)

        relevance = constraint_satisfaction * 0.75 + min(candidate.score, constraint_satisfaction) * 0.25
        return clamp_score(relevance)

    def _sort_candidates(self, candidates: list[SearchCandidate]) -> tuple[SearchCandidate, ...]:
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    len(candidate.document.path.parts),
                    str(candidate.document.path).lower(),
                ),
            )
        )

    def _toolbox_supports_widened(self) -> bool:
        try:
            sig = inspect.signature(self.toolbox.search)
            return "widened" in sig.parameters or any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )
        except (TypeError, ValueError):
            return False
