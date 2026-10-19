"""Top-level ScoutX orchestration."""

from __future__ import annotations

from pathlib import Path

from scoutx.downstream import ContextMemoryModule, ExplanationModule, PersonalizationModule, RerankingModule
from scoutx.models import SearchResponse, UserProfile
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent
from scoutx.storage import UsageLog


class ScoutXEngine:
    def __init__(
        self,
        corpus: DocumentCorpus,
        usage_log: UsageLog | None = None,
        profile: UserProfile | None = None,
    ):
        self.corpus = corpus
        self.usage_log = usage_log
        self.query_understanding = QueryUnderstandingModule()
        self.router = ComplexityRoutingAgent(RetrievalToolbox(corpus))
        self.reranker = RerankingModule()
        self.context_memory = ContextMemoryModule(profile)
        self.personalizer = PersonalizationModule()
        self.explainer = ExplanationModule()

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        usage_db: str | Path | None = None,
    ) -> "ScoutXEngine":
        usage_log = UsageLog(usage_db) if usage_db is not None else None
        return cls(DocumentCorpus.from_directory(root), usage_log=usage_log)

    def search(self, raw_query: str, limit: int = 5) -> SearchResponse:
        parsed_query = self.query_understanding.parse(raw_query)
        router_result = self.router.run(parsed_query)
        candidates = router_result.candidates

        if router_result.trace and router_result.trace[-1].strategy.requires_reranking:
            candidates = self.reranker.rerank(candidates, parsed_query)

        profile = self.usage_log.load_profile() if self.usage_log else self.context_memory.load_context()
        candidates = self.personalizer.personalize(candidates, parsed_query, profile)
        recommendations = self.explainer.explain(candidates[:limit], router_result, profile)

        if self.usage_log:
            self.usage_log.record_search(raw_query, len(recommendations))

        return SearchResponse(recommendations=recommendations, router_result=router_result)

