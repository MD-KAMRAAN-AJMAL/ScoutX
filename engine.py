"""Top-level ScoutX orchestration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from scoutx.document_loader import DocumentLoader
from scoutx.downstream import (
    ContextMemoryModule,
    ExplanationModule,
    PersonalizationModule,
    RerankingModule,
)
from scoutx.embeddings import EmbeddingError
from scoutx.models import SearchResponse, UserProfile
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent
from scoutx.storage import MetadataIndex, UsageLog, UserProfileStore, VectorIndex

if TYPE_CHECKING:
    from scoutx.embeddings import EmbeddingProvider
    from scoutx.laya import LayaAdvisor
    from scoutx.llm import LLMProvider


class ScoutXEngine:
    def __init__(
        self,
        corpus: DocumentCorpus,
        usage_log: UsageLog | None = None,
        profile: UserProfile | None = None,
        *,
        llm_provider: LLMProvider | None = None,
        require_llm: bool | None = None,
        allow_degraded_fallback: bool = False,
        embedding_provider: EmbeddingProvider | None = None,
        profile_store: UserProfileStore | None = None,
        metadata_index: MetadataIndex | None = None,
        vector_index: VectorIndex | None = None,
        strict_embeddings: bool = False,
        laya_advisor: LayaAdvisor | None = None,
    ):
        self.corpus = corpus
        self.usage_log = usage_log
        self.profile_store = profile_store or (
            usage_log.profile_store if usage_log is not None else None
        )
        self.metadata_index = metadata_index
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.strict_embeddings = bool(strict_embeddings)
        self.llm_provider = llm_provider
        self.laya_advisor = laya_advisor
        self.query_understanding = QueryUnderstandingModule(llm_provider=llm_provider)
        self.toolbox = RetrievalToolbox(
            corpus,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            metadata_index=metadata_index,
            strict_embeddings=strict_embeddings,
        )
        self.router = ComplexityRoutingAgent(
            self.toolbox,
            reasoner=llm_provider,
            laya_advisor=laya_advisor,
        )
        self.reranker = RerankingModule()
        self.context_memory = ContextMemoryModule(profile)
        self.personalizer = PersonalizationModule()
        self.explainer = ExplanationModule(
            llm_provider=llm_provider,
            require_llm=require_llm,
            allow_degraded_fallback=allow_degraded_fallback,
        )

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        usage_db: str | Path | None = None,
        *,
        index_db: str | Path | None = None,
        loader: DocumentLoader | None = None,
        llm_provider: LLMProvider | None = None,
        require_llm: bool | None = None,
        allow_degraded_fallback: bool = False,
        embedding_provider: EmbeddingProvider | None = None,
        strict_embeddings: bool = False,
        laya_advisor: LayaAdvisor | None = None,
    ) -> "ScoutXEngine":
        usage_log = UsageLog(usage_db) if usage_db is not None else None
        profile_store = usage_log.profile_store if usage_log is not None else None

        effective_index_db = index_db if index_db is not None else usage_db
        metadata_index = (
            MetadataIndex(effective_index_db) if effective_index_db is not None else None
        )
        vector_index = (
            VectorIndex(effective_index_db) if effective_index_db is not None else None
        )

        corpus = DocumentCorpus.from_directory(root, loader=loader)

        if vector_index is not None and embedding_provider is not None:
            try:
                vector_index.index_documents(
                    corpus.documents,
                    embedding_provider,
                    prune_missing=True,
                )
            except (EmbeddingError, sqlite3.Error):
                if strict_embeddings:
                    raise

        return cls(
            corpus,
            usage_log=usage_log,
            profile_store=profile_store,
            metadata_index=metadata_index,
            vector_index=vector_index,
            llm_provider=llm_provider,
            require_llm=require_llm,
            allow_degraded_fallback=allow_degraded_fallback,
            embedding_provider=embedding_provider,
            strict_embeddings=strict_embeddings,
            laya_advisor=laya_advisor,
        )

    def search(self, raw_query: str, limit: int = 5) -> SearchResponse:
        parsed_query = self.query_understanding.parse(raw_query)
        router_result = self.router.run(parsed_query)
        candidates = router_result.candidates

        if router_result.trace and router_result.trace[-1].strategy.requires_reranking:
            candidates = self.reranker.rerank(candidates, parsed_query)

        if self.profile_store is not None:
            profile = self.profile_store.load_profile()
        elif self.usage_log is not None:
            profile = self.usage_log.load_profile()
        else:
            profile = self.context_memory.load_context()

        candidates = self.personalizer.personalize(candidates, parsed_query, profile)
        recommendations = self.explainer.explain(candidates[:limit], router_result, profile)

        if self.usage_log:
            final_lane = (
                router_result.trace[-1].strategy.lane.value
                if router_result.trace
                else None
            )
            self.usage_log.record_search(
                raw_query,
                len(recommendations),
                lane=final_lane,
                forced_exit=router_result.forced_exit,
            )

        return SearchResponse(recommendations=recommendations, router_result=router_result)
