"""fast-graphrag (Circlemind) adapter."""

from __future__ import annotations

import tempfile
import time

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class FastGraphRAG(GraphRAGFramework):
    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from fast_graphrag import GraphRAG as FastGraphRAGEngine
            from fast_graphrag import DefaultLLMService, DefaultEmbeddingService
        except ImportError:
            raise ImportError("Install fast-graphrag: pip install graphrag-lab[fast]")

        working_dir = tempfile.mkdtemp(prefix="fast_graphrag_")

        llm_service = DefaultLLMService(model=self.llm_config.model)
        embedding_service = DefaultEmbeddingService()

        engine = FastGraphRAGEngine(
            working_dir=working_dir,
            domain="general knowledge, science, and technology",
            example_queries="What are the main topics? How are entities related?",
            entity_types=["Person", "Organization", "Technology", "Concept", "Location", "Event"],
            config=FastGraphRAGEngine.Config(
                llm_service=llm_service,
                embedding_service=embedding_service,
            ),
        )

        start = time.perf_counter()
        await engine.async_insert(documents)
        elapsed = time.perf_counter() - start

        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="fast", path=working_dir, data=engine)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        engine = index.data

        start = time.perf_counter()
        response = await engine.async_query(question)
        elapsed = (time.perf_counter() - start) * 1000

        answer = response.response if hasattr(response, 'response') else str(response)

        result = QueryResult(
            answer=answer,
            latency_ms=elapsed,
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
