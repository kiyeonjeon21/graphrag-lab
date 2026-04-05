"""Cognee adapter.

Cognee provides an extremely simple API for building knowledge graphs:
  await cognee.add(text) -> await cognee.cognify() -> await cognee.search(query)

Auto-optimizes chunking, LLM selection, and data model.
Supports 30+ source formats.

pip install cognee
"""

from __future__ import annotations

import time

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class CogneeGraphRAG(GraphRAGFramework):
    """Cognee adapter — 6-line knowledge graph construction.

    Extra config options (via config.extra):
        llm_api_key: str - LLM API key (defaults to env var)
        vector_db: str - Vector DB backend (default: "lancedb")
        graph_db: str - Graph DB backend (default: "networkx", also: "neo4j", "falkordb")
        reset: bool - Reset Cognee state before indexing (default: True)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            import cognee
        except ImportError:
            raise ImportError("Install cognee: pip install graphrag-lab[cognee]")

        extra = config.extra

        # Configure Cognee LLM provider
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        cognee.config.set_llm_config({
            "llm_provider": "openai",
            "llm_model": self.llm_config.model,
            "llm_api_key": api_key,
        })

        if extra.get("reset", True):
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

        start = time.perf_counter()

        # Add all documents
        for doc in documents:
            await cognee.add(doc)

        # Build knowledge graph (cognify = chunk + extract entities + build relations + summarize)
        await cognee.cognify()

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="cognee", data={"cognee_module": cognee})

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        cognee = index.data["cognee_module"]

        start = time.perf_counter()

        search_type_map = {
            "local": "GRAPH_COMPLETION",
            "global": "SUMMARIES",
            None: "GRAPH_COMPLETION",
        }
        cognee_search_type = search_type_map.get(search_type, "GRAPH_COMPLETION")

        from cognee.api.v1.search import SearchType
        results = await cognee.search(query_text=question, query_type=SearchType[cognee_search_type])

        elapsed = (time.perf_counter() - start) * 1000

        # Parse results
        answer_parts = []
        context_docs = []
        if results:
            for item in results:
                if isinstance(item, dict):
                    answer_parts.append(str(item.get("text", item)))
                    context_docs.append(str(item)[:200])
                else:
                    answer_parts.append(str(item))
                    context_docs.append(str(item)[:200])

        result = QueryResult(
            answer="\n".join(answer_parts) if answer_parts else "",
            context_documents=context_docs[:5],
            latency_ms=elapsed,
            metadata={"search_type": cognee_search_type},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
