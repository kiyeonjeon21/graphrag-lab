"""Graphiti (by Zep) adapter.

Temporal-aware knowledge graph engine designed as an AI agent memory layer.
Each fact has a validity window (valid_from, valid_to) for tracking changes over time.
Combines semantic embeddings + BM25 + graph traversal.

pip install graphiti-core
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class GraphitiGraphRAG(GraphRAGFramework):
    """Graphiti adapter — temporal knowledge graph.

    Extra config options (via config.extra):
        neo4j_uri: str - Neo4j connection URI (default: bolt://localhost:7687)
        neo4j_user: str - Neo4j username (default: neo4j)
        neo4j_password: str - Neo4j password (default: neo4j)
        group_id: str - Namespace for data isolation (default: "graphrag_lab")
        destroy_existing: bool - Clear existing graph data before indexing (default: True)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()
        self._client = None

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from graphiti_core import Graphiti
        except ImportError:
            raise ImportError("Install graphiti: pip install graphrag-lab[graphiti]")

        import os
        extra = config.extra
        neo4j_uri = extra.get("neo4j_uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = extra.get("neo4j_user") or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = extra.get("neo4j_password") or os.getenv("NEO4J_PASSWORD", "graphraglab")
        group_id = extra.get("group_id", "graphrag_lab")

        # Initialize Graphiti client
        self._client = Graphiti(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
        )

        if extra.get("destroy_existing", True):
            try:
                await self._client.build_indices_and_constraints()
            except Exception:
                pass

        start = time.perf_counter()

        # Add episodes (documents) with timestamps
        for i, doc in enumerate(documents):
            await self._client.add_episode(
                name=f"doc_{i}",
                episode_body=doc,
                source_description="graphrag_lab_experiment",
                group_id=group_id,
                reference_time=datetime.now(timezone.utc),
            )

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="graphiti", data={"client": self._client, "group_id": group_id})

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        client = index.data["client"]
        group_id = index.data["group_id"]

        start = time.perf_counter()

        # Graphiti search combines semantic + BM25 + graph traversal
        results = await client.search(
            query=question,
            group_ids=[group_id],
            num_results=10,
        )

        elapsed = (time.perf_counter() - start) * 1000

        answer_parts = []
        context_docs = []
        if results:
            for edge in results:
                fact = getattr(edge, "fact", str(edge))
                answer_parts.append(str(fact))
                context_docs.append(str(fact)[:200])

        result = QueryResult(
            answer="\n".join(answer_parts) if answer_parts else "",
            context_documents=context_docs[:5],
            latency_ms=elapsed,
            metadata={"search_type": "hybrid_temporal", "num_results": len(results) if results else 0},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
