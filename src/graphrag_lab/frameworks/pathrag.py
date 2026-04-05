"""PathRAG adapter (by BUPT-GAMMA).

Extracts key relationship paths between relevant nodes in the knowledge graph
and uses flow-based pruning to remove redundant information.
Includes visualization UI.

EDBT 2025

pip install pathrag (or clone from GitHub)
"""

from __future__ import annotations

import time
import tempfile

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class PathRAGFramework(GraphRAGFramework):
    """PathRAG adapter — path-based graph retrieval.

    Extra config options (via config.extra):
        working_dir: str - Working directory (default: temp dir)
        embedding_model: str - Embedding model (default: "text-embedding-3-small")
        max_path_length: int - Maximum path length for traversal (default: 3)
        pruning_strategy: str - "flow" or "none" (default: "flow")
        top_k: int - Number of paths to retrieve (default: 5)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            import pathrag
        except ImportError:
            raise ImportError(
                "Install pathrag: pip install graphrag-lab[pathrag] "
                "or clone from https://github.com/BUPT-GAMMA/PathRAG"
            )

        extra = config.extra
        working_dir = extra.get("working_dir", tempfile.mkdtemp(prefix="pathrag_"))

        start = time.perf_counter()

        # PathRAG indexing:
        # 1. Text chunking
        # 2. LLM-based entity & relationship extraction
        # 3. Knowledge graph construction
        # 4. Path index preparation
        engine = pathrag.PathRAG(
            working_dir=working_dir,
            llm_model=self.llm_config.model,
            embedding_model=extra.get("embedding_model", "text-embedding-3-small"),
        )

        for doc in documents:
            await engine.ainsert(doc)

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="pathrag", data={"engine": engine}, path=working_dir)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        engine = index.data["engine"]
        extra = {}

        start = time.perf_counter()

        # PathRAG retrieval:
        # 1. Identify query-relevant nodes
        # 2. Extract key paths between relevant nodes
        # 3. Flow-based pruning to remove redundant paths
        # 4. Convert paths to text for LLM prompting
        search_mode = search_type or "path"
        answer = await engine.aquery(question, mode=search_mode)

        elapsed = (time.perf_counter() - start) * 1000

        result = QueryResult(
            answer=str(answer) if answer else "",
            latency_ms=elapsed,
            metadata={"search_type": search_mode},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
