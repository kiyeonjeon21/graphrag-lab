"""HippoRAG adapter (by OSU NLP Group).

Inspired by human hippocampal memory indexing theory.
Combines Knowledge Graph + Personalized PageRank for efficient multi-hop retrieval.
Compresses multi-hop retrieval into a single step.

NeurIPS'24 (v1), ICML'25 (v2)

pip install hipporag
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


class HippoRAGFramework(GraphRAGFramework):
    """HippoRAG adapter — hippocampus-inspired retrieval.

    Extra config options (via config.extra):
        working_dir: str - Working directory (default: temp dir)
        extraction_model: str - Entity extraction model (default: uses llm_config.model)
        retrieval_model: str - Retrieval model (default: uses llm_config.model)
        recognition_threshold: float - Entity recognition threshold (default: 0.9)
        damping_factor: float - PageRank damping factor (default: 0.5)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            import hipporag
        except ImportError:
            raise ImportError("Install hipporag: pip install graphrag-lab[hipporag]")

        extra = config.extra
        working_dir = extra.get("working_dir", tempfile.mkdtemp(prefix="hipporag_"))

        start = time.perf_counter()

        # HippoRAG indexing pipeline:
        # 1. Named entity recognition from passages
        # 2. OpenIE-style triple extraction
        # 3. Knowledge graph construction with synonym/paraphrase edges
        # 4. Personalized PageRank preparation
        hippo = hipporag.HippoRAG(
            working_dir=working_dir,
            llm_model=self.llm_config.model,
            extraction_model=extra.get("extraction_model", self.llm_config.model),
            recognition_threshold=extra.get("recognition_threshold", 0.9),
            damping_factor=extra.get("damping_factor", 0.5),
        )

        for doc in documents:
            hippo.index(doc)

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="hipporag", data={"hippo": hippo}, path=working_dir)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        hippo = index.data["hippo"]

        start = time.perf_counter()

        # HippoRAG retrieval:
        # 1. Extract query entities (NER)
        # 2. Link to KG nodes (synonym matching)
        # 3. Run Personalized PageRank from query nodes
        # 4. Retrieve top-k passages by PPR score
        results = hippo.retrieve(question, top_k=5)

        elapsed = (time.perf_counter() - start) * 1000

        answer_parts = []
        context_docs = []
        if results:
            for passage in results:
                text = passage if isinstance(passage, str) else str(passage)
                answer_parts.append(text)
                context_docs.append(text[:200])

        result = QueryResult(
            answer="\n\n".join(answer_parts) if answer_parts else "",
            context_documents=context_docs,
            latency_ms=elapsed,
            metadata={"search_type": "personalized_pagerank"},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
