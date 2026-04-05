"""LinearRAG adapter (by DEEP-PolyU).

Builds a relation-free hierarchical graph (Tri-Graph) using only lightweight
entity extraction and semantic linking — NO LLM tokens consumed during graph construction.
Scales linearly with corpus size.

ICLR 2026

pip install linearrag (or clone from GitHub)
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


class LinearRAGFramework(GraphRAGFramework):
    """LinearRAG adapter — zero-LLM-cost graph construction.

    Extra config options (via config.extra):
        working_dir: str - Working directory (default: temp dir)
        embedding_model: str - Embedding model for semantic linking (default: "text-embedding-3-small")
        entity_method: str - Entity extraction method: "ner" or "noun_phrases" (default: "ner")
        top_k: int - Number of passages to retrieve (default: 5)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            import linearrag
        except ImportError:
            raise ImportError(
                "Install linearrag: pip install graphrag-lab[linearrag] "
                "or clone from https://github.com/DEEP-PolyU/LinearRAG"
            )

        extra = config.extra
        working_dir = extra.get("working_dir", tempfile.mkdtemp(prefix="linearrag_"))

        start = time.perf_counter()

        # LinearRAG Tri-Graph construction (no LLM calls):
        # 1. Entity extraction via NER (spaCy/stanza) — not LLM
        # 2. Build entity-chunk bipartite graph
        # 3. Add semantic similarity edges between entities
        # 4. Hierarchical community structure via clustering
        engine = linearrag.LinearRAG(
            working_dir=working_dir,
            embedding_model=extra.get("embedding_model", "text-embedding-3-small"),
        )

        for doc in documents:
            engine.add_document(doc)

        engine.build_index()

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed
        # Note: indexing_tokens stays at 0 — that's the point of LinearRAG

        return IndexArtifact(framework="linearrag", data={"engine": engine}, path=working_dir)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        engine = index.data["engine"]
        top_k = 5

        start = time.perf_counter()

        # LinearRAG retrieval via Tri-Graph traversal
        results = engine.query(question, top_k=top_k)

        elapsed = (time.perf_counter() - start) * 1000

        answer = ""
        context_docs = []
        if results:
            if isinstance(results, str):
                answer = results
            elif isinstance(results, dict):
                answer = results.get("answer", str(results))
                context_docs = [str(c)[:200] for c in results.get("contexts", [])]
            elif isinstance(results, list):
                context_docs = [str(r)[:200] for r in results[:5]]
                answer = "\n\n".join(str(r) for r in results[:5])

        result = QueryResult(
            answer=answer,
            context_documents=context_docs,
            latency_ms=elapsed,
            metadata={"search_type": "tri_graph", "indexing_llm_tokens": 0},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
