"""RAG-Anything adapter (by HKUDS, LightRAG team).

All-in-one multimodal RAG framework built on top of LightRAG.
For text-only use, we create a LightRAG instance and pass it to RAGAnything.

pip install raganything
"""

from __future__ import annotations

import tempfile
import time
from functools import partial

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class RAGAnythingFramework(GraphRAGFramework):
    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from raganything import RAGAnything
            from lightrag import LightRAG as LightRAGEngine
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
        except ImportError:
            raise ImportError("Install raganything: pip install graphrag-lab[raganything]")

        working_dir = config.extra.get("working_dir", tempfile.mkdtemp(prefix="raganything_"))

        # Build LightRAG instance (same pattern as lightrag adapter)
        llm_func = partial(openai_complete_if_cache, self.llm_config.model)
        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=partial(openai_embed, model="text-embedding-3-small"),
        )

        lightrag_instance = LightRAGEngine(
            working_dir=working_dir,
            llm_model_func=llm_func,
            llm_model_name=self.llm_config.model,
            llm_model_kwargs={"temperature": self.llm_config.temperature},
            embedding_func=embedding_func,
        )
        await lightrag_instance.initialize_storages()

        rag = RAGAnything(lightrag=lightrag_instance)

        start = time.perf_counter()
        # Use LightRAG's insert for text-only documents
        await lightrag_instance.ainsert(documents)
        elapsed = time.perf_counter() - start

        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="raganything", data={"rag": rag}, path=working_dir)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        rag = index.data["rag"]
        mode = search_type or "hybrid"
        valid_modes = {"naive", "local", "global", "hybrid", "mix"}
        if mode not in valid_modes:
            mode = "hybrid"

        start = time.perf_counter()
        answer = await rag.aquery(question, mode=mode)
        elapsed = (time.perf_counter() - start) * 1000

        result = QueryResult(
            answer=str(answer) if answer else "",
            latency_ms=elapsed,
            metadata={"search_type": mode},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
