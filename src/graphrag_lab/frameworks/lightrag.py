"""LightRAG adapter."""

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


class LightRAGFramework(GraphRAGFramework):
    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from lightrag import LightRAG as LightRAGEngine
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
        except ImportError:
            raise ImportError("Install lightrag: pip install graphrag-lab[lightrag]")

        working_dir = tempfile.mkdtemp(prefix="lightrag_")

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embed,
        )

        # partial bind model as first positional arg because LightRAG
        # calls llm_func(prompt, ...) but openai_complete_if_cache expects (model, prompt, ...)
        llm_func = partial(openai_complete_if_cache, self.llm_config.model)

        engine = LightRAGEngine(
            working_dir=working_dir,
            llm_model_func=llm_func,
            llm_model_name=self.llm_config.model,
            llm_model_kwargs={"temperature": self.llm_config.temperature},
            embedding_func=embedding_func,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
        )

        await engine.initialize_storages()

        start = time.perf_counter()
        await engine.ainsert(documents)
        elapsed = time.perf_counter() - start

        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="lightrag", path=working_dir, data=engine)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        from lightrag import QueryParam

        engine = index.data
        mode = search_type or "hybrid"

        # LightRAG modes: naive, local, global, hybrid, mix
        valid_modes = {"naive", "local", "global", "hybrid", "mix"}
        if mode not in valid_modes:
            mode = "hybrid"

        start = time.perf_counter()
        answer = await engine.aquery(question, param=QueryParam(mode=mode))
        elapsed = (time.perf_counter() - start) * 1000

        result = QueryResult(
            answer=answer if isinstance(answer, str) else str(answer),
            latency_ms=elapsed,
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
