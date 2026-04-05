"""nano-graphrag adapter."""

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


class NanoGraphRAG(GraphRAGFramework):
    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from nano_graphrag import GraphRAG as NanoGraphRAGEngine
            from nano_graphrag._llm import openai_complete_if_cache, openai_embedding
        except ImportError:
            raise ImportError("Install nano-graphrag: pip install graphrag-lab[nano]")

        working_dir = tempfile.mkdtemp(prefix="nano_graphrag_")
        model_name = self.llm_config.model

        async def llm_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        engine = NanoGraphRAGEngine(
            working_dir=working_dir,
            best_model_func=llm_complete,
            cheap_model_func=llm_complete,
            embedding_func=openai_embedding,
        )

        start = time.perf_counter()
        await engine.ainsert(documents)
        elapsed = time.perf_counter() - start

        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(framework="nano", path=working_dir, data=engine)

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        from nano_graphrag.base import QueryParam

        engine = index.data
        mode = search_type or "local"

        # nano-graphrag modes: local, global, naive
        valid_modes = {"local", "global", "naive"}
        if mode not in valid_modes:
            mode = "local"

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
