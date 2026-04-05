"""Microsoft GraphRAG v3 adapter."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


class MicrosoftGraphRAG(GraphRAGFramework):
    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    def _create_config(self, output_dir: str):
        from graphrag.config.models.graph_rag_config import GraphRagConfig
        from graphrag_llm.config.model_config import ModelConfig

        api_key = os.getenv("OPENAI_API_KEY", "")

        return GraphRagConfig(
            completion_models={
                "default_completion_model": ModelConfig(
                    model_provider="openai",
                    model=self.llm_config.model,
                    api_key=api_key,
                ),
            },
            embedding_models={
                "default_embedding_model": ModelConfig(
                    model_provider="openai",
                    model="text-embedding-3-large",
                    api_key=api_key,
                ),
            },
        )

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        try:
            from graphrag.api import build_index
        except ImportError:
            raise ImportError("Install microsoft graphrag: pip install graphrag-lab[microsoft]")

        import pandas as pd

        graphrag_config = self._create_config("")

        df = pd.DataFrame([
            {"id": f"doc-{i}", "text": doc, "title": f"document_{i}"}
            for i, doc in enumerate(documents)
        ])

        start = time.perf_counter()
        results = await build_index(config=graphrag_config, input_documents=df)
        elapsed = time.perf_counter() - start

        self._cost_info.indexing_latency_s = elapsed

        errors = [r for r in results if r.error]
        # Embedding generation failures are non-fatal for search
        fatal_errors = [r for r in errors if r.workflow != "generate_text_embeddings"]
        if fatal_errors:
            error_msgs = [f"{r.workflow}: {r.error}" for r in fatal_errors]
            raise RuntimeError(f"GraphRAG indexing errors: {error_msgs}")

        output_dir = graphrag_config.output_storage.base_dir

        return IndexArtifact(
            framework="microsoft",
            path=output_dir,
            data={"config": graphrag_config, "output_dir": output_dir},
        )

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        from graphrag.api import local_search, global_search
        import pandas as pd

        data = index.data
        config = data["config"]
        output_dir = data["output_dir"]

        search_type = search_type or "local"

        entities = pd.read_parquet(f"{output_dir}/entities.parquet")
        communities = pd.read_parquet(f"{output_dir}/communities.parquet")
        community_reports = pd.read_parquet(f"{output_dir}/community_reports.parquet")
        text_units = pd.read_parquet(f"{output_dir}/text_units.parquet")
        relationships = pd.read_parquet(f"{output_dir}/relationships.parquet")

        start = time.perf_counter()

        if search_type == "global":
            answer, context = await global_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                community_level=0,
                dynamic_community_selection=False,
                response_type="Multiple Paragraphs",
                query=question,
            )
        else:
            answer, context = await local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=None,
                community_level=0,
                response_type="Multiple Paragraphs",
                query=question,
            )

        elapsed = (time.perf_counter() - start) * 1000

        result = QueryResult(
            answer=str(answer),
            latency_ms=elapsed,
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
