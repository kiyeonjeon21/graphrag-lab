"""Neo4j GraphRAG Python adapter.

Uses neo4j-graphrag package (SimpleKGPipeline for indexing, various Retrievers for querying).
Requires a running Neo4j instance.

Supports search_type: local (vector), hybrid, vector_cypher, text2cypher
"""

from __future__ import annotations

import time
from typing import Any

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig
from graphrag_lab.frameworks.base import (
    CostInfo,
    GraphRAGFramework,
    IndexArtifact,
    QueryResult,
    TokenUsage,
)


# Default Cypher query for vector_cypher retriever: expand Chunk to related entities
DEFAULT_RETRIEVAL_QUERY = """
WITH node AS chunk, score
OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)
OPTIONAL MATCH (entity)-[r]->(related:__Entity__)
WITH chunk, score, collect(DISTINCT entity.name + ' -> ' + type(r) + ' -> ' + related.name) AS triples
RETURN chunk.text + '\nRelated: ' + reduce(s='', t IN triples | s + t + '; ') AS text, score
"""


class Neo4jGraphRAG(GraphRAGFramework):
    """Neo4j GraphRAG adapter using neo4j-graphrag-python package.

    Extra config options (via config.extra):
        neo4j_uri: str - Neo4j connection URI (default: neo4j://localhost:7687)
        neo4j_user: str - Neo4j username (default: neo4j)
        neo4j_password: str - Neo4j password (default: neo4j)
        schema: dict | None - KG schema {node_types, relationship_types, patterns}
        index_name: str - Vector index name (default: graphrag_lab_index)
        retrieval_query: str - Custom Cypher for vector_cypher retriever
        perform_entity_resolution: bool - Enable entity resolution (default: True)
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()
        self._driver = None
        self._llm = None
        self._embedder = None

    def _ensure_deps(self):
        try:
            import neo4j_graphrag  # noqa: F401
        except ImportError:
            raise ImportError("Install neo4j-graphrag: pip install 'neo4j-graphrag[openai,experimental]'")

    def _init_driver(self, extra: dict[str, Any]):
        import os
        from neo4j import GraphDatabase

        uri = extra.get("neo4j_uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = extra.get("neo4j_user") or os.getenv("NEO4J_USER", "neo4j")
        password = extra.get("neo4j_password") or os.getenv("NEO4J_PASSWORD", "graphraglab")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def _init_llm(self):
        """Create a neo4j_graphrag LLM instance from our LLMProviderConfig."""
        if self.llm_config.provider == "openai":
            from neo4j_graphrag.llm import OpenAILLM
            self._llm = OpenAILLM(
                model_name=self.llm_config.model,
                model_params={"temperature": self.llm_config.temperature},
            )
        elif self.llm_config.provider == "anthropic":
            from neo4j_graphrag.llm import AnthropicLLM
            self._llm = AnthropicLLM(
                model_name=self.llm_config.model,
                model_params={"temperature": self.llm_config.temperature},
            )
        elif self.llm_config.provider == "ollama":
            from neo4j_graphrag.llm import OllamaLLM
            base_url = self.llm_config.api_base or "http://localhost:11434"
            self._llm = OllamaLLM(
                model_name=self.llm_config.model,
                base_url=base_url,
            )
        else:
            raise ValueError(f"Unsupported provider for Neo4j: {self.llm_config.provider}")

    def _init_embedder(self):
        """Create an embedder. Defaults to OpenAI text-embedding-3-large."""
        from neo4j_graphrag.embeddings import OpenAIEmbeddings
        self._embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        self._ensure_deps()
        extra = config.extra
        self._init_driver(extra)
        self._init_llm()
        self._init_embedder()

        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

        schema = extra.get("schema")
        if schema is None:
            # Provide a reasonable default schema to avoid LLM-based schema generation failures
            schema = {
                "node_types": ["Person", "Organization", "Location", "Technology", "Concept", "Event"],
                "relationship_types": [
                    "WORKS_FOR", "LOCATED_IN", "DEVELOPS", "REGULATES",
                    "COLLABORATES_WITH", "RELATED_TO", "PART_OF",
                ],
            }
        index_name = extra.get("index_name", "graphrag_lab_index")
        perform_entity_resolution = extra.get("perform_entity_resolution", True)

        kg_builder = SimpleKGPipeline(
            llm=self._llm,
            driver=self._driver,
            embedder=self._embedder,
            schema=schema,
            from_pdf=False,
            perform_entity_resolution=perform_entity_resolution,
        )

        start = time.perf_counter()
        for doc in documents:
            await kg_builder.run_async(text=doc)
        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        # Create vector index
        from neo4j_graphrag.indexes import create_vector_index
        try:
            create_vector_index(
                self._driver,
                index_name,
                label="Chunk",
                embedding_property="embedding",
                dimensions=1536,
                similarity_fn="cosine",
            )
        except Exception:
            pass  # Index may already exist

        return IndexArtifact(
            framework="neo4j",
            data={"index_name": index_name, "retrieval_query": extra.get("retrieval_query", DEFAULT_RETRIEVAL_QUERY)},
        )

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        search_type = search_type or "local"
        index_name = index.data["index_name"]
        retrieval_query = index.data.get("retrieval_query", DEFAULT_RETRIEVAL_QUERY)

        start = time.perf_counter()

        retriever = self._build_retriever(search_type, index_name, retrieval_query)

        from neo4j_graphrag.generation import GraphRAG as Neo4jGraphRAGEngine
        rag = Neo4jGraphRAGEngine(retriever=retriever, llm=self._llm)
        response = rag.search(query_text=question, retriever_config={"top_k": 5}, return_context=True)

        elapsed = (time.perf_counter() - start) * 1000

        context_docs = []
        if hasattr(response, "retriever_result") and response.retriever_result:
            context_docs = [str(item) for item in response.retriever_result.items[:5]]

        result = QueryResult(
            answer=response.answer,
            context_documents=context_docs,
            latency_ms=elapsed,
            metadata={"search_type": search_type},
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def _build_retriever(self, search_type: str, index_name: str, retrieval_query: str):
        if search_type in ("local", "vector"):
            from neo4j_graphrag.retrievers import VectorRetriever
            return VectorRetriever(driver=self._driver, index_name=index_name, embedder=self._embedder)

        elif search_type == "vector_cypher":
            from neo4j_graphrag.retrievers import VectorCypherRetriever
            return VectorCypherRetriever(
                driver=self._driver,
                index_name=index_name,
                embedder=self._embedder,
                retrieval_query=retrieval_query,
            )

        elif search_type == "hybrid":
            from neo4j_graphrag.retrievers import HybridRetriever
            return HybridRetriever(
                driver=self._driver,
                vector_index_name=index_name,
                fulltext_index_name=f"{index_name}_fulltext",
                embedder=self._embedder,
            )

        elif search_type == "text2cypher":
            from neo4j_graphrag.retrievers import Text2CypherRetriever
            return Text2CypherRetriever(driver=self._driver, llm=self._llm)

        else:
            raise ValueError(f"Unsupported search_type for Neo4j: {search_type}")

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
