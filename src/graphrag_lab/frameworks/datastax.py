"""DataStax Graph RAG adapter.

Uses langchain-graph-retriever to perform graph traversal over vector store metadata.
No separate graph database required — edges are defined by metadata field relationships.

Supports search_type: local (eager), hybrid (mmr)
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


class DataStaxGraphRAG(GraphRAGFramework):
    """DataStax Graph RAG adapter using langchain-graph-retriever.

    Extra config options (via config.extra):
        edges: list[tuple[str, str]] - Metadata field pairs defining graph edges
            Example: [["category", "category"], ["keywords", "keywords"], ["mentions", "$id"]]
        strategy: str - Traversal strategy: "eager" (default) or "mmr"
        start_k: int - Initial vector search results (default: 3)
        select_k: int - Final number of documents to return (default: 10)
        max_depth: int - Maximum graph traversal depth (default: 2)
        lambda_mult: float - MMR lambda (0=diversity, 1=relevance, default: 0.5)

        # Vector store config (one of):
        vector_store: str - "chroma" (default, local) or "astra"

        # For Astra DB:
        astra_endpoint: str - Astra DB API endpoint
        astra_token: str - Astra DB application token
        collection_name: str - Astra DB collection name (default: "graphrag_lab")
    """

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        super().__init__(llm_config)
        self._cost_info = CostInfo()

    def _ensure_deps(self):
        try:
            import langchain_graph_retriever  # noqa: F401
        except ImportError:
            raise ImportError(
                "Install datastax graph-rag dependencies: "
                "pip install langchain-graph-retriever langchain-chroma langchain-openai"
            )

    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        self._ensure_deps()
        extra = config.extra

        start = time.perf_counter()

        vector_store_type = extra.get("vector_store", "chroma")
        store = self._build_vector_store(documents, extra, vector_store_type)

        elapsed = time.perf_counter() - start
        self._cost_info.indexing_latency_s = elapsed

        return IndexArtifact(
            framework="datastax",
            data={
                "store": store,
                "edges": extra.get("edges", []),
                "strategy": extra.get("strategy", "eager"),
                "start_k": extra.get("start_k", 3),
                "select_k": extra.get("select_k", 10),
                "max_depth": extra.get("max_depth", 2),
                "lambda_mult": extra.get("lambda_mult", 0.5),
            },
        )

    def _build_vector_store(self, documents: list[str], extra: dict[str, Any], store_type: str):
        """Build and populate a vector store with documents + metadata."""
        from langchain_core.documents import Document as LCDocument
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Convert raw texts to LangChain Documents with basic metadata
        lc_docs = []
        for i, text in enumerate(documents):
            # Extract simple metadata for graph edges
            keywords = self._extract_keywords(text)
            lc_docs.append(
                LCDocument(
                    page_content=text[:2000],  # Truncate for embedding
                    metadata={
                        "doc_id": f"doc_{i}",
                        "keywords": keywords,
                        "chunk_index": i,
                    },
                )
            )

        if store_type in ("chroma", "memory"):
            from langchain_core.vectorstores import InMemoryVectorStore
            store = InMemoryVectorStore(embedding=embeddings)
            store.add_documents(lc_docs)
        elif store_type == "astra":
            from langchain_astradb import AstraDBVectorStore
            store = AstraDBVectorStore(
                embedding=embeddings,
                api_endpoint=extra["astra_endpoint"],
                token=extra["astra_token"],
                collection_name=extra.get("collection_name", "graphrag_lab"),
            )
            store.add_documents(lc_docs)
        else:
            raise ValueError(f"Unsupported vector_store type: {store_type}")

        return store

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Simple keyword extraction from text (no LLM needed)."""
        import re
        from collections import Counter

        words = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
        counter = Counter(words)
        return [word for word, _ in counter.most_common(max_keywords)]

    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        data = index.data
        store = data["store"]
        edges = data["edges"]

        start = time.perf_counter()

        retriever = self._build_retriever(store, edges, data, search_type)
        retrieved_docs = retriever.invoke(question)

        # Generate answer using LLM
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
        answer = await self._generate_answer(question, context)

        elapsed = (time.perf_counter() - start) * 1000

        result = QueryResult(
            answer=answer,
            context_documents=[doc.page_content[:200] for doc in retrieved_docs[:5]],
            latency_ms=elapsed,
            metadata={
                "search_type": search_type or "eager",
                "num_retrieved": len(retrieved_docs),
            },
        )
        self._cost_info.query_latencies[question] = elapsed
        return result

    def _build_retriever(self, store, edges: list, data: dict, search_type: str | None):
        from langchain_graph_retriever import GraphRetriever

        strategy_name = search_type or data.get("strategy", "eager")

        if strategy_name in ("local", "eager"):
            from graph_retriever.strategies import Eager
            strategy = Eager(
                k=data["select_k"],
                start_k=data["start_k"],
                max_depth=data["max_depth"],
            )
        elif strategy_name in ("hybrid", "mmr"):
            from graph_retriever.strategies import Mmr
            strategy = Mmr(
                k=data["select_k"],
                start_k=data["start_k"],
                max_depth=data["max_depth"],
                lambda_mult=data["lambda_mult"],
            )
        else:
            raise ValueError(f"Unsupported strategy for DataStax: {strategy_name}")

        return GraphRetriever(
            store=store,
            edges=edges or [("keywords", "keywords")],
            strategy=strategy,
        )

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using configured LLM provider."""
        if self.llm_config.provider == "openai":
            from graphrag_lab.providers.openai import OpenAIProvider
            provider = OpenAIProvider(self.llm_config)
        elif self.llm_config.provider == "anthropic":
            from graphrag_lab.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider(self.llm_config)
        else:
            from graphrag_lab.providers.ollama import OllamaProvider
            provider = OllamaProvider(self.llm_config)

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        response = await provider.complete(prompt)
        return response.text

    def get_cost_info(self) -> CostInfo:
        return self._cost_info
