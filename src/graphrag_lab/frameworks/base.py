"""Abstract base class for all GraphRAG framework adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from graphrag_lab.config.schema import FrameworkConfig, LLMProviderConfig


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self


@dataclass
class QueryResult:
    answer: str
    context_documents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class CostInfo:
    indexing_tokens: TokenUsage = field(default_factory=TokenUsage)
    indexing_latency_s: float = 0.0
    query_tokens: dict[str, TokenUsage] = field(default_factory=dict)
    query_latencies: dict[str, float] = field(default_factory=dict)


@dataclass
class IndexArtifact:
    """Opaque container for a built graph index. Framework-specific data goes in `data`."""

    framework: str
    data: Any = None
    path: str | None = None


class GraphRAGFramework(ABC):
    """Abstract interface that all framework adapters must implement."""

    def __init__(self, llm_config: LLMProviderConfig) -> None:
        self.llm_config = llm_config

    @abstractmethod
    async def build_index(self, documents: list[str], config: FrameworkConfig) -> IndexArtifact:
        """Build the knowledge graph index from documents.

        Args:
            documents: List of document texts.
            config: Framework-specific configuration.

        Returns:
            An IndexArtifact that can be passed to query().
        """

    @abstractmethod
    async def query(self, question: str, index: IndexArtifact, search_type: str | None = None) -> QueryResult:
        """Run a single query against the built index.

        Args:
            question: The query string.
            index: The IndexArtifact from build_index().
            search_type: Optional search strategy override.

        Returns:
            QueryResult with answer, metadata, and usage info.
        """

    @abstractmethod
    def get_cost_info(self) -> CostInfo:
        """Return accumulated token usage and timing information."""
