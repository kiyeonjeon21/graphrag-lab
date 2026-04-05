"""Pydantic configuration models for all experiment types."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-5.4-mini"
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "semantic"] = "fixed"
    chunk_size: int = 1200
    overlap: int = 100


class FrameworkConfig(BaseModel):
    name: Literal[
        "microsoft", "lightrag", "nano", "fast", "neo4j", "datastax",
        "cognee", "graphiti", "raganything", "hipporag", "linearrag", "pathrag",
    ]
    search_type: Literal[
        "local", "global", "drift", "lazy", "hybrid", "text2cypher", "vector_cypher",
        "naive", "mix", "path", "tree",
    ] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["comprehensiveness", "diversity", "relevance", "faithfulness"]
    )
    judge_model: LLMProviderConfig | None = None
    num_eval_samples: int = 50


class CostTrackingConfig(BaseModel):
    enabled: bool = True
    track_indexing: bool = True
    track_query: bool = True


class DatasetConfig(BaseModel):
    name: str
    path: str
    domain: Literal["legal", "medical", "financial", "general"] | None = None
    sample_size: int | None = None


class SweepConfig(BaseModel):
    """Parameter sweep definition. Keys are dot-separated config paths, values are lists."""

    parameters: dict[str, list[Any]] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    id: str = Field(default_factory=lambda: f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}")
    name: str
    description: str = ""
    dataset: DatasetConfig
    frameworks: list[FrameworkConfig]
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    cost_tracking: CostTrackingConfig = Field(default_factory=CostTrackingConfig)
    queries: list[str] = Field(default_factory=list)
    sweep: SweepConfig | None = None
    output_dir: str = "results"
