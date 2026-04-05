"""Cost tracking and calculation for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphrag_lab.frameworks.base import CostInfo, TokenUsage

# Pricing per 1M tokens (USD)
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # Local (Ollama)
    "ollama": {"input": 0.0, "output": 0.0},
}


def compute_cost(token_usage: TokenUsage, model: str) -> float:
    """Compute USD cost from token usage and model name."""
    pricing = PRICING.get(model)
    if pricing is None:
        # Fallback for Ollama or unknown models
        if "ollama" in model.lower() or model.startswith("llama") or model.startswith("mistral"):
            return 0.0
        return 0.0

    input_cost = (token_usage.prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (token_usage.completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class CostReport:
    model: str = ""
    indexing_tokens: TokenUsage = field(default_factory=TokenUsage)
    indexing_latency_s: float = 0.0
    indexing_cost_usd: float = 0.0
    query_tokens: dict[str, TokenUsage] = field(default_factory=dict)
    query_latencies: dict[str, float] = field(default_factory=dict)
    query_costs: dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "indexing": {
                "tokens": {
                    "prompt": self.indexing_tokens.prompt_tokens,
                    "completion": self.indexing_tokens.completion_tokens,
                    "total": self.indexing_tokens.total_tokens,
                },
                "latency_s": self.indexing_latency_s,
                "cost_usd": self.indexing_cost_usd,
            },
            "queries": {
                q: {
                    "tokens": {
                        "prompt": t.prompt_tokens,
                        "completion": t.completion_tokens,
                        "total": t.total_tokens,
                    },
                    "latency_ms": self.query_latencies.get(q, 0.0),
                    "cost_usd": self.query_costs.get(q, 0.0),
                }
                for q, t in self.query_tokens.items()
            },
            "total_cost_usd": self.total_cost_usd,
        }


def build_cost_report(cost_info: CostInfo, model: str) -> CostReport:
    """Build a CostReport from a framework's CostInfo."""
    report = CostReport(model=model)
    report.indexing_tokens = cost_info.indexing_tokens
    report.indexing_latency_s = cost_info.indexing_latency_s
    report.indexing_cost_usd = compute_cost(cost_info.indexing_tokens, model)

    total = report.indexing_cost_usd
    for query, tokens in cost_info.query_tokens.items():
        report.query_tokens[query] = tokens
        report.query_latencies[query] = cost_info.query_latencies.get(query, 0.0)
        cost = compute_cost(tokens, model)
        report.query_costs[query] = cost
        total += cost

    report.total_cost_usd = round(total, 6)
    return report
