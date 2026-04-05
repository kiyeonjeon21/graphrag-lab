"""Tests for evaluation metrics and cost tracking."""

from __future__ import annotations

from graphrag_lab.evaluation.cost import compute_cost, build_cost_report
from graphrag_lab.evaluation.metrics import MetricsSummary, PairwiseSummary, PairwiseResult
from graphrag_lab.frameworks.base import CostInfo, TokenUsage


def test_metrics_summary():
    summary = MetricsSummary()
    summary.add_score("relevance", 4.0)
    summary.add_score("relevance", 5.0)
    summary.add_score("relevance", 3.0)

    result = summary.summary()
    assert result["relevance"]["mean"] == 4.0
    assert result["relevance"]["min"] == 3.0
    assert result["relevance"]["max"] == 5.0
    assert result["relevance"]["n"] == 3


def test_pairwise_summary():
    summary = PairwiseSummary()
    summary.add_result("a_vs_b", PairwiseResult(metric="relevance", winner="a"))
    summary.add_result("a_vs_b", PairwiseResult(metric="relevance", winner="b"))
    summary.add_result("a_vs_b", PairwiseResult(metric="relevance", winner="a"))

    rates = summary.win_rates()
    assert rates["a_vs_b"]["relevance"]["wins_a"] == 2
    assert rates["a_vs_b"]["relevance"]["wins_b"] == 1
    assert rates["a_vs_b"]["relevance"]["ties"] == 0


def test_compute_cost_openai():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    cost = compute_cost(usage, "gpt-4o")
    # (1000/1M) * 2.50 + (500/1M) * 10.00 = 0.0025 + 0.005 = 0.0075
    assert abs(cost - 0.0075) < 1e-6


def test_compute_cost_ollama():
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    cost = compute_cost(usage, "llama3.1:70b")
    assert cost == 0.0


def test_build_cost_report():
    cost_info = CostInfo(
        indexing_tokens=TokenUsage(prompt_tokens=10000, completion_tokens=5000),
        indexing_latency_s=30.0,
    )
    cost_info.query_tokens["q1"] = TokenUsage(prompt_tokens=500, completion_tokens=200)
    cost_info.query_latencies["q1"] = 1.5

    report = build_cost_report(cost_info, "gpt-4o-mini")
    assert report.indexing_latency_s == 30.0
    assert report.total_cost_usd > 0

    d = report.to_dict()
    assert "indexing" in d
    assert "queries" in d
    assert d["model"] == "gpt-4o-mini"
