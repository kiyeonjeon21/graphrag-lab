"""Evaluation metrics definitions and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev


METRIC_NAMES = ["comprehensiveness", "diversity", "relevance", "faithfulness"]

METRIC_DESCRIPTIONS = {
    "comprehensiveness": "Does the answer cover all relevant aspects of the question?",
    "diversity": "Does the answer provide varied perspectives and information?",
    "relevance": "Is the answer directly relevant to what the question asks?",
    "faithfulness": "Is the answer grounded in the provided context without hallucination?",
}


@dataclass
class PointwiseScore:
    metric: str
    score: float  # 1-5
    rationale: str = ""


@dataclass
class PairwiseResult:
    metric: str
    winner: str  # "a", "b", or "tie"
    rationale: str = ""


@dataclass
class MetricsSummary:
    scores: dict[str, list[float]] = field(default_factory=dict)

    def add_score(self, metric: str, score: float) -> None:
        self.scores.setdefault(metric, []).append(score)

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for metric, values in self.scores.items():
            result[metric] = {
                "mean": round(mean(values), 3),
                "std": round(stdev(values), 3) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        return result


@dataclass
class PairwiseSummary:
    results: dict[str, list[PairwiseResult]] = field(default_factory=dict)

    def add_result(self, pair_key: str, result: PairwiseResult) -> None:
        self.results.setdefault(pair_key, []).append(result)

    def win_rates(self) -> dict[str, dict[str, dict[str, int]]]:
        output = {}
        for pair_key, results in self.results.items():
            by_metric: dict[str, dict[str, int]] = {}
            for r in results:
                counts = by_metric.setdefault(r.metric, {"wins_a": 0, "wins_b": 0, "ties": 0})
                if r.winner == "a":
                    counts["wins_a"] += 1
                elif r.winner == "b":
                    counts["wins_b"] += 1
                else:
                    counts["ties"] += 1
            output[pair_key] = by_metric
        return output
