"""LLM-as-judge evaluation for pointwise and pairwise comparisons."""

from __future__ import annotations

import json
import re
import random

import structlog

from graphrag_lab.evaluation.metrics import (
    METRIC_DESCRIPTIONS,
    PairwiseResult,
    PairwiseSummary,
    PointwiseScore,
    MetricsSummary,
)
from graphrag_lab.providers.base import LLMProvider

logger = structlog.get_logger()

POINTWISE_PROMPT = """You are an expert evaluator. Score the answer on this metric:
{metric}: {description}

Question: {question}
Answer: {answer}

Respond with ONLY this JSON, no other text:
{{"score": <int 1-5>, "rationale": "<brief explanation>"}}"""

PAIRWISE_PROMPT = """You are an expert evaluator. Which answer is better on this metric?
{metric}: {description}

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Respond with ONLY this JSON, no other text:
{{"winner": "<a|b|tie>", "rationale": "<brief explanation>"}}"""


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response that may contain extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON in markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try to find any JSON object
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No JSON found", text, 0)


class LLMJudge:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def pointwise_evaluate(
        self, question: str, answer: str, metrics: list[str]
    ) -> list[PointwiseScore]:
        scores = []
        for metric in metrics:
            prompt = POINTWISE_PROMPT.format(
                metric=metric,
                description=METRIC_DESCRIPTIONS.get(metric, ""),
                question=question,
                answer=answer,
            )
            try:
                response = await self.provider.complete(prompt)
                parsed = _extract_json(response.text)
                scores.append(
                    PointwiseScore(
                        metric=metric,
                        score=float(parsed["score"]),
                        rationale=parsed.get("rationale", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("judge_parse_error", metric=metric, error=str(e))
                scores.append(PointwiseScore(metric=metric, score=0.0, rationale=f"Parse error: {e}"))
        return scores

    async def pairwise_evaluate(
        self, question: str, answer_a: str, answer_b: str, metrics: list[str]
    ) -> list[PairwiseResult]:
        # Randomize order to avoid position bias
        swapped = random.random() > 0.5
        if swapped:
            display_a, display_b = answer_b, answer_a
        else:
            display_a, display_b = answer_a, answer_b

        results = []
        for metric in metrics:
            prompt = PAIRWISE_PROMPT.format(
                metric=metric,
                description=METRIC_DESCRIPTIONS.get(metric, ""),
                question=question,
                answer_a=display_a,
                answer_b=display_b,
            )
            try:
                response = await self.provider.complete(prompt)
                parsed = _extract_json(response.text)
                winner = parsed["winner"].lower().strip()

                # Unswap the winner
                if swapped and winner in ("a", "b"):
                    winner = "b" if winner == "a" else "a"

                results.append(
                    PairwiseResult(
                        metric=metric,
                        winner=winner,
                        rationale=parsed.get("rationale", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("judge_parse_error", metric=metric, error=str(e))
                results.append(PairwiseResult(metric=metric, winner="tie", rationale=f"Parse error: {e}"))
        return results

    async def evaluate_all_pointwise(
        self,
        questions: list[str],
        answers_by_framework: dict[str, list[str]],
        metrics: list[str],
    ) -> dict[str, MetricsSummary]:
        summaries: dict[str, MetricsSummary] = {}
        for framework, answers in answers_by_framework.items():
            summary = MetricsSummary()
            for question, answer in zip(questions, answers):
                scores = await self.pointwise_evaluate(question, answer, metrics)
                for s in scores:
                    summary.add_score(s.metric, s.score)
            summaries[framework] = summary
        return summaries

    async def evaluate_all_pairwise(
        self,
        questions: list[str],
        answers_by_framework: dict[str, list[str]],
        metrics: list[str],
    ) -> PairwiseSummary:
        frameworks = list(answers_by_framework.keys())
        pairwise_summary = PairwiseSummary()

        for i in range(len(frameworks)):
            for j in range(i + 1, len(frameworks)):
                fw_a, fw_b = frameworks[i], frameworks[j]
                pair_key = f"{fw_a}_vs_{fw_b}"

                for question, ans_a, ans_b in zip(
                    questions,
                    answers_by_framework[fw_a],
                    answers_by_framework[fw_b],
                ):
                    results = await self.pairwise_evaluate(question, ans_a, ans_b, metrics)
                    for r in results:
                        pairwise_summary.add_result(pair_key, r)

        return pairwise_summary
