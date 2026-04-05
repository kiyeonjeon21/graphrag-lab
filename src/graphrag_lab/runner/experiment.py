"""Experiment orchestration: index, query, evaluate, save."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog

from graphrag_lab.config.schema import ExperimentConfig, LLMProviderConfig
from graphrag_lab.datasets.manager import DatasetManager
from graphrag_lab.evaluation.cost import CostReport, build_cost_report
from graphrag_lab.evaluation.judge import LLMJudge
from graphrag_lab.evaluation.metrics import MetricsSummary, PairwiseSummary
from graphrag_lab.frameworks.base import GraphRAGFramework, QueryResult
from graphrag_lab.utils.io import save_json, save_yaml

logger = structlog.get_logger()

# Framework registry
_FRAMEWORK_CLASSES: dict[str, type[GraphRAGFramework]] = {}


def _ensure_registry() -> None:
    if _FRAMEWORK_CLASSES:
        return
    # Lazy imports to avoid requiring all framework dependencies
    try:
        from graphrag_lab.frameworks.microsoft import MicrosoftGraphRAG
        _FRAMEWORK_CLASSES["microsoft"] = MicrosoftGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.lightrag import LightRAGFramework
        _FRAMEWORK_CLASSES["lightrag"] = LightRAGFramework
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.nano import NanoGraphRAG
        _FRAMEWORK_CLASSES["nano"] = NanoGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.fast import FastGraphRAG
        _FRAMEWORK_CLASSES["fast"] = FastGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.neo4j import Neo4jGraphRAG
        _FRAMEWORK_CLASSES["neo4j"] = Neo4jGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.datastax import DataStaxGraphRAG
        _FRAMEWORK_CLASSES["datastax"] = DataStaxGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.cognee import CogneeGraphRAG
        _FRAMEWORK_CLASSES["cognee"] = CogneeGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.graphiti import GraphitiGraphRAG
        _FRAMEWORK_CLASSES["graphiti"] = GraphitiGraphRAG
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.raganything import RAGAnythingFramework
        _FRAMEWORK_CLASSES["raganything"] = RAGAnythingFramework
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.hipporag import HippoRAGFramework
        _FRAMEWORK_CLASSES["hipporag"] = HippoRAGFramework
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.linearrag import LinearRAGFramework
        _FRAMEWORK_CLASSES["linearrag"] = LinearRAGFramework
    except ImportError:
        pass
    try:
        from graphrag_lab.frameworks.pathrag import PathRAGFramework
        _FRAMEWORK_CLASSES["pathrag"] = PathRAGFramework
    except ImportError:
        pass


def get_framework(name: str, llm_config: LLMProviderConfig) -> GraphRAGFramework:
    _ensure_registry()
    cls = _FRAMEWORK_CLASSES.get(name)
    if cls is None:
        available = list(_FRAMEWORK_CLASSES.keys())
        raise ValueError(f"Framework '{name}' not available. Installed: {available}")
    return cls(llm_config)


def _get_provider(config: LLMProviderConfig):
    """Create an LLM provider for evaluation (judge)."""
    if config.provider == "openai":
        from graphrag_lab.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    elif config.provider == "anthropic":
        from graphrag_lab.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif config.provider == "ollama":
        from graphrag_lab.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_path = Path(config.output_dir) / config.id

    async def run(self) -> dict[str, Any]:
        logger.info("experiment_start", id=self.config.id, name=self.config.name)

        # 1. Load dataset
        documents = DatasetManager.load(self.config.dataset)

        # 2. Run each framework
        all_results: dict[str, list[QueryResult]] = {}
        cost_reports: dict[str, CostReport] = {}

        for fw_config in self.config.frameworks:
            fw_name = fw_config.name
            logger.info("framework_start", framework=fw_name)

            try:
                framework = get_framework(fw_name, self.config.llm)
            except ValueError as e:
                logger.warning("framework_skipped", framework=fw_name, reason=str(e))
                continue

            try:
                # Build index
                index = await framework.build_index(documents, fw_config)
                logger.info("index_built", framework=fw_name)

                # Run queries
                fw_results: list[QueryResult] = []
                for query in self.config.queries:
                    result = await framework.query(query, index, fw_config.search_type)
                    fw_results.append(result)
                    logger.info("query_done", framework=fw_name, query=query[:50], latency_ms=result.latency_ms)

                all_results[fw_name] = fw_results

                # Collect cost info
                if self.config.cost_tracking.enabled:
                    cost_info = framework.get_cost_info()
                    cost_reports[fw_name] = build_cost_report(cost_info, self.config.llm.model)

            except Exception as e:
                logger.error("framework_failed", framework=fw_name, error=str(e))
                continue

        # 3. Evaluate
        eval_report = await self._evaluate(all_results)

        # 4. Save results
        self._save(all_results, eval_report, cost_reports)

        logger.info("experiment_complete", id=self.config.id)
        return {
            "experiment_id": self.config.id,
            "evaluation": eval_report,
            "costs": {k: v.to_dict() for k, v in cost_reports.items()},
        }

    async def _evaluate(self, all_results: dict[str, list[QueryResult]]) -> dict[str, Any]:
        eval_config = self.config.evaluation
        if not eval_config.judge_model:
            logger.info("skipping_evaluation", reason="no judge_model configured")
            return {}

        provider = _get_provider(eval_config.judge_model)
        judge = LLMJudge(provider)

        questions = self.config.queries
        answers_by_framework = {
            fw: [r.answer for r in results] for fw, results in all_results.items()
        }

        # Pointwise evaluation
        pointwise = await judge.evaluate_all_pointwise(questions, answers_by_framework, eval_config.metrics)
        pointwise_dict = {fw: summary.summary() for fw, summary in pointwise.items()}

        # Pairwise evaluation (only if multiple frameworks)
        pairwise_dict = {}
        if len(all_results) > 1:
            pairwise = await judge.evaluate_all_pairwise(questions, answers_by_framework, eval_config.metrics)
            pairwise_dict = pairwise.win_rates()

        return {"pointwise": pointwise_dict, "pairwise": pairwise_dict}

    def _save(
        self,
        all_results: dict[str, list[QueryResult]],
        eval_report: dict[str, Any],
        cost_reports: dict[str, CostReport],
    ) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Config snapshot
        save_yaml(self.config.model_dump(), self.output_path / "config_snapshot.yaml")

        # Evaluation metrics
        if eval_report:
            save_json(eval_report, self.output_path / "metrics.json")

        # Cost reports
        if cost_reports:
            save_json(
                {fw: report.to_dict() for fw, report in cost_reports.items()},
                self.output_path / "cost_report.json",
            )

        # Raw results
        raw = {}
        for fw, results in all_results.items():
            raw[fw] = [
                {
                    "query": q,
                    "answer": r.answer,
                    "latency_ms": r.latency_ms,
                    "token_usage": {
                        "prompt": r.token_usage.prompt_tokens,
                        "completion": r.token_usage.completion_tokens,
                    },
                }
                for q, r in zip(self.config.queries, results)
            ]
        save_json(raw, self.output_path / "raw_results.json")

        logger.info("results_saved", path=str(self.output_path))
