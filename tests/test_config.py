"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_lab.config.loader import apply_overrides, deep_merge, load_experiment
from graphrag_lab.config.schema import ExperimentConfig


def test_deep_merge_simple():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    deep_merge(base, override)
    assert base["a"]["b"] == 1


def test_load_experiment(configs_dir: Path):
    config = load_experiment(
        configs_dir / "experiments" / "framework_comparison.yaml",
        base_path=configs_dir / "base.yaml",
    )
    assert isinstance(config, ExperimentConfig)
    assert config.name == "Framework Comparison"
    assert len(config.frameworks) == 5
    assert config.llm.model == "gpt-5.4-mini"
    assert config.evaluation.metrics == ["comprehensiveness", "diversity", "relevance", "faithfulness"]


def test_apply_overrides():
    data = {"llm": {"model": "gpt-5.4-mini", "temperature": 0.0}}
    result = apply_overrides(data, ["llm.model=gpt-4o", "llm.temperature=0.5"])
    assert result["llm"]["model"] == "gpt-4o"
    assert result["llm"]["temperature"] == 0.5


def test_experiment_config_defaults():
    config = ExperimentConfig(
        name="test",
        dataset={"name": "test", "path": "/tmp/test"},
        frameworks=[{"name": "microsoft"}],
    )
    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-5.4-mini"
    assert config.chunking.strategy == "fixed"
    assert config.cost_tracking.enabled is True
    assert config.id.startswith("exp-")


def test_all_framework_names():
    """Verify all 12 framework names are valid."""
    all_names = [
        "microsoft", "lightrag", "nano", "fast", "neo4j", "datastax",
        "cognee", "graphiti", "raganything", "hipporag", "linearrag", "pathrag",
    ]
    for name in all_names:
        config = ExperimentConfig(
            name="test",
            dataset={"name": "test", "path": "/tmp/test"},
            frameworks=[{"name": name}],
        )
        assert config.frameworks[0].name == name


def test_load_all_frameworks_experiment(configs_dir: Path):
    config = load_experiment(
        configs_dir / "experiments" / "all_frameworks.yaml",
        base_path=configs_dir / "base.yaml",
    )
    assert len(config.frameworks) == 12


def test_experiment_config_with_sweep():
    config = ExperimentConfig(
        name="test",
        dataset={"name": "test", "path": "/tmp/test"},
        frameworks=[{"name": "microsoft"}],
        sweep={"parameters": {"llm.model": ["gpt-4o", "gpt-5.4-mini"]}},
    )
    assert config.sweep is not None
    assert len(config.sweep.parameters["llm.model"]) == 2
