"""Tests for sweep runner logic."""

from __future__ import annotations

from graphrag_lab.config.schema import ExperimentConfig
from graphrag_lab.runner.sweep import expand_sweep


def test_expand_sweep_no_sweep():
    config = ExperimentConfig(
        name="test",
        dataset={"name": "test", "path": "/tmp/test"},
        frameworks=[{"name": "microsoft"}],
    )
    result = expand_sweep(config)
    assert len(result) == 1


def test_expand_sweep_cartesian():
    config = ExperimentConfig(
        name="test",
        dataset={"name": "test", "path": "/tmp/test"},
        frameworks=[{"name": "microsoft"}],
        sweep={"parameters": {
            "llm.model": ["gpt-4o", "gpt-4o-mini"],
            "chunking.chunk_size": [600, 1200],
        }},
    )
    result = expand_sweep(config)
    # 2 models x 2 chunk sizes = 4 variants
    assert len(result) == 4
    assert all(isinstance(c, ExperimentConfig) for c in result)
    assert all(c.sweep is None for c in result)

    models = [c.llm.model for c in result]
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
