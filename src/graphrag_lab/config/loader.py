"""YAML configuration loading with base + experiment merge."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from graphrag_lab.config.schema import ExperimentConfig

# Resolve project root: walk up from this file until we find pyproject.toml
def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_experiment(experiment_path: str | Path, base_path: str | Path | None = None) -> ExperimentConfig:
    """Load an experiment config by merging base.yaml with the experiment YAML.

    Args:
        experiment_path: Path to the experiment YAML file.
        base_path: Path to base.yaml. Defaults to configs/base.yaml in the project root.

    Returns:
        Validated ExperimentConfig.
    """
    if base_path is None:
        base_path = PROJECT_ROOT / "configs" / "base.yaml"

    base_path = Path(base_path)
    experiment_path = Path(experiment_path)

    base = _load_yaml(base_path) if base_path.exists() else {}
    experiment = _load_yaml(experiment_path)

    merged = deep_merge(base, experiment)
    return ExperimentConfig(**merged)


def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-separated key=value overrides to a config dict.

    Example: --set llm.model=gpt-4o → sets config_dict["llm"]["model"] = "gpt-4o"
    """
    result = copy.deepcopy(config_dict)
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _coerce_value(value.strip())
    return result


def _coerce_value(value: str) -> Any:
    """Try to coerce a string value to int, float, bool, or leave as str."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
