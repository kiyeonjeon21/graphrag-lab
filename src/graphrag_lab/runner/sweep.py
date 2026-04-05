"""Parameter sweep runner: generates Cartesian product of configurations."""

from __future__ import annotations

import copy
import itertools
from typing import Any

import structlog

from graphrag_lab.config.schema import ExperimentConfig
from graphrag_lab.runner.experiment import ExperimentRunner

logger = structlog.get_logger()


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key path."""
    parts = key_path.split(".")
    target = d
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def expand_sweep(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Expand a config with sweep parameters into individual configs."""
    if not config.sweep or not config.sweep.parameters:
        return [config]

    base_dict = config.model_dump()
    params = config.sweep.parameters

    # Generate Cartesian product
    keys = list(params.keys())
    values = list(params.values())
    combinations = list(itertools.product(*values))

    configs = []
    for i, combo in enumerate(combinations):
        variant = copy.deepcopy(base_dict)
        variant.pop("sweep", None)

        # Apply each parameter value
        label_parts = []
        for key, value in zip(keys, combo):
            _set_nested(variant, key, value)
            label_parts.append(f"{key.split('.')[-1]}={value}")

        # Generate unique ID
        variant["id"] = f"{config.id}-sweep-{i:03d}"
        variant["description"] = f"{config.description} [{', '.join(label_parts)}]"

        configs.append(ExperimentConfig(**variant))

    logger.info("sweep_expanded", num_variants=len(configs), parameters=keys)
    return configs


class SweepRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    async def run(self) -> list[dict]:
        configs = expand_sweep(self.config)
        results = []

        for i, variant_config in enumerate(configs):
            logger.info("sweep_variant_start", index=i, total=len(configs), id=variant_config.id)
            runner = ExperimentRunner(variant_config)
            result = await runner.run()
            results.append(result)

        logger.info("sweep_complete", num_variants=len(results))
        return results
