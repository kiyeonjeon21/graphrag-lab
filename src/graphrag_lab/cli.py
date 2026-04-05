"""CLI entry point for graphrag-lab."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from graphrag_lab.utils.logging import setup_logging


@click.group()
@click.option("--json-log", is_flag=True, help="Output JSON-formatted logs")
def main(json_log: bool) -> None:
    """GraphRAG Lab - Experiment framework for comparing GraphRAG implementations."""
    load_dotenv()
    setup_logging(json_output=json_log)

    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Warning: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your keys.", err=True)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--set", "overrides", multiple=True, help="Override config values (e.g., --set llm.model=gpt-4o)")
def run(config_path: str, overrides: tuple[str, ...]) -> None:
    """Run a single experiment from a YAML config file."""
    from graphrag_lab.config.loader import apply_overrides, load_experiment
    from graphrag_lab.runner.experiment import ExperimentRunner
    from graphrag_lab.utils.io import load_yaml

    if overrides:
        raw = load_yaml(config_path)
        raw = apply_overrides(raw, list(overrides))
        from graphrag_lab.config.schema import ExperimentConfig
        config = ExperimentConfig(**raw)
    else:
        config = load_experiment(config_path)

    click.echo(f"Running experiment: {config.name} ({config.id})")

    runner = ExperimentRunner(config)
    result = asyncio.run(runner.run())

    click.echo(f"\nExperiment complete: {config.id}")
    click.echo(f"Results saved to: {runner.output_path}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--set", "overrides", multiple=True, help="Override config values")
def sweep(config_path: str, overrides: tuple[str, ...]) -> None:
    """Run a parameter sweep from a YAML config file."""
    from graphrag_lab.config.loader import load_experiment
    from graphrag_lab.runner.sweep import SweepRunner

    config = load_experiment(config_path)
    click.echo(f"Running sweep: {config.name}")

    runner = SweepRunner(config)
    results = asyncio.run(runner.run())

    click.echo(f"\nSweep complete: {len(results)} variants executed")


@main.group()
def results() -> None:
    """View and compare experiment results."""


@results.command(name="list")
@click.option("--output-dir", default="results", help="Results directory")
def list_results(output_dir: str) -> None:
    """List all experiment results."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        click.echo("No results directory found.")
        return

    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            config_file = exp_dir / "config_snapshot.yaml"
            if config_file.exists():
                from graphrag_lab.utils.io import load_yaml
                config = load_yaml(config_file)
                click.echo(f"  {exp_dir.name}: {config.get('name', 'N/A')}")
            else:
                click.echo(f"  {exp_dir.name}: (no config snapshot)")


@results.command()
@click.argument("run_ids", nargs=2)
@click.option("--output-dir", default="results", help="Results directory")
def compare(run_ids: tuple[str, str], output_dir: str) -> None:
    """Compare two experiment runs side by side."""
    from graphrag_lab.utils.io import load_json

    results_dir = Path(output_dir)

    for run_id in run_ids:
        metrics_file = results_dir / run_id / "metrics.json"
        cost_file = results_dir / run_id / "cost_report.json"

        click.echo(f"\n--- {run_id} ---")
        if metrics_file.exists():
            metrics = load_json(metrics_file)
            if "pointwise" in metrics:
                for fw, scores in metrics["pointwise"].items():
                    click.echo(f"  [{fw}]")
                    for metric, vals in scores.items():
                        click.echo(f"    {metric}: {vals['mean']:.2f} (+/- {vals['std']:.2f})")

        if cost_file.exists():
            costs = load_json(cost_file)
            for fw, report in costs.items():
                click.echo(f"  [{fw}] Total cost: ${report['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
