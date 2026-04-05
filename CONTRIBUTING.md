# Contributing to GraphRAG Lab

Thanks for your interest! This project compares GraphRAG frameworks, and contributions are welcome.

## Getting started

```bash
git clone https://github.com/kiyeonjeon21/graphrag-lab.git
cd graphrag-lab
pip install -e ".[dev]"
cp .env.example .env
# Fill in OPENAI_API_KEY (required), ANTHROPIC_API_KEY (for evaluation)
```

Run tests:
```bash
pytest tests/
```

## Ways to contribute

### Add a new framework adapter

1. Create `src/graphrag_lab/frameworks/<name>.py` implementing `GraphRAGFramework` ABC
2. Implement `build_index()`, `query()`, `get_cost_info()`
3. Add the framework name to `FrameworkConfig.name` Literal in `config/schema.py`
4. Register the lazy import in `runner/experiment.py` `_ensure_registry()`
5. Add optional dependency in `pyproject.toml`
6. Test with: `graphrag-lab run configs/experiments/quick_test.yaml --set frameworks.0.name=<name>`

See existing adapters (e.g. `fast.py`, `cognee.py`) for patterns.

### Improve evaluation

- Better judge prompts for more discriminative scoring
- New metrics beyond the current four
- Structured output support for judge LLM calls

### Add datasets

- Place files in `data/<dataset_name>/`
- Supported formats: `.txt`, `.md`, `.html`
- Create a matching experiment config in `configs/experiments/`

### Fix bugs or improve docs

Always welcome. Check [open issues](https://github.com/kiyeonjeon21/graphrag-lab/issues) for ideas.

## Code style

- Python 3.11+, type hints
- `ruff` for linting (included in `.[dev]`)
- Follow existing patterns — no unnecessary abstractions

## Pull requests

1. Fork and create a feature branch
2. Keep changes focused — one PR per concern
3. Include test results if changing framework adapters
4. Update docs if behavior changes

## Project structure

```
src/graphrag_lab/
  config/       # Pydantic schemas + YAML loader
  frameworks/   # One file per framework adapter
  providers/    # LLM providers (OpenAI, Anthropic, Ollama)
  evaluation/   # LLM-as-judge + cost tracking
  runner/       # Experiment orchestrator + sweep
  cli.py        # CLI entry point
configs/        # YAML experiment definitions
docs/           # Benchmark results + ecosystem research
```
