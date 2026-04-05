# GraphRAG Lab

A framework for systematically benchmarking GraphRAG implementations.

Automates indexing → query → LLM-as-judge evaluation under identical conditions across nine frameworks.

## Quick Start

```bash
# Base install
pip install -e "."

# Optional framework extras
pip install -e ".[fast,openai]"           # fast-graphrag only
pip install -e ".[lightrag,nano,openai]"  # LightRAG + nano
pip install -e ".[neo4j,graphiti,openai]" # Neo4j-based (Docker required)
```

```bash
# Configure .env
cp .env.example .env
# Set OPENAI_API_KEY, ANTHROPIC_API_KEY

# Run an experiment
graphrag-lab run configs/experiments/full_benchmark.yaml

# Inspect results
graphrag-lab results list
```

## Supported frameworks (9 working)

| # | Framework | Approach | Search strategy | Install |
|:-:|-----------|----------|-----------------|---------|
| 1 | **Microsoft GraphRAG** | Leiden community detection + summarization | local, global | `.[microsoft]` |
| 2 | **LightRAG** | KG + vector hybrid | local, global, hybrid, naive, mix | `.[lightrag]` |
| 3 | **nano-graphrag** | Lightweight MS GraphRAG-style clone | local, global, naive | `.[nano]` |
| 4 | **fast-graphrag** | PageRank-based | Built-in | `.[fast]` |
| 5 | **Neo4j GraphRAG** | Schema-based KG | vector, vector_cypher, hybrid, text2cypher | `.[neo4j]` |
| 6 | **DataStax Graph RAG** | Metadata graph exploration | eager, mmr | `.[datastax]` |
| 7 | **Cognee** | Auto-optimized ECL | GRAPH_COMPLETION, SUMMARIES | `.[cognee]` |
| 8 | **Graphiti** (Zep) | Temporal-aware KG | hybrid_temporal | `.[graphiti]` |
| 9 | **RAG-Anything** | Multimodal (LightRAG-based) | local, global, hybrid, mix | `.[raganything]` |

> HippoRAG, LinearRAG, and PathRAG are not on PyPI; integration is planned.

## Benchmark summary (6/9 frameworks)

> Answers generated with gpt-5.4-mini; scored with claude-haiku-4-5 in a cross-model setup (reduces self-eval bias).
> neo4j, datastax, and raganything were fixed after the benchmark — unit tests pass; a full benchmark re-run is planned.

| Rank | Framework | Avg score | Query latency | Notes |
|:---:|-----------|:--------:|:--------:|------|
| 1 | **nano-graphrag** | 3.95 | 4.1s | Best quality; Leiden community detection |
| 2 | **cognee** | 3.75 | 1.8s | Strong speed/quality balance |
| 3 | **fast-graphrag** | 3.70 | 2.8s | Fastest indexing |
| 4 | **lightrag** | 3.60 | 4.7s | Five search modes |
| 5 | **microsoft** | 3.10 | 0.9s | Top faithfulness |
| 6 | **graphiti** | 2.30 | 0.3s | Fastest queries; temporal-aware |

> Deeper analysis, pairwise comparisons, and takeaways: **[docs/benchmark.md](docs/benchmark.md)**

## Evaluation

- **Cross-model**: generation (OpenAI) ≠ scoring (Anthropic) to reduce self-eval bias
- **Four metrics**: comprehensiveness, diversity, relevance, faithfulness (1–5)
- **Two modes**: Pointwise (single-answer grading) + Pairwise (head-to-head)

## Project layout

```
src/graphrag_lab/
  config/          # Pydantic schemas + YAML loader
  frameworks/      # Nine framework adapters (shared ABC)
  providers/       # LLM providers (OpenAI, Anthropic, Ollama)
  evaluation/      # LLM-as-judge + cost tracking
  runner/          # Experiment orchestrator + parameter sweeps
  datasets/        # Dataset load/validate
  cli.py           # Click CLI
configs/           # Experiment YAMLs
docs/
  benchmark.md     # Benchmark results & insights
  ecosystem.md     # GraphRAG ecosystem research
```

## Environment

```bash
cp .env.example .env
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

For Neo4j (neo4j, graphiti frameworks):
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/graphraglab -e 'NEO4J_PLUGINS=["apoc"]' neo4j:5
```
