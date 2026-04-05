# Benchmark results & insights

> **Benchmark scope**: 6/9 frameworks (neo4j, datastax, raganything were fixed after the benchmark; only individual tests pass so far).
> **Setup**: Answers from gpt-5.4-mini; cross-model scoring with claude-haiku-4-5.
> **Data**: 3 docs (~7.8K chars), 5 queries, 4 metrics (comprehensiveness, diversity, relevance, faithfulness).
> **Date**: 2026-04-05

---

## 1. Overall ranking (pointwise)

| Rank | Framework | Comp. | Div. | Rel. | Faith. | **Avg** |
|:---:|-----------|:-----:|:-----:|:-----:|:-----:|:------:|
| 1 | **nano-graphrag** | 4.40 | 4.20 | 4.80 | 2.40 | **3.95** |
| 2 | **cognee** | 4.00 | 3.80 | 4.40 | 2.80 | **3.75** |
| 3 | **fast-graphrag** | 3.80 | 4.00 | 4.60 | 2.40 | **3.70** |
| 4 | **lightrag** | 3.60 | 4.20 | 4.60 | 2.00 | **3.60** |
| 5 | **microsoft** | 2.40 | 1.20 | 3.80 | 5.00 | **3.10** |
| 6 | **graphiti** | 2.00 | 3.00 | 3.00 | 1.20 | **2.30** |

> neo4j was omitted in this run due to schema-creation errors.

### Key observations

**nano-graphrag ranks first on answer quality.** Leiden community detection plus community reports produce the richest answers; it is also the slowest.

**Microsoft GraphRAG scores a perfect 5.0 on faithfulness.** It is the only setup that includes source citations (Data: Claims references). Comprehensiveness and diversity lag, so overall rank is fifth.

**Graphiti scores lower on answer quality.** It returns fact lists rather than fluent prose. Query latency (~296 ms) is best-in-class, which hurts under a narrative RAG-quality rubric.

**Faithfulness is generally low (1.2–2.8).** Small corpora invite hallucination. Microsoft’s 5.0 is unique because community reports embed explicit references.

---

## 2. Head-to-head (pairwise)

**Elo-style strength** from pairwise win rates:

```
nano (strongest) > lightrag > fast ≈ cognee > microsoft > graphiti
```

| Matchup | Outcome |
|---------|---------|
| nano vs. everyone else | **Sweep** (20/20 × 5) |
| lightrag vs. fast/cognee/microsoft/graphiti | **Sweep** |
| fast vs. cognee | fast ahead 12:6 |
| fast vs. microsoft | fast ahead 15:5 |
| microsoft vs. graphiti | graphiti slightly ahead 11:9 (!!) |

**Surprise**: Microsoft GraphRAG loses to Graphiti near the bottom of the stack, suggesting community-report pipelines underperform on tiny data.

---

## 3. Speed vs. quality

```
Query latency (fast → slow)     Answer quality (high → low)
─────────────────────          ─────────────────────
1. graphiti    296ms           1. nano       3.95
2. microsoft   865ms           2. cognee     3.75
3. cognee    1,755ms           3. fast       3.70
4. fast      2,787ms           4. lightrag   3.60
5. nano      4,119ms           5. microsoft  3.10
6. lightrag  4,669ms           6. graphiti   2.30
```

**Speed and quality trade off.** The fastest (graphiti) is lowest on quality; the slowest (nano) is highest.

**Exception**: **cognee** sits mid-latency (~1.7 s) with second-best quality — a practical sweet spot.

---

## 4. Framework cheat sheet

| Framework | Strengths | Weaknesses | Good for |
|-----------|-----------|------------|----------|
| **nano-graphrag** | Best quality, detailed answers | Slowest | Quality-first research |
| **cognee** | Speed/quality balance, simple API | Black-box internals | Fast prototyping |
| **fast-graphrag** | Fast indexing, stable | Single retrieval strategy | General experiments |
| **lightrag** | Five flexible search modes | Slower queries | Search-strategy studies |
| **microsoft** | Best source faithfulness | Weak on tiny corpora | Large corpora, audit trails |
| **graphiti** | Fastest queries, temporal KG | Fact-list answers | Agent memory, realtime |

---

## 5. Cross-model methodology

### Why cross-model

- **Reduces self-eval bias**: gpt-5.4-mini generates; claude-haiku scores.
- Same-model eval tends to inflate scores (prior runs were all 5/5).
- Cross-model spreads scores (1.2–5.0).

### Design

```
Generation: gpt-5.4-mini (OpenAI) — cost/latency friendly
Scoring: claude-haiku-4-5 (Anthropic) — different provider, lower bias
Modes: Pointwise (1–5) + Pairwise (head-to-head)
De-biasing: Randomize A/B order in pairwise
```

---

## 6. Dependencies & operational lessons

### Fixes applied

- **DataStax**: chromadb version clash → switched to `InMemoryVectorStore` instead of Chroma.
- **Neo4j**: schema creation on empty DB → provide default schema + `max_tokens` → `max_completion_tokens`.
- **RAG-Anything**: API drift → construct LightRAG instance and pass it through.
- **OpenAI provider**: gpt-5.4-mini tiktoken gap + `max_tokens` change → fallback encoding + `max_completion_tokens`.

### API compatibility (resolved)

| Framework | Issue | Fix |
|-----------|-------|-----|
| LightRAG | Duplicate `model` parameter | `partial` for positional binding |
| Cognee | `search()` signature change | Keyword arguments |
| Graphiti | `OpenAIGenericClient` rename | Drop stale import |
| Microsoft | Embedding persistence error | Handle non-fatally |

### Takeaways

1. **Prefer separate venvs per framework** — one env for everything becomes dependency hell.
2. **Docs ≠ shipped API** — install errors right after `pip install` are common.
3. **Small data flattens differences** — aim for 100+ documents for meaningful comparison.

---

## 7. Status & next experiments

### Verified (9/12)

fast, cognee, microsoft, neo4j, graphiti, lightrag, nano, datastax, raganything — all pass individual tests. The benchmark run above used the first six.

### Not included (3/12)

HippoRAG, LinearRAG, PathRAG — not on PyPI; will add when packages ship.

### Next steps

1. **Full 9-framework benchmark** — rerun including neo4j, datastax, raganything.
2. **Larger corpus (100+ docs)** — effects that vanish on small data may show up at scale.
3. **Multi-hop queries** — e.g. graph reasoning: “What is the tech funded by org X?”
4. **Ollama local models** — repeat experiments without API cost.
5. **Token/cost tracking** — compare real API spend per framework.
6. **Domain focus** — legal/medical corpora to test fit.
