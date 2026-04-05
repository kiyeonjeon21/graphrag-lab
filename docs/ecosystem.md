# GraphRAG ecosystem overview (2025–2026)

## 1. What is GraphRAG?

Technique from Microsoft Research (April 2024; Darren Edge et al.). **Automatically builds a knowledge graph (KG) from unstructured text** and uses community detection plus summarization to outperform vanilla vector RAG on **complex multi-hop questions and global summarization**.

### Traditional RAG vs. GraphRAG

| Aspect | Traditional RAG | GraphRAG |
|--------|-----------------|----------|
| Retrieval unit | Text chunks (vector similarity) | Entities / relations / communities |
| Multi-hop reasoning | Weak (~45–50%) | Strong (~80–85%) |
| Global summaries | Not supported | Community-driven summaries |
| Indexing cost | Low | High (many LLM calls) |
| Single-hop QA | Strong (68.18% F1) | Slightly weaker (65.44% F1) |
| Latency | Baseline | ~2.3–2.4× higher |

### Benchmark highlights

- **Multi-hop QA**: GraphRAG **~3.2× accuracy** vs. vector RAG on MultiHop-RAG (+5.4 pp).
- **Enterprise schema queries**: GraphRAG **90%+ accuracy** vs. ~3.4× gap for vector search alone.
- **Single-hop QA**: Traditional RAG wins (+2.74 pp).
- Source: [RAG vs. GraphRAG: A Systematic Evaluation (arXiv 2502.11371)](https://arxiv.org/abs/2502.11371)

## 2. Framework deep dive

### 2.1 Microsoft GraphRAG (v3.0.8)

- **GitHub**: https://github.com/microsoft/graphrag (31.6K stars)
- **Architecture**: TextUnit → entity/relation extraction → Leiden communities → community summaries → embeddings
- **Seven knowledge artifacts**: Document, TextUnit, Entity, Relationship, Covariate, Community, Community Report
- **Four search modes**: Local (entity-centric), Global (Map-Reduce over communities), DRIFT (Global+Local hybrid), LazyGraphRAG (~99.9% indexing savings)
- **DRIFT Search**: vs. local — ~78% win rate on comprehensiveness, ~81% on diversity
- **LazyGraphRAG**: ~99.9% indexing cost cut, ~700× query cost reduction

### 2.2 LightRAG (~28K stars)

- **GitHub**: https://github.com/HKUDS/LightRAG
- **EMNLP 2025**
- **Pros**: Incremental updates, fast and cheap, many vector DB backends
- **2026 additions**: OpenSearch backend, setup wizard, RAGAS integration, Langfuse tracing

### 2.3 nano-graphrag (~3.7K stars)

- **GitHub**: https://github.com/gusye1234/nano-graphrag
- ~1,100-line minimal core, async support, incremental insert
- **Backends**: NetworkX/Neo4j (graph), nano-vectordb/hnswlib/milvus/faiss (vectors)
- Ideal for teaching and prototyping

### 2.4 fast-graphrag (~3.8K stars)

- **GitHub**: https://github.com/circlemind-ai/fast-graphrag
- Claims **27× faster than MS GraphRAG** with **40%+ accuracy gain**
- Cost: $0.08 vs. GraphRAG $0.48 (**~6× cheaper**)
- PageRank-based, fully async, realtime incremental updates

### 2.5 Neo4j GraphRAG Python (v1.14.1)

- **GitHub**: https://github.com/neo4j/neo4j-graphrag-python (1.1K stars)
- **End-to-end**: SimpleKGPipeline (indexing) + six retrievers + GraphRAG (generation)
- **Retrieval**: Vector, VectorCypher, Hybrid, HybridCypher, Text2Cypher, ToolsRetriever
- **Schema-first KG**: typed entities/properties/patterns; optional auto schema extraction
- **No community detection** (no Global Search)

### 2.6 DataStax Graph RAG

- **GitHub**: https://github.com/datastax/graph-rag
- **Idea**: Graph traversal over vector-store metadata without a separate graph DB
- **Package**: `langchain-graph-retriever`
- **Strategies**: Eager (BFS), MMR (relevance + diversity), Scored (custom)

### 2.7 Cognee (~12K stars)

- **GitHub**: https://github.com/topoteretes/cognee
- **Six-line API**: `add() → cognify() → search()`
- **Auto-tuning**: chunk size, LLM choice, data model
- **30+ source formats**

### 2.8 Graphiti / Zep (~24K stars)

- **GitHub**: https://github.com/getzep/graphiti
- **Temporal KG**: facts carry `valid_from` / `valid_to`
- Hybrid: semantic embeddings + BM25 + graph walks
- P95 latency ~300 ms, MCP server support

### 2.9 RAG-Anything (~14.7K stars)

- **GitHub**: https://github.com/HKUDS/RAG-Anything
- From the LightRAG team (HKUDS)
- **Multimodal**: text + images + tables + math in one pipeline

### 2.10 HippoRAG (~3.3K stars)

- **GitHub**: https://github.com/OSU-NLP-Group/HippoRAG
- **NeurIPS’24** (v1), **ICML’25** (v2)
- Hippocampus-inspired memory; KG + Personalized PageRank

### 2.11 LinearRAG (~445 stars)

- **GitHub**: https://github.com/DEEP-PolyU/LinearRAG
- **ICLR 2026**
- **Zero indexing LLM tokens**: Tri-Graph from lightweight NER + semantic linking only

### 2.12 PathRAG (~348 stars)

- **GitHub**: https://github.com/BUPT-GAMMA/PathRAG
- **EDBT 2025**
- Key paths between related nodes in the KG, flow-based pruning

## 3. Other frameworks to watch

### 3.1 Academic (ICLR/AAAI/NAACL 2025–2026)

| Project | Stars | Idea | Paper | GitHub |
|---------|:-----:|------|------|--------|
| **Youtu-GraphRAG** (Tencent) | ~1.1K | Schema-based agentic flow; ~33.6% lower cost | ICLR 2026 | [TencentCloudADP/youtu-graphrag](https://github.com/TencentCloudADP/youtu-graphrag) |
| **LogicRAG** | ~188 | Dynamic logic-dependency graph at query time | AAAI 2026 | [chensyCN/LogicRAG](https://github.com/chensyCN/LogicRAG) |
| **StructRAG** | ~161 | Picks optimal structure per question | ICLR 2025 | [icip-cas/StructRAG](https://github.com/icip-cas/StructRAG) |
| **MedGraphRAG** | ~752 | Medical-domain triple graph (three tiers) | ACL 2025 | [ImprintLab/Medical-Graph-RAG](https://github.com/ImprintLab/Medical-Graph-RAG) |

### 3.2 Platforms / integrations

| Project | Stars | Role | GitHub |
|---------|:-----:|------|--------|
| **R2R** (SciPhi) | ~7.7K | Production-grade all-in-one RAG | [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) |
| **LlamaIndex PropertyGraph** | ~40K+ | Flexible KG extractors + retrievers | [run-llama/llama_index](https://github.com/run-llama/llama_index) |
| **Kotaemon** | ~25K | RAG toolkit with Gradio UI | [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) |
| **FalkorDB GraphRAG-SDK** | ~593 | GraphBLAS-accelerated graph DB | [FalkorDB/GraphRAG-SDK](https://github.com/FalkorDB/GraphRAG-SDK) |

## 4. 2025–2026 trends

| Trend | Example projects | Core idea |
|--------|------------------|-----------|
| **Zero indexing cost** | LinearRAG, LazyGraphRAG | Build graphs without LLM calls, or defer work to query time |
| **Agentic GraphRAG** | Youtu-GraphRAG, Agent-G | Agents autonomously construct/search KGs |
| **Temporal graphs** | Graphiti | Dynamic KGs that track how facts change over time |
| **Multimodal** | RAG-Anything | Unified graph RAG over text, images, tables, and math |
| **Cognitive-science inspired** | HippoRAG 2 | Models inspired by human memory |
| **Domain-specific** | MedGraphRAG | Tuning for verticals such as healthcare |

## 5. References

- [From Local to Global: A Graph RAG Approach (Microsoft, 2024.04)](https://arxiv.org/abs/2404.16130)
- [RAG vs. GraphRAG: A Systematic Evaluation (2025)](https://arxiv.org/abs/2502.11371)
- [DRIFT Search (Microsoft, 2024.10)](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [LazyGraphRAG (Microsoft, 2024.11)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [LightRAG (EMNLP 2025)](https://github.com/HKUDS/LightRAG)
- [HippoRAG (NeurIPS 2024)](https://github.com/OSU-NLP-Group/HippoRAG)
- [LinearRAG (ICLR 2026)](https://github.com/DEEP-PolyU/LinearRAG)
- [PathRAG (EDBT 2025)](https://arxiv.org/abs/2502.14902)
- [Graphiti Temporal KG (arXiv 2501.13956)](https://arxiv.org/abs/2501.13956)
