#!/usr/bin/env python
"""Smoke-test benchmark: run each implemented framework on sample data.

Usage:
    python scripts/benchmark_smoke.py                    # all frameworks
    python scripts/benchmark_smoke.py lightrag fast      # specific frameworks
    python scripts/benchmark_smoke.py --query "custom question here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Suppress noisy library logs
logging.disable(logging.WARNING)

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from graphrag_lab.config.schema import DatasetConfig, FrameworkConfig, LLMProviderConfig
from graphrag_lab.datasets.manager import DatasetManager

# Frameworks and their default search types
FRAMEWORK_DEFAULTS: dict[str, str | None] = {
    "fast": None,
    "cognee": None,
    "microsoft": "local",
    "neo4j": "local",
    "graphiti": None,
    "lightrag": "hybrid",
    "nano": "local",
    "datastax": None,
    "raganything": "hybrid",
}

DEFAULT_QUERY = "What are the main themes discussed across all documents?"


def _get_framework_instance(name: str, llm_config: LLMProviderConfig):
    """Lazy-import and instantiate a framework adapter."""
    if name == "lightrag":
        from graphrag_lab.frameworks.lightrag import LightRAGFramework
        return LightRAGFramework(llm_config)
    elif name == "fast":
        from graphrag_lab.frameworks.fast import FastGraphRAG
        return FastGraphRAG(llm_config)
    elif name == "nano":
        from graphrag_lab.frameworks.nano import NanoGraphRAG
        return NanoGraphRAG(llm_config)
    elif name == "microsoft":
        from graphrag_lab.frameworks.microsoft import MicrosoftGraphRAG
        return MicrosoftGraphRAG(llm_config)
    elif name == "neo4j":
        from graphrag_lab.frameworks.neo4j import Neo4jGraphRAG
        return Neo4jGraphRAG(llm_config)
    elif name == "cognee":
        from graphrag_lab.frameworks.cognee import CogneeGraphRAG
        return CogneeGraphRAG(llm_config)
    elif name == "graphiti":
        from graphrag_lab.frameworks.graphiti import GraphitiGraphRAG
        return GraphitiGraphRAG(llm_config)
    elif name == "datastax":
        from graphrag_lab.frameworks.datastax import DataStaxGraphRAG
        return DataStaxGraphRAG(llm_config)
    elif name == "raganything":
        from graphrag_lab.frameworks.raganything import RAGAnythingFramework
        return RAGAnythingFramework(llm_config)
    else:
        raise ValueError(f"Unknown framework: {name}")


async def run_single(
    name: str,
    llm_config: LLMProviderConfig,
    docs: list[str],
    query: str,
) -> dict:
    """Run index + query for a single framework, return result dict."""
    search_type = FRAMEWORK_DEFAULTS.get(name)
    fw_config = FrameworkConfig(name=name, search_type=search_type)

    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")

    try:
        fw = _get_framework_instance(name, llm_config)

        # Index
        print(f"  Indexing {len(docs)} documents...")
        t0 = time.perf_counter()
        index = await fw.build_index(docs, fw_config)
        index_s = time.perf_counter() - t0
        print(f"  Index built in {index_s:.1f}s")

        # Query
        print(f"  Querying: {query[:60]}...")
        t0 = time.perf_counter()
        result = await fw.query(query, index, search_type)
        query_ms = (time.perf_counter() - t0) * 1000
        print(f"  Answer ({query_ms:.0f}ms): {result.answer[:200]}...")

        return {
            "status": "ok",
            "index_s": round(index_s, 1),
            "query_ms": round(query_ms, 0),
            "search_type": search_type,
            "answer_snippet": result.answer[:300],
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="Smoke-test benchmark")
    parser.add_argument("frameworks", nargs="*", help="Frameworks to test (default: all)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to run")
    parser.add_argument("--dataset", default="data/sample", help="Dataset path")
    parser.add_argument("-o", "--output", default="results/smoke_test.json", help="Output file")
    args = parser.parse_args()

    frameworks = args.frameworks or list(FRAMEWORK_DEFAULTS.keys())
    llm_config = LLMProviderConfig(provider="openai", model=args.model)

    # Load dataset
    dataset_config = DatasetConfig(name="sample", path=args.dataset)
    docs = DatasetManager.load(dataset_config)
    print(f"Loaded {len(docs)} documents from {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print(f"Frameworks: {', '.join(frameworks)}")

    # Run each framework
    results = {}
    for name in frameworks:
        results[name] = await run_single(name, llm_config, docs, args.query)

    # Build report
    report = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": args.model,
        "dataset": f"{args.dataset} ({len(docs)} docs)",
        "query": args.query,
        "frameworks": results,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")

    # Summary table
    print(f"\n{'Framework':<15} {'Status':<8} {'Index(s)':<10} {'Query(ms)':<10}")
    print("-" * 45)
    for name, r in results.items():
        status = r["status"]
        idx = f"{r.get('index_s', '-')}" if status == "ok" else "-"
        qry = f"{r.get('query_ms', '-')}" if status == "ok" else "-"
        print(f"{name:<15} {status:<8} {idx:<10} {qry:<10}")


if __name__ == "__main__":
    asyncio.run(main())
