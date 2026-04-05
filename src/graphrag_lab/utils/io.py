"""File I/O helpers for JSON, JSONL, YAML, and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def save_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(path: str | Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f) or {}
