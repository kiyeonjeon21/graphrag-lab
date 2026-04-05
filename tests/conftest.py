"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def sample_data_dir(project_root: Path) -> Path:
    return project_root / "data" / "sample"


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"
