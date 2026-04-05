"""Dataset loading, validation, and management."""

from __future__ import annotations

from pathlib import Path

import structlog

from graphrag_lab.config.schema import DatasetConfig

logger = structlog.get_logger()


class DatasetManager:
    @staticmethod
    def load(config: DatasetConfig) -> list[str]:
        """Load documents from the dataset path.

        Reads all .txt and .md files from the given directory.
        If sample_size is set, returns only that many documents.
        """
        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        documents: list[str] = []

        if path.is_file():
            documents.append(path.read_text(encoding="utf-8"))
        elif path.is_dir():
            extensions = {".txt", ".md", ".html", ".pdf"}
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in extensions and file_path.is_file():
                    try:
                        documents.append(file_path.read_text(encoding="utf-8"))
                    except UnicodeDecodeError:
                        logger.warning("skipping_non_text_file", path=str(file_path))

        if not documents:
            raise ValueError(f"No documents found in: {path}")

        if config.sample_size and config.sample_size < len(documents):
            documents = documents[: config.sample_size]

        logger.info("dataset_loaded", name=config.name, num_documents=len(documents), domain=config.domain)
        return documents
