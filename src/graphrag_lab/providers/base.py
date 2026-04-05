"""Abstract LLM provider interface for evaluation (LLM-as-judge) and custom calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from graphrag_lab.frameworks.base import TokenUsage


@dataclass
class LLMResponse:
    text: str
    token_usage: TokenUsage


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        """Generate a completion for the given prompt."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
