"""Ollama local LLM provider."""

from __future__ import annotations

from graphrag_lab.config.schema import LLMProviderConfig
from graphrag_lab.frameworks.base import TokenUsage
from graphrag_lab.providers.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMProviderConfig) -> None:
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError("Install ollama: pip install graphrag-lab[ollama]")

        self.config = config
        base_url = config.api_base or "http://localhost:11434"
        self.client = AsyncClient(host=base_url)

    async def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        response = await self.client.chat(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return LLMResponse(
            text=response["message"]["content"],
            token_usage=TokenUsage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            ),
        )

    def count_tokens(self, text: str) -> int:
        # Approximate: ~4 chars per token
        return len(text) // 4
