"""Anthropic LLM provider."""

from __future__ import annotations

from graphrag_lab.config.schema import LLMProviderConfig
from graphrag_lab.frameworks.base import TokenUsage
from graphrag_lab.providers.base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMProviderConfig) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install graphrag-lab[anthropic]")

        self.config = config
        self.client = AsyncAnthropic()

    async def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=response.content[0].text if response.content else "",
            token_usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
        )

    def count_tokens(self, text: str) -> int:
        # Approximate: ~4 chars per token for Claude models
        return len(text) // 4
