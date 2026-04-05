"""OpenAI LLM provider."""

from __future__ import annotations

import tiktoken

from graphrag_lab.config.schema import LLMProviderConfig
from graphrag_lab.frameworks.base import TokenUsage
from graphrag_lab.providers.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMProviderConfig) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install graphrag-lab[openai]")

        self.config = config
        self.client = AsyncOpenAI(base_url=config.api_base)
        try:
            self._encoding = tiktoken.encoding_for_model(config.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")

    async def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 4096) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
