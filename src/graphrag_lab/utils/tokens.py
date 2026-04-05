"""Token counting utilities."""

from __future__ import annotations

import tiktoken


_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for the given text using the appropriate tokenizer.

    For OpenAI models, uses tiktoken. For others, uses an approximation.
    """
    # Try tiktoken first (works for OpenAI models)
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback: use cl100k_base (GPT-4 tokenizer) as approximation
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")

    return len(_ENCODING_CACHE[model].encode(text))
