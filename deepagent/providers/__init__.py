from __future__ import annotations

from .base import Event, Message, Provider, ProviderError, ToolSpec
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

REGISTRY: dict[str, type[Provider]] = {
    "openai": OpenAICompatProvider,
    "openai-compat": OpenAICompatProvider,
    "nvidia": OpenAICompatProvider,
    "vllm": OpenAICompatProvider,
    "llamacpp": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def build_provider(cfg) -> Provider:
    key = (cfg.provider or "openai").lower()
    if key not in REGISTRY:
        raise ProviderError(
            f"unknown provider {key!r}; known: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[key](cfg)


__all__ = [
    "Event",
    "Message",
    "Provider",
    "ProviderError",
    "ToolSpec",
    "REGISTRY",
    "build_provider",
]
