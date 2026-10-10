"""Provider registry: map a config ``type`` string to an adapter class.

Add your own backend::

    from omnicode.providers import Provider, register_provider

    @register_provider
    class MyProvider(Provider):
        type_name = "mine"
        ...
"""

from __future__ import annotations

import httpx

from omnicode.config import ConfigError, ProviderSpec
from omnicode.providers.anthropic import AnthropicProvider
from omnicode.providers.base import Provider
from omnicode.providers.coder import CoderProvider
from omnicode.providers.gemini import GeminiProvider
from omnicode.providers.openai import (
    DeepSeekProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAICompatProvider,
    OpenAIProvider,
    VLLMProvider,
)

PROVIDER_TYPES: dict[str, type[Provider]] = {}


def register_provider(cls: type[Provider]) -> type[Provider]:
    if not cls.type_name:
        raise ValueError("Provider subclasses must define type_name")
    PROVIDER_TYPES[cls.type_name] = cls
    return cls


for _cls in (
    CoderProvider,
    GeminiProvider,
    AnthropicProvider,
    OpenAIProvider,
    DeepSeekProvider,
    OllamaProvider,
    LMStudioProvider,
    VLLMProvider,
    OpenAICompatProvider,
):
    register_provider(_cls)


def create_provider(
    name: str, spec: ProviderSpec, client: httpx.AsyncClient | None = None
) -> Provider:
    try:
        cls = PROVIDER_TYPES[spec.type]
    except KeyError:
        raise ConfigError(
            f"Provider '{name}' has unknown type '{spec.type}'. "
            f"Available: {', '.join(sorted(PROVIDER_TYPES))}"
        ) from None
    return cls(
        name,
        api_key=spec.resolve_api_key(),
        base_url=spec.base_url,
        model=spec.model,
        headers=spec.headers,
        client=client,
    )
