"""LLMEngine: the object the agent talks to. Wraps the active provider plus an
ordered fallback chain (e.g. Claude -> Gemini -> local Ollama)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from omnicode.config import Config, ConfigError
from omnicode.providers.base import (
    GenerationConfig,
    Message,
    Provider,
    ProviderError,
    StreamEvent,
    ToolSpec,
)
from omnicode.providers.registry import create_provider


class NotConfiguredError(ConfigError):
    def __init__(self, provider: str):
        super().__init__(
            f"Provider '{provider}' has no API key. Enter one with `omnicode key {provider}` "
            "(or `/key` inside the REPL / Settings in the web UI)."
        )
        self.provider = provider


@dataclass
class Target:
    name: str
    provider: Provider
    config: GenerationConfig


class LLMEngine:
    def __init__(self, targets: list[Target]):
        if not targets:
            raise ValueError("LLMEngine needs at least one target")
        self.targets = targets

    @property
    def primary(self) -> Target:
        return self.targets[0]

    @property
    def label(self) -> str:
        t = self.primary
        return f"{t.name}:{t.config.model or t.provider.model}"

    @classmethod
    def from_config(cls, cfg: Config, client: httpx.AsyncClient | None = None) -> "LLMEngine":
        active = cfg.spec(cfg.active)
        if not active.configured:
            raise NotConfiguredError(cfg.active)
        targets: list[Target] = []
        seen: set[str] = set()
        for name in [cfg.active, *cfg.fallback]:
            if name in seen or name not in cfg.providers:
                continue
            spec = cfg.providers[name]
            if not spec.configured:
                continue  # silently skip unconfigured fallbacks
            seen.add(name)
            provider = create_provider(name, spec, client)
            targets.append(
                Target(
                    name,
                    provider,
                    GenerationConfig(
                        model=spec.model or provider.model,
                        temperature=cfg.effective_temperature(name),
                        max_tokens=cfg.effective_max_tokens(name),
                    ),
                )
            )
        return cls(targets)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from the primary target; on failure *before any output*,
        transparently fall through to the next target."""
        last_error: ProviderError | None = None
        for i, target in enumerate(self.targets):
            cfg = GenerationConfig(
                model=target.config.model,
                temperature=target.config.temperature,
                max_tokens=target.config.max_tokens,
                system=system,
            )
            emitted = False
            try:
                async for ev in target.provider.stream(messages, tools, cfg):
                    if ev.kind in ("text", "tool_call"):
                        emitted = True
                    yield ev
                return
            except ProviderError as exc:
                last_error = exc
                if emitted or i == len(self.targets) - 1:
                    raise
                nxt = self.targets[i + 1].name
                yield StreamEvent("notice", text=f"{target.name} failed ({exc}); falling back to {nxt}")
        if last_error:  # pragma: no cover - loop always returns or raises
            raise last_error

    async def aclose(self) -> None:
        for t in self.targets:
            await t.provider.aclose()
