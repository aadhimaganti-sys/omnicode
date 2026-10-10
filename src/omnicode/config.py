"""Configuration management (``~/.omnicode/config.json``).

API keys can be entered inside the app (CLI ``/key`` or the web Settings
dialog) and are persisted here with owner-only file permissions. Environment
variables (``GEMINI_API_KEY`` ...) are used as a fallback.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Provider types that cannot work without an API key.
KEY_REQUIRED_TYPES = {"gemini", "anthropic", "openai", "coder", "deepseek"}

# Environment variables consulted when no key is stored in the config file.
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "coder": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


class ConfigError(Exception):
    """Invalid or unreadable configuration."""


def config_dir() -> Path:
    return Path(os.environ.get("OMNICODE_HOME") or (Path.home() / ".omnicode"))


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class ProviderSpec:
    """Settings for one named provider instance."""

    type: str
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def requires_key(self) -> bool:
        return self.type in KEY_REQUIRED_TYPES

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        names = (self.api_key_env,) if self.api_key_env else ENV_KEYS.get(self.type, ())
        for name in names:
            value = os.environ.get(name or "")
            if value:
                return value
        return None

    def key_source(self) -> str | None:
        if self.api_key:
            return "config"
        return "env" if self.resolve_api_key() else None

    @property
    def configured(self) -> bool:
        return (not self.requires_key) or bool(self.resolve_api_key())

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_type: str) -> "ProviderSpec":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("type", default_type)
        return cls(**kwargs)


@dataclass
class PermissionSettings:
    # ask: confirm every edit & command | auto_edits: auto-apply file edits
    # yolo: auto-run non-destructive commands too. Destructive commands
    # ALWAYS require confirmation in every mode.
    mode: str = "ask"
    command_timeout: int = 120
    allowed_commands: list[str] = field(default_factory=list)  # e.g. ["pytest", "npm test"]


def default_providers() -> dict[str, ProviderSpec]:
    return {
        "coder": ProviderSpec(type="coder", model="gemini-3.8-flash", temperature=0.1),
        "gemini": ProviderSpec(type="gemini"),
        "anthropic": ProviderSpec(type="anthropic"),
        "openai": ProviderSpec(type="openai"),
        "deepseek": ProviderSpec(type="deepseek", model="deepseek-chat"),
        "ollama": ProviderSpec(type="ollama"),
        "lmstudio": ProviderSpec(type="lmstudio"),
        "vllm": ProviderSpec(type="vllm"),
    }


@dataclass
class Config:
    active: str = "gemini"
    fallback: list[str] = field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 8192
    providers: dict[str, ProviderSpec] = field(default_factory=default_providers)
    system_prompt: str = ""  # extra instructions appended to the built-in prompt
    custom_prompts: dict[str, str] = field(default_factory=dict)  # name -> template using {input}
    permissions: PermissionSettings = field(default_factory=PermissionSettings)
    max_agent_steps: int = 25
    path: Path = field(default_factory=config_path, repr=False)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else config_path()
        cfg = cls(path=path)
        if not path.exists():
            return cfg
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a JSON object")

        cfg.active = str(raw.get("active", cfg.active))
        cfg.fallback = [str(x) for x in raw.get("fallback", [])]
        cfg.temperature = float(raw.get("temperature", cfg.temperature))
        cfg.max_tokens = int(raw.get("max_tokens", cfg.max_tokens))
        cfg.system_prompt = str(raw.get("system_prompt", ""))
        cfg.custom_prompts = {str(k): str(v) for k, v in raw.get("custom_prompts", {}).items()}
        cfg.max_agent_steps = int(raw.get("max_agent_steps", cfg.max_agent_steps))

        perms = raw.get("permissions", {})
        cfg.permissions = PermissionSettings(
            mode=str(perms.get("mode", "ask")),
            command_timeout=int(perms.get("command_timeout", 120)),
            allowed_commands=[str(x) for x in perms.get("allowed_commands", [])],
        )
        for name, data in raw.get("providers", {}).items():
            default_type = cfg.providers[name].type if name in cfg.providers else name
            cfg.providers[name] = ProviderSpec.from_dict(dict(data), default_type)

        # Inherit key from gemini if coder has no key specified
        if "coder" in cfg.providers and not cfg.providers["coder"].api_key:
            if "gemini" in cfg.providers and cfg.providers["gemini"].api_key:
                cfg.providers["coder"].api_key = cfg.providers["gemini"].api_key

        return cfg

    def to_dict(self) -> dict[str, Any]:
        providers = {}
        for name, spec in self.providers.items():
            d = {k: v for k, v in vars(spec).items() if v not in (None, {}, "")}
            d["type"] = spec.type
            providers[name] = d
        return {
            "active": self.active,
            "fallback": self.fallback,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "custom_prompts": self.custom_prompts,
            "max_agent_steps": self.max_agent_steps,
            "permissions": vars(self.permissions),
            "providers": providers,
        }

    def save(self) -> None:
        """Atomically write the config with owner-only permissions (0600)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".config-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)
                fh.write("\n")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------- helpers
    def spec(self, name: str) -> ProviderSpec:
        try:
            return self.providers[name]
        except KeyError:
            raise ConfigError(
                f"Unknown provider '{name}'. Known: {', '.join(sorted(self.providers))}"
            ) from None

    def effective_temperature(self, name: str) -> float:
        t = self.spec(name).temperature
        return self.temperature if t is None else t

    def effective_max_tokens(self, name: str) -> int:
        m = self.spec(name).max_tokens
        return self.max_tokens if m is None else m

    def set_api_key(self, name: str, key: str | None) -> None:
        self.spec(name).api_key = (key or "").strip() or None

    def set_model(self, name: str, model: str | None) -> None:
        self.spec(name).model = (model or "").strip() or None

    def set_base_url(self, name: str, url: str | None) -> None:
        self.spec(name).base_url = (url or "").strip() or None
