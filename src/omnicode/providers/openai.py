"""OpenAI Chat Completions adapter, also used for any OpenAI-compatible
endpoint: Ollama, LM Studio, vLLM, llama.cpp server, OpenRouter, ..."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from omnicode.providers.base import (
    GenerationConfig,
    Message,
    Provider,
    ProviderError,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
    parse_json_args,
)


def to_openai_messages(messages: list[Message], system: str | None) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role in ("system", "user"):
            out.append({"role": m.role, "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


class OpenAIProvider(Provider):
    type_name = "openai"
    requires_api_key = True
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o"
    # Official OpenAI wants max_completion_tokens; most compatible servers want max_tokens.
    token_param = "max_completion_tokens"
    send_stream_usage = True

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    def build_payload(
        self, messages: list[Message], tools: list[ToolSpec] | None, cfg: GenerationConfig
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": cfg.model or self.model,
            "messages": to_openai_messages(messages, cfg.system),
            "stream": True,
            "temperature": cfg.temperature,
        }
        if self.send_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        if cfg.max_tokens:
            payload[self.token_param] = cfg.max_tokens
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        return payload

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._check_key()
        cfg = config or GenerationConfig()
        payload = self.build_payload(messages, tools, cfg)

        slots: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        finish: str | None = None
        async for data in self._sse_post(f"{self.base_url}/chat/completions", self._headers(), payload):
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                err = chunk["error"]
                raise ProviderError(f"{self.name}: {err.get('message', err) if isinstance(err, dict) else err}", provider=self.name)
            if u := chunk.get("usage"):
                usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if text := delta.get("content"):
                    yield StreamEvent("text", text=text)
                for i, tc in enumerate(delta.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    if "index" in tc:
                        idx = tc["index"]
                    elif fn.get("name") or tc.get("id"):
                        idx = len(slots)  # servers that omit index: new call per named chunk
                    else:
                        idx = max(slots, default=0)
                    slot = slots.setdefault(idx, {"id": None, "name": "", "args": ""})
                    if tc.get("id") and not slot["id"]:
                        slot["id"] = tc["id"]
                    if fn.get("name") and not slot["name"]:
                        slot["name"] = fn["name"]
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        slot["args"] = json.dumps(args)
                    elif args:
                        slot["args"] += args
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        for idx in sorted(slots):
            s = slots[idx]
            if not s["name"]:
                continue
            yield StreamEvent(
                "tool_call",
                tool_call=ToolCall(s["id"] or f"call_{uuid.uuid4().hex[:12]}", s["name"], parse_json_args(s["args"])),
            )
        if usage:
            yield StreamEvent("usage", usage=usage)
        stop = "tool_use" if slots else ("length" if finish == "length" else "stop")
        yield StreamEvent("done", stop_reason=stop)

    async def list_models(self) -> list[str]:
        data = await self._get_json(f"{self.base_url}/models", self._headers())
        return sorted(m["id"] for m in data.get("data", []) if "id" in m)


class _LocalMixin:
    requires_api_key = False
    token_param = "max_tokens"
    send_stream_usage = False


class OllamaProvider(_LocalMixin, OpenAIProvider):
    type_name = "ollama"
    default_base_url = "http://localhost:11434/v1"
    default_model = "llama3.1"


class LMStudioProvider(_LocalMixin, OpenAIProvider):
    type_name = "lmstudio"
    default_base_url = "http://localhost:1234/v1"
    default_model = "local-model"


class VLLMProvider(_LocalMixin, OpenAIProvider):
    type_name = "vllm"
    default_base_url = "http://localhost:8000/v1"
    default_model = "default"


class OpenAICompatProvider(_LocalMixin, OpenAIProvider):
    """Any other OpenAI-compatible server (set ``base_url`` in the config)."""

    type_name = "openai_compat"
    default_base_url = "http://localhost:8080/v1"


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek official API (deepseek-chat / deepseek-reasoner)."""

    type_name = "deepseek"
    requires_api_key = True
    default_base_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-chat"
    token_param = "max_tokens"
    send_stream_usage = False
