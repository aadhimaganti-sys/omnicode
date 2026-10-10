"""Anthropic Claude adapter (Messages API, SSE streaming, tool_use)."""

from __future__ import annotations

import json
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

_STOP_MAP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_use", "max_tokens": "length"}


def to_anthropic_messages(messages: list[Message]) -> tuple[list[dict], list[str]]:
    out: list[dict] = []
    system: list[str] = []

    def push(role: str, blocks: list[dict]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m.role == "system":
            system.append(m.content)
        elif m.role == "user":
            if m.content:
                push("user", [{"type": "text", "text": m.content}])
        elif m.role == "assistant":
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            if blocks:
                push("assistant", blocks)
        elif m.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content or "(empty)",
            }
            if m.is_error:
                block["is_error"] = True
            push("user", [block])
    return out, system


class AnthropicProvider(Provider):
    type_name = "anthropic"
    requires_api_key = True
    default_base_url = "https://api.anthropic.com/v1"
    default_model = "claude-sonnet-4-5"
    api_version = "2023-06-01"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def build_payload(
        self, messages: list[Message], tools: list[ToolSpec] | None, cfg: GenerationConfig
    ) -> dict[str, Any]:
        msgs, system_texts = to_anthropic_messages(messages)
        if cfg.system:
            system_texts.insert(0, cfg.system)
        payload: dict[str, Any] = {
            "model": cfg.model or self.model,
            "max_tokens": cfg.max_tokens,
            "temperature": min(max(cfg.temperature, 0.0), 1.0),
            "messages": msgs,
            "stream": True,
        }
        if system_texts:
            payload["system"] = "\n\n".join(system_texts)
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
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

        blocks: dict[int, dict[str, Any]] = {}
        in_tok = out_tok = 0
        stop: str | None = None
        saw_call = False
        async for data in self._sse_post(f"{self.base_url}/messages", self._headers(), payload):
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "message_start":
                u = ev.get("message", {}).get("usage", {})
                in_tok, out_tok = u.get("input_tokens", 0), u.get("output_tokens", 0)
            elif kind == "content_block_start":
                cb = ev.get("content_block", {})
                blocks[ev["index"]] = {"type": cb.get("type"), "id": cb.get("id"), "name": cb.get("name"), "json": ""}
                if cb.get("type") == "text" and cb.get("text"):
                    yield StreamEvent("text", text=cb["text"])
            elif kind == "content_block_delta":
                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield StreamEvent("text", text=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta" and ev["index"] in blocks:
                    blocks[ev["index"]]["json"] += delta.get("partial_json", "")
            elif kind == "content_block_stop":
                blk = blocks.pop(ev.get("index"), None)
                if blk and blk["type"] == "tool_use":
                    saw_call = True
                    yield StreamEvent(
                        "tool_call",
                        tool_call=ToolCall(blk["id"], blk["name"], parse_json_args(blk["json"])),
                    )
            elif kind == "message_delta":
                stop = ev.get("delta", {}).get("stop_reason") or stop
                out_tok = ev.get("usage", {}).get("output_tokens", out_tok)
            elif kind == "error":
                err = ev.get("error", {})
                raise ProviderError(f"anthropic: {err.get('message', err)}", provider=self.name)
        yield StreamEvent("usage", usage=Usage(in_tok, out_tok))
        yield StreamEvent("done", stop_reason="tool_use" if saw_call else _STOP_MAP.get(stop or "", "stop"))

    async def list_models(self) -> list[str]:
        self._check_key()
        data = await self._get_json(f"{self.base_url}/models?limit=100", self._headers())
        return [m["id"] for m in data.get("data", [])]
