"""Google Gemini adapter (using google-genai Interactions API in stateless mode)."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from google.genai.types import Tool as GenaiTool
from google.genai.types import FunctionDeclaration

from omnicode.providers.base import (
    GenerationConfig,
    Message,
    Provider,
    ProviderError,
    AuthError,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

def sanitize_schema(schema: Any) -> Any:
    # Optional: we can keep the sanitize_schema from previous logic if needed,
    # but google-genai takes dicts nicely.
    _ALLOWED_SCHEMA_KEYS = {"type", "description", "properties", "required", "items", "enum", "format", "nullable"}
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = sanitize_schema(value)
        else:
            out[key] = value
    return out


def to_gemini_history(messages: list[Message]) -> list[types.Content]:
    history = []
    for m in messages:
        if m.role == "system": continue
        elif m.role == "user" and m.content:
            history.append(types.Content(role="user", parts=[types.Part.from_text(text=m.content)]))
        elif m.role == "assistant":
            parts = []
            if m.content:
                parts.append(types.Part.from_text(text=m.content))
            for tc in m.tool_calls:
                parts.append(types.Part.from_function_call(name=tc.name, args=tc.arguments))
            if parts:
                history.append(types.Content(role="model", parts=parts))
        elif m.role == "tool":
            res = m.content or "success"
            if m.is_error: res = f"Error: {res}"
            history.append(types.Content(role="user", parts=[types.Part.from_function_response(name=m.name or "tool", response={"result": res})]))
    return history



class GeminiProvider(Provider):
    type_name = "gemini"
    requires_api_key = True
    default_base_url = ""
    default_model = "gemini-3.1-pro-preview"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._genai_client = None

    def _get_client(self) -> genai.Client:
        if not self._genai_client:
            self._check_key()
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._check_key()
        client = self._get_client()
        cfg = config or GenerationConfig()
        
        history = to_gemini_history(messages)
        system_texts = [m.content for m in messages if m.role == "system"]
        if cfg.system:
            system_texts.insert(0, cfg.system)
            
        genai_tools = []
        if tools:
            for t in tools:
                params = sanitize_schema(t.parameters)
                if not params.get("properties"):
                    params = {"type": "OBJECT", "properties": {}}
                genai_tools.append({
                    "function_declarations": [{
                        "name": t.name,
                        "description": t.description,
                        "parameters": params
                    }]
                })
                
        gen_config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            system_instruction="\n\n".join(system_texts) if system_texts else None,
            tools=genai_tools if genai_tools else None,
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        )
        model_name = (cfg.model or self.model).replace("models/", "")
        
        try:
            yield StreamEvent("notice", text="Thinking...")
            stream_iter = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=history,
                config=gen_config
            )
            
            saw_call = False
            finish_reason = "stop"
            
            async for chunk in stream_iter:
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish_reason = chunk.candidates[0].finish_reason.name.lower()
                    
                if not chunk.parts:
                    continue
                    
                for p in chunk.parts:
                    if getattr(p, "thought", False):
                        continue # We already yielded thinking notice
                    elif p.text:
                        yield StreamEvent("text", text=p.text)
                    elif p.function_call:
                        saw_call = True
                        yield StreamEvent("tool_call", tool_call=ToolCall(
                            id=p.function_call.name, # Use name as ID since gemini doesn't have IDs
                            name=p.function_call.name,
                            arguments=p.function_call.args
                        ))
                        
            yield StreamEvent("done", stop_reason="tool_use" if saw_call else finish_reason)
            
        except genai.errors.APIError as e:
            if e.code in (401, 403):
                raise AuthError(f"gemini auth error: {e.message}", provider=self.name)
            raise ProviderError(f"gemini API error: {e.message}", provider=self.name)
        except Exception as e:
            raise ProviderError(f"gemini connection error: {str(e)}", retryable=True, provider=self.name)

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            # We use synchronous call for list_models for simplicity since it's just setup
            models = client.models.list()
            return [m.name.replace("models/", "") for m in models]
        except Exception:
            return ["gemini-3.8-flash"]
