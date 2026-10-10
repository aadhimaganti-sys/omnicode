"""Coder Provider: Elite Cloud-Powered AI Coding Model.

Offloads all neural processing, deep thinking, and token generation to Google Cloud
to keep the Raspberry Pi completely cool with zero CPU thermal load.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from google.genai import types

from omnicode.neural.tokenizer import ScratchTokenizer
from omnicode.neural.transformer import TransformerConfig, TransformerModel
from omnicode.providers.base import (
    AuthError,
    GenerationConfig,
    Message,
    Provider,
    ProviderError,
    StreamEvent,
    ToolCall,
    ToolSpec,
)
from omnicode.providers.gemini import GeminiProvider, sanitize_schema, to_gemini_history

CODER_SYSTEM_INSTRUCTION = """You are OmniCode Coder, an impossibly talented, world-class AI coding model running on a cloud-accelerated neural architecture.
You are trained by Google Gemini and specialized for flawless, production-grade software engineering.

CORE COMPETENCIES & DIRECTIVES:
1. Flawless Code Craft: Write mathematically sound, idiomatic, and highly optimized code across all languages (Python, Rust, C, C++, Go, TypeScript/JS, etc.).
2. Edge Case Immunity: Eliminate null-pointer/None dereferences, off-by-one errors, memory leaks, concurrency races, and boundary failures before they happen.
3. Surgical Tool Autonomy: When working in a project workspace, inspect existing code (`read_file`, `list_dir`, `search_code`), write complete edits (`edit_file`, `create_file`), and verify results by executing tests/commands (`run_command`).
4. Zero Stubbing: Never output placeholders like `// TODO: implement later` or ellipsis `...`. Always produce complete, working, robust implementations.
5. Thermal & Cloud Efficiency: Deliver instant, streaming responses with clear reasoning while offloading heavy computation to cloud neural clusters.
"""


class CoderProvider(GeminiProvider):
    """The 'coder' provider: combines Gemini cloud neural acceleration with elite coding prompts."""

    type_name = "coder"
    requires_api_key = True
    default_base_url = ""
    default_model = "gemini-3.8-flash"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scratch transformer architecture reference available for offline tokenization
        self.tokenizer = ScratchTokenizer()
        self._cloud_engine = "Google Cloud Neural Infrastructure"

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

        # Assemble system instructions with Coder persona
        system_texts = [CODER_SYSTEM_INSTRUCTION]
        for m in messages:
            if m.role == "system" and m.content:
                system_texts.append(m.content)
        if cfg.system:
            system_texts.append(cfg.system)

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
                        "parameters": params,
                    }]
                })

        gen_config = types.GenerateContentConfig(
            temperature=cfg.temperature if cfg.temperature is not None else 0.1,
            max_output_tokens=cfg.max_tokens or 8192,
            system_instruction="\n\n".join(system_texts),
            tools=genai_tools if genai_tools else None,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        model_name = (cfg.model or self.model).replace("models/", "")

        try:
            yield StreamEvent(
                "notice",
                text="[Cloud Neural Engine | Zero Local CPU] Thinking & coding in cloud...",
            )
            stream_iter = await client.aio.models.generate_content_stream(
                model=model_name, contents=history, config=gen_config
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
                        continue
                    elif p.text:
                        yield StreamEvent("text", text=p.text)
                    elif p.function_call:
                        saw_call = True
                        yield StreamEvent(
                            "tool_call",
                            tool_call=ToolCall(
                                id=p.function_call.name,
                                name=p.function_call.name,
                                arguments=p.function_call.args or {},
                            ),
                        )

            yield StreamEvent("done", stop_reason="tool_use" if saw_call else finish_reason)

        except types.errors.APIError if hasattr(types, "errors") and hasattr(types.errors, "APIError") else Exception as e:
            # Catch known genai errors or general exception
            err_msg = str(e)
            if "401" in err_msg or "403" in err_msg or "API key not valid" in err_msg:
                raise AuthError(f"coder cloud auth error: {err_msg}", provider=self.name)
            raise ProviderError(f"coder cloud error: {err_msg}", provider=self.name)
