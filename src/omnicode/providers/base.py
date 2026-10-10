"""Unified provider abstraction: messages, tools, streaming events, base class."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar, Literal

import httpx

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------- errors
class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        provider: str = "",
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.provider = provider


class AuthError(ProviderError):
    """Missing / rejected API key."""


# ---------------------------------------------------------------- data model
@dataclass
class ToolSpec:
    """Provider-neutral function declaration (JSON-Schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Provider-specific data that must be echoed back (e.g. Gemini thoughtSignature).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    name: str | None = None  # tool only: name of the called tool
    is_error: bool = False  # tool only

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls("system", text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls("assistant", text, tool_calls=list(tool_calls or []))

    @classmethod
    def tool_result(
        cls, call_id: str, name: str, content: str, is_error: bool = False
    ) -> "Message":
        return cls("tool", content, tool_call_id=call_id, name=name, is_error=is_error)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GenerationConfig:
    model: str | None = None  # None -> provider default
    temperature: float = 0.2
    max_tokens: int = 4096
    system: str | None = None


@dataclass
class StreamEvent:
    kind: Literal["text", "tool_call", "usage", "notice", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    stop_reason: str | None = None  # "stop" | "tool_use" | "length"


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage | None
    stop_reason: str | None


# ------------------------------------------------------------------- helpers
def parse_json_args(raw: str | dict | None) -> dict[str, Any]:
    """Parse streamed tool-call arguments, tolerating empty/invalid JSON."""
    if isinstance(raw, dict):
        return raw
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": "arguments were not valid JSON", "_raw": raw}
    return value if isinstance(value, dict) else {"value": value}


def extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:500] or "empty error body"
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        return str(err.get("message") or err)[:500]
    return str(err)[:500]


async def iter_sse(resp: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data:`` payload of each server-sent event."""
    buf: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if buf:
                yield "\n".join(buf)
                buf = []
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            value = line[5:]
            buf.append(value[1:] if value.startswith(" ") else value)
    if buf:
        yield "\n".join(buf)


# ------------------------------------------------------------ provider base
class Provider(ABC):
    """Contract every LLM backend implements.

    Subclasses only translate between the neutral model above and the vendor
    wire format; retries, SSE parsing and error mapping live here.
    """

    type_name: ClassVar[str] = ""
    requires_api_key: ClassVar[bool] = True
    default_base_url: ClassVar[str] = ""
    default_model: ClassVar[str] = ""

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        self.name = name or self.type_name
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.model = model or self.default_model
        self.extra_headers = dict(headers or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------ interface
    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream text / tool-call / usage events and finish with ``done``."""

    async def list_models(self) -> list[str]:
        return []

    # --------------------------------------------------------------- common
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: GenerationConfig | None = None,
    ) -> Completion:
        text: list[str] = []
        calls: list[ToolCall] = []
        usage: Usage | None = None
        stop: str | None = None
        async for ev in self.stream(messages, tools, config):
            if ev.kind == "text":
                text.append(ev.text)
            elif ev.kind == "tool_call" and ev.tool_call:
                calls.append(ev.tool_call)
            elif ev.kind == "usage":
                usage = ev.usage
            elif ev.kind == "done":
                stop = ev.stop_reason
        return Completion("".join(text), calls, usage, stop)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _check_key(self) -> None:
        if self.requires_api_key and not self.api_key:
            raise AuthError(
                f"No API key for provider '{self.name}'. Run `omnicode key {self.name}` "
                "or enter it in Settings.",
                provider=self.name,
            )

    def _http_error(self, status: int, body: str) -> ProviderError:
        msg = f"{self.name}: HTTP {status}: {extract_error_message(body)}"
        if status in (401, 403):
            return AuthError(msg, status=status, provider=self.name)
        retryable = status in (408, 409, 425, 429) or status >= 500
        return ProviderError(msg, status=status, retryable=retryable, provider=self.name)

    async def _sse_post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        """POST and stream SSE payloads. Retries transient failures *before*
        the first byte is received; mid-stream failures are surfaced."""
        attempt = 0
        while True:
            started = False
            try:
                async with self.client.stream(
                    "POST", url, headers={**headers, **self.extra_headers}, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        err = self._http_error(resp.status_code, body)
                        if err.retryable and attempt < self.max_retries:
                            delay = self._retry_delay(attempt, resp)
                            attempt += 1
                            await asyncio.sleep(delay)
                            continue
                        raise err
                    started = True
                    async for data in iter_sse(resp):
                        yield data
                    return
            except httpx.TransportError as exc:
                if not started and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                detail = str(exc) or exc.__class__.__name__
                raise ProviderError(
                    f"{self.name}: cannot reach {self.base_url} ({detail})",
                    retryable=True,
                    provider=self.name,
                ) from exc

    def _retry_delay(self, attempt: int, resp: httpx.Response) -> float:
        header = resp.headers.get("retry-after")
        if header and header.replace(".", "", 1).isdigit():
            return min(float(header), 30.0)
        return min(self.retry_delay * (2**attempt), 10.0)

    async def _get_json(self, url: str, headers: dict[str, str]) -> Any:
        try:
            resp = await self.client.get(url, headers={**headers, **self.extra_headers})
        except httpx.TransportError as exc:
            raise ProviderError(
                f"{self.name}: cannot reach {self.base_url} ({exc})",
                retryable=True,
                provider=self.name,
            ) from exc
        if resp.status_code >= 400:
            raise self._http_error(resp.status_code, resp.text)
        return resp.json()
