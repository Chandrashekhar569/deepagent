"""OpenAI-compatible chat-completions provider.

Covers NVIDIA NIM (build.nvidia.com and self-hosted NIM containers), vLLM,
llama.cpp's llama-server, LM Studio, Ollama's /v1 shim, OpenRouter, Groq,
DeepSeek, OpenAI itself - anything that speaks /chat/completions.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .base import Event, Message, Provider, ProviderError, ToolSpec


class OpenAICompatProvider(Provider):
    name = "openai"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        if "openrouter" in (self.cfg.base_url or ""):
            h.setdefault("HTTP-Referer", "https://localhost/deepagent")
            h.setdefault("X-Title", "deepagent")
        h.update(self.cfg.extra_headers or {})
        return h

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.cfg.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        payload.update(self.cfg.extra_body or {})
        return payload

    # ------------------------------------------------------------------ stream
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        if not self.cfg.stream:
            yield from self._blocking(url, messages, tools, model, max_tokens)
            return

        payload = self._payload(messages, tools, model, max_tokens, True)
        calls: dict[int, dict[str, Any]] = {}
        finish = None
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                with client.stream("POST", url, headers=self._headers(), json=payload) as r:
                    if r.status_code >= 400:
                        body = r.read().decode("utf-8", "replace")
                        raise ProviderError(f"HTTP {r.status_code} from {url}: {body[:800]}")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line in ("", "[DONE]"):
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise ProviderError(str(chunk["error"]))
                        if usage := chunk.get("usage"):
                            yield {
                                "type": "usage",
                                "input": usage.get("prompt_tokens", 0),
                                "output": usage.get("completion_tokens", 0),
                            }
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            for key in ("reasoning_content", "reasoning"):
                                if delta.get(key):
                                    yield {"type": "reasoning", "text": delta[key]}
                            if delta.get("content"):
                                yield {"type": "text", "text": delta["content"]}
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                slot = calls.setdefault(
                                    idx, {"id": "", "name": "", "arguments": ""}
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["arguments"] += fn["arguments"]
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"connection to {url} failed: {exc}") from exc

        for idx in sorted(calls):
            slot = calls[idx]
            if not slot["name"]:
                continue
            yield {
                "type": "tool_call",
                "call": {
                    "id": slot["id"] or f"call_{idx}",
                    "name": slot["name"],
                    "arguments": self._parse_args(slot["arguments"]),
                },
            }
        yield {"type": "done", "finish_reason": finish or "stop"}

    # ---------------------------------------------------------------- blocking
    def _blocking(
        self,
        url: str,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str | None,
        max_tokens: int | None,
    ) -> Iterator[Event]:
        payload = self._payload(messages, tools, model, max_tokens, False)
        headers = self._headers() | {"Accept": "application/json"}
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise ProviderError(f"HTTP {r.status_code} from {url}: {r.text[:800]}")
            data = r.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"connection to {url} failed: {exc}") from exc

        if usage := data.get("usage"):
            yield {
                "type": "usage",
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        if msg.get("reasoning_content"):
            yield {"type": "reasoning", "text": msg["reasoning_content"]}
        if msg.get("content"):
            yield {"type": "text", "text": msg["content"]}
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            yield {
                "type": "tool_call",
                "call": {
                    "id": tc.get("id") or f"call_{i}",
                    "name": fn.get("name", ""),
                    "arguments": self._parse_args(fn.get("arguments")),
                },
            }
        yield {"type": "done", "finish_reason": choice.get("finish_reason", "stop")}

    # ------------------------------------------------------------------ models
    def list_models(self) -> list[str]:
        url = self.cfg.base_url.rstrip("/") + "/models"
        try:
            r = httpx.get(url, headers=self._headers() | {"Accept": "application/json"}, timeout=30)
            r.raise_for_status()
            return sorted(m.get("id", "") for m in r.json().get("data", []))
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"could not list models at {url}: {exc}") from exc


def _to_openai_message(m: Message) -> dict[str, Any]:
    role = m["role"]
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m["tool_call_id"],
            "content": m.get("content") or "",
        }
    out: dict[str, Any] = {"role": role, "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("arguments") or {}),
                },
            }
            for c in m["tool_calls"]
        ]
        # OpenAI requires content to be present (may be null) alongside calls
        out["content"] = m.get("content") or None
    return out
