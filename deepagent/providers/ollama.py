"""Native Ollama provider (/api/chat) - keeps tool calling and keep_alive."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .base import Event, Message, Provider, ProviderError, ToolSpec


class OllamaProvider(Provider):
    name = "ollama"

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        payload: dict[str, Any] = {
            "model": model or self.cfg.model,
            "messages": [_msg(m) for m in messages],
            "stream": True,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_ctx": min(self.cfg.context_window, 131072),
                "num_predict": max_tokens or self.cfg.max_tokens,
            },
            "keep_alive": "30m",
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
        payload.update(self.cfg.extra_body or {})

        pending: list[dict[str, Any]] = []
        try:
            with httpx.Client(timeout=httpx.Timeout(900.0, connect=15.0)) as client:
                with client.stream("POST", url, json=payload) as r:
                    if r.status_code >= 400:
                        raise ProviderError(
                            f"HTTP {r.status_code} from {url}: "
                            f"{r.read().decode('utf-8', 'replace')[:600]}"
                        )
                    for line in r.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise ProviderError(str(chunk["error"]))
                        msg = chunk.get("message") or {}
                        if msg.get("thinking"):
                            yield {"type": "reasoning", "text": msg["thinking"]}
                        if msg.get("content"):
                            yield {"type": "text", "text": msg["content"]}
                        for i, tc in enumerate(msg.get("tool_calls") or []):
                            fn = tc.get("function") or {}
                            pending.append(
                                {
                                    "id": tc.get("id") or f"call_{len(pending)}_{i}",
                                    "name": fn.get("name", ""),
                                    "arguments": self._parse_args(fn.get("arguments")),
                                }
                            )
                        if chunk.get("done"):
                            yield {
                                "type": "usage",
                                "input": chunk.get("prompt_eval_count", 0),
                                "output": chunk.get("eval_count", 0),
                            }
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cannot reach ollama at {self.cfg.base_url} ({exc}). "
                "Is `ollama serve` running?"
            ) from exc

        for call in pending:
            yield {"type": "tool_call", "call": call}
        yield {"type": "done", "finish_reason": "tool_calls" if pending else "stop"}

    def list_models(self) -> list[str]:
        r = httpx.get(self.cfg.base_url.rstrip("/") + "/api/tags", timeout=20)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))


def _msg(m: Message) -> dict[str, Any]:
    role = m["role"]
    if role == "tool":
        return {"role": "tool", "content": m.get("content") or ""}
    out: dict[str, Any] = {"role": role, "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {"function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
            for c in m["tool_calls"]
        ]
    return out
