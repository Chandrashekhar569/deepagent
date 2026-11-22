"""Anthropic Messages API provider (native tool use + extended thinking)."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .base import Event, Message, Provider, ProviderError, ToolSpec


class AnthropicProvider(Provider):
    name = "anthropic"

    def _headers(self) -> dict[str, str]:
        if not self.cfg.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return {
            "x-api-key": self.cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        } | (self.cfg.extra_headers or {})

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        system, converted = _convert(messages)
        payload: dict[str, Any] = {
            "model": model or self.cfg.model,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": converted,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [t.to_anthropic() for t in tools]
        payload.update(self.cfg.extra_body or {})

        url = self.cfg.base_url.rstrip("/") + "/messages"
        blocks: dict[int, dict[str, Any]] = {}
        stop_reason = None
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                with client.stream("POST", url, headers=self._headers(), json=payload) as r:
                    if r.status_code >= 400:
                        raise ProviderError(
                            f"HTTP {r.status_code}: {r.read().decode('utf-8', 'replace')[:800]}"
                        )
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        et = ev.get("type")
                        if et == "content_block_start":
                            blocks[ev["index"]] = dict(ev["content_block"])
                            blocks[ev["index"]]["_json"] = ""
                        elif et == "content_block_delta":
                            d = ev.get("delta") or {}
                            if d.get("type") == "text_delta":
                                yield {"type": "text", "text": d["text"]}
                            elif d.get("type") == "thinking_delta":
                                yield {"type": "reasoning", "text": d["thinking"]}
                            elif d.get("type") == "input_json_delta":
                                blocks.setdefault(ev["index"], {"_json": ""})
                                blocks[ev["index"]]["_json"] += d.get("partial_json", "")
                        elif et == "content_block_stop":
                            blk = blocks.get(ev["index"], {})
                            if blk.get("type") == "tool_use":
                                args = blk.get("input") or {}
                                if blk.get("_json"):
                                    args = self._parse_args(blk["_json"])
                                yield {
                                    "type": "tool_call",
                                    "call": {
                                        "id": blk.get("id", f"call_{ev['index']}"),
                                        "name": blk.get("name", ""),
                                        "arguments": args,
                                    },
                                }
                        elif et == "message_delta":
                            stop_reason = (ev.get("delta") or {}).get("stop_reason")
                            if u := ev.get("usage"):
                                yield {
                                    "type": "usage",
                                    "input": u.get("input_tokens", 0),
                                    "output": u.get("output_tokens", 0),
                                }
                        elif et == "message_start":
                            u = (ev.get("message") or {}).get("usage") or {}
                            if u:
                                yield {
                                    "type": "usage",
                                    "input": u.get("input_tokens", 0),
                                    "output": u.get("output_tokens", 0),
                                }
                        elif et == "error":
                            raise ProviderError(str(ev.get("error")))
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"connection to {url} failed: {exc}") from exc

        yield {"type": "done", "finish_reason": stop_reason or "stop"}


def _convert(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m.get("content") or "")
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        content: list[dict[str, Any]] = []
        if m.get("content"):
            content.append({"type": "text", "text": m["content"]})
        for c in m.get("tool_calls") or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": c["id"],
                    "name": c["name"],
                    "input": c.get("arguments") or {},
                }
            )
        out.append({"role": role, "content": content or [{"type": "text", "text": ""}]})
    return "\n\n".join(p for p in system_parts if p), out
