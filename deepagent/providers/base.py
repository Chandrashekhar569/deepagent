"""Provider abstraction.

Internally deepagent speaks an OpenAI-ish message format:

    {"role": "system"|"user"|"assistant"|"tool", "content": str|None,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}],
     "tool_call_id": str}

Each provider translates that to its own wire format and streams back a
sequence of events:

    {"type": "text",      "text": "..."}            incremental assistant text
    {"type": "reasoning", "text": "..."}            incremental thinking text
    {"type": "tool_call", "call": {...}}            a complete tool call
    {"type": "usage",     "input": n, "output": n}
    {"type": "done",      "finish_reason": "..."}
    {"type": "error",     "message": "..."}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

Message = dict[str, Any]
Event = dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ProviderError(RuntimeError):
    pass


class Provider:
    name = "base"
    supports_tools = True

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Event]:
        raise NotImplementedError

    # ------------------------------------------------------------------ utils
    def complete(
        self, messages: list[Message], *, model: str | None = None, max_tokens: int = 1024
    ) -> str:
        """Non-agentic single-shot completion (used for compaction, titles)."""
        out: list[str] = []
        for ev in self.stream(messages, [], model=model, max_tokens=max_tokens):
            if ev["type"] == "text":
                out.append(ev["text"])
            elif ev["type"] == "error":
                raise ProviderError(ev["message"])
        return "".join(out)

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            # Some local models emit python-dict-ish or trailing-garbage JSON.
            try:
                import ast

                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            dec = json.JSONDecoder()
            try:
                parsed, _ = dec.raw_decode(raw.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ProviderError(f"could not parse tool arguments: {raw[:400]!r}")
