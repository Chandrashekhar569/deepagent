"""Tool plumbing: registry, declaration helpers, result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..providers import ToolSpec


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # short line shown in the transcript instead of the full payload
    summary: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, summary: str = "", **meta: Any) -> "ToolResult":
        return cls(content=content, summary=summary, meta=meta)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(content=f"Error: {message}", is_error=True, summary="error")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], Any], ToolResult]
    read_only: bool = False
    concurrency_safe: bool = True

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def run(self, args: dict[str, Any], ctx: Any) -> ToolResult:
        try:
            return self.handler(args, ctx)
        except Exception as exc:  # tools must never kill the loop
            return ToolResult.error(f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        read_only: bool = False,
    ) -> Callable:
        def deco(fn: Callable[[dict, Any], ToolResult]) -> Callable:
            self.register(Tool(name, description, parameters, fn, read_only=read_only))
            return fn

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, include: list[str] | None = None) -> list[ToolSpec]:
        return [
            t.spec()
            for n, t in sorted(self._tools.items())
            if include is None or n in include
        ]

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._tools)]


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props}


def schema(required: list[str] | None = None, **props: Any) -> dict[str, Any]:
    s = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        s["required"] = required
    return s


def string(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, **kw}


def integer(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "integer", "description": desc, **kw}


def boolean(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "boolean", "description": desc, **kw}


def array(desc: str, items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "description": desc, "items": items}
