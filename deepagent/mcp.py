"""Minimal MCP (Model Context Protocol) client - stdio servers only.

Lets deepagent pull in external tool servers declared in config:

    [mcp.filesystem]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]

Each remote tool is exposed to the model as `mcp__<server>__<tool>`.
This is a small, dependency-free JSON-RPC-over-stdio implementation covering
initialize / tools/list / tools/call. Servers that need HTTP/SSE are out of
scope here.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any

from .tools.base import Tool, ToolRegistry, ToolResult


class MCPServer:
    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        self.name = name
        self.command = spec.get("command")
        self.args = spec.get("args") or []
        self.env = spec.get("env") or {}
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self.tools: list[dict] = []

    def start(self) -> None:
        if not self.command:
            raise RuntimeError(f"mcp server {self.name}: no command")
        self.proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env={**os.environ, **self.env},
        )
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "deepagent", "version": "0.1.0"},
        })
        self._notify("notifications/initialized", {})
        resp = self._request("tools/list", {})
        self.tools = (resp or {}).get("tools", [])

    def _send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, timeout: float = 30) -> dict | None:
        assert self.proc and self.proc.stdout
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            for _ in range(10000):
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"mcp server {self.name} closed the connection")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(str(msg["error"]))
                    return msg.get("result")
        return None

    def call(self, tool: str, arguments: dict) -> str:
        result = self._request("tools/call", {"name": tool, "arguments": arguments}, timeout=300)
        parts = []
        for block in (result or {}).get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block))
        if (result or {}).get("isError"):
            return "Error: " + ("\n".join(parts) or "mcp tool error")
        return "\n".join(parts) or "(no content)"

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


def load_mcp_servers(cfg, registry: ToolRegistry, ui=None) -> list[MCPServer]:
    servers: list[MCPServer] = []
    for name, spec in (cfg.mcp_servers or {}).items():
        if not isinstance(spec, dict):
            continue
        server = MCPServer(name, spec)
        try:
            server.start()
        except Exception as exc:
            if ui:
                ui.on_error(f"mcp server {name!r} failed to start: {exc}")
            continue
        for t in server.tools:
            _register_remote(registry, server, t)
        servers.append(server)
        if ui:
            ui.system(f"  mcp: {name} ({len(server.tools)} tools)")
    return servers


def _register_remote(registry: ToolRegistry, server: MCPServer, t: dict) -> None:
    tool_name = t["name"]
    full = f"mcp__{server.name}__{tool_name}"
    schema = t.get("inputSchema") or {"type": "object", "properties": {}}

    def handler(args: dict, ctx, _s=server, _n=tool_name) -> ToolResult:
        out = _s.call(_n, args)
        return ToolResult(content=out, is_error=out.startswith("Error:"),
                          summary=f"{_s.name}:{_n}")

    registry.register(Tool(
        name=full,
        description=(t.get("description") or f"{tool_name} (from MCP server {server.name})")[:1024],
        parameters=schema, handler=handler, read_only=False,
    ))
