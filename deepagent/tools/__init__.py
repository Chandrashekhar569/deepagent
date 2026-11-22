"""Assemble the full tool registry from the individual modules."""
from __future__ import annotations

from .base import Tool, ToolRegistry, ToolResult
from . import fs, planning, shell, web


def build_registry(cfg) -> ToolRegistry:
    reg = ToolRegistry()
    for mod in (fs, shell, planning):
        for tool in mod.registry.all():
            reg.register(tool)
    if cfg.enable_web:
        for tool in web.registry.all():
            reg.register(tool)
    return reg


__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_registry"]
