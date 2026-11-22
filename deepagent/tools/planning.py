"""Planning / productivity tools: TodoWrite, Think, and the Task subagent."""
from __future__ import annotations

from .base import ToolRegistry, ToolResult, array, boolean, schema, string

registry = ToolRegistry()


@registry.add(
    "TodoWrite",
    "Record and update the task list for the current work. Call it at the start "
    "of any multi-step task and whenever a step completes. Exactly one item "
    "should be 'in_progress' at a time.",
    schema(
        ["todos"],
        todos=array(
            "The full, ordered task list (replaces the previous list).",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                },
                "required": ["content", "status"],
            },
        ),
    ),
    read_only=True,
)
def _todo(args: dict, ctx) -> ToolResult:
    todos = args.get("todos") or []
    ctx.todos = todos
    ctx.render_todos(todos)
    done = sum(1 for t in todos if t.get("status") == "completed")
    cur = next((t["content"] for t in todos if t.get("status") == "in_progress"), None)
    summ = f"{done}/{len(todos)} done" + (f"; now: {cur[:40]}" if cur else "")
    return ToolResult.ok("Task list updated.", summary=summ)


@registry.add(
    "Think",
    "Use as a private scratchpad to reason through a hard problem, weigh options "
    "or plan before acting. The text is not shown to the user and takes no other "
    "action.",
    schema(["thought"], thought=string("Your reasoning.")),
    read_only=True,
)
def _think(args: dict, ctx) -> ToolResult:
    return ToolResult.ok("(noted)", summary="thinking")


@registry.add(
    "Task",
    "Delegate a self-contained piece of work to a fresh subagent that has the "
    "same tools but its own context window. Use it for wide searches or "
    "independent subtasks so the main context stays focused. The subagent "
    "runs to completion and returns only its final report.",
    schema(
        ["description", "prompt"],
        description=string("A 3-6 word label for the subtask."),
        prompt=string(
            "The full instructions for the subagent: what to do and exactly what "
            "to report back. It cannot ask follow-up questions."
        ),
        agent_type=string(
            "Optional role: 'explore' (read-only search), 'code' (can edit), or "
            "'general' (default).",
            enum=["explore", "code", "general"],
        ),
    ),
)
def _task(args: dict, ctx) -> ToolResult:
    if ctx.depth >= ctx.max_depth:
        return ToolResult.error(
            f"subagent depth limit ({ctx.max_depth}) reached; do this work directly."
        )
    return ctx.run_subagent(
        description=args["description"],
        prompt=args["prompt"],
        agent_type=args.get("agent_type") or "general",
    )


__all__ = ["registry"]
