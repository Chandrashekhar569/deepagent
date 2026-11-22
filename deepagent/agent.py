"""The core agentic loop: model <-> tools, with permissions and compaction."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from .context import AgentContext, estimate_tokens
from .permissions import Decision, PermissionEngine
from .prompts import build_subagent_prompt, build_system_prompt
from .providers import Provider, ProviderError, build_provider
from .tools import ToolRegistry, ToolResult, build_registry

# Tools the user approves via the permission engine. Returning "deny" aborts
# the call; the agent is told and continues.
ConfirmFn = Callable[[str, dict, Decision], str]  # -> "allow"|"deny"|"always"


class Agent:
    def __init__(self, cfg, *, ui: Any = None, confirm: ConfirmFn | None = None,
                 context: AgentContext | None = None,
                 provider: Provider | None = None,
                 registry: ToolRegistry | None = None,
                 perms: PermissionEngine | None = None) -> None:
        self.cfg = cfg
        self.ui = ui
        self.provider = provider or build_provider(cfg)
        self.registry = registry or build_registry(cfg)
        self.perms = perms or PermissionEngine(cfg)
        self.confirm = confirm or (lambda t, a, d: "allow" if d.verdict == "allow" else "deny")
        self.ctx = context or AgentContext(cfg, ui=ui)
        self.ctx.run_subagent = self._spawn_subagent
        self.system_prompt = build_system_prompt(cfg)
        self._interrupt = False
        self.mcp_servers = []
        if context is None and getattr(cfg, "mcp_servers", None):
            try:
                from .mcp import load_mcp_servers
                self.mcp_servers = load_mcp_servers(cfg, self.registry, ui)
            except Exception as exc:  # never fail startup over MCP
                if ui:
                    ui.on_error(f"mcp init failed: {exc}")

    # ------------------------------------------------------------------- setup
    def _ensure_system(self) -> None:
        if not self.ctx.messages or self.ctx.messages[0]["role"] != "system":
            self.ctx.messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            self.ctx.messages[0]["content"] = self.system_prompt

    def interrupt(self) -> None:
        self._interrupt = True

    # -------------------------------------------------------------------- main
    def run(self, user_input: str | None) -> str:
        """Run one user turn to completion; returns the final assistant text."""
        self._ensure_system()
        if user_input is not None:
            self.ctx.messages.append({"role": "user", "content": user_input})
        self._interrupt = False

        final_text = ""
        turns = 0
        while turns < self.cfg.max_turns and not self._interrupt:
            turns += 1
            self._maybe_compact()
            try:
                assistant, tool_calls = self._one_model_step()
            except ProviderError as exc:
                if self.ui:
                    self.ui.on_error(str(exc))
                return f"[provider error] {exc}"

            final_text = assistant or final_text
            if not tool_calls:
                break

            # execute every requested tool call, append results
            results: list[dict] = []
            for call in tool_calls:
                if self._interrupt:
                    results.append(self._tool_message(call["id"],
                                   "[interrupted by user before execution]"))
                    continue
                results.append(self._execute_tool(call))
            self.ctx.messages.extend(results)

        if turns >= self.cfg.max_turns and self.ui:
            self.ui.on_error(f"stopped after {self.cfg.max_turns} turns")
        return final_text

    # -------------------------------------------------------------- model step
    def _one_model_step(self) -> tuple[str, list[dict]]:
        specs = self.registry.specs()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        if self.ui:
            self.ui.on_assistant_start()

        for ev in self.provider.stream(self.ctx.messages, specs):
            if self._interrupt:
                break
            t = ev["type"]
            if t == "text":
                text_parts.append(ev["text"])
                if self.ui:
                    self.ui.on_text(ev["text"])
            elif t == "reasoning":
                if self.ui:
                    self.ui.on_reasoning(ev["text"])
            elif t == "tool_call":
                tool_calls.append(ev["call"])
            elif t == "usage":
                self.ctx.usage["input"] += ev.get("input", 0)
                self.ctx.usage["output"] += ev.get("output", 0)
            elif t == "done":
                pass
            elif t == "error":
                raise ProviderError(ev["message"])

        self.ctx.usage["requests"] += 1
        text = "".join(text_parts)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.ctx.messages.append(assistant_msg)
        if self.ui:
            self.ui.on_assistant_end(text, tool_calls)
        return text, tool_calls

    # ------------------------------------------------------------ tool exec
    def _execute_tool(self, call: dict) -> dict:
        name, args = call["name"], call.get("arguments") or {}
        tool = self.registry.get(name)
        if tool is None:
            return self._tool_message(
                call["id"], f"Error: unknown tool {name!r}. Available: "
                + ", ".join(self.registry.names()),
            )

        decision = self.perms.check(name, args)
        if self.ui:
            self.ui.on_tool_start(name, args, decision)

        if decision.verdict != "allow":
            verdict = self.confirm(name, args, decision)
            if verdict == "always":
                self.perms.add_session_allow(self._always_rule(name, args))
                verdict = "allow"
            if verdict != "allow":
                msg = f"[user declined this {name} call: {decision.reason}]"
                if self.ui:
                    self.ui.on_tool_end(name, ToolResult.error("declined by user"), declined=True)
                return self._tool_message(call["id"], msg)

        start = time.time()
        result = tool.run(args, self.ctx)
        result.meta["duration"] = round(time.time() - start, 2)
        if self.ui:
            self.ui.on_tool_end(name, result)
        return self._tool_message(call["id"], result.content)

    @staticmethod
    def _always_rule(name: str, args: dict) -> str:
        if name == "Bash":
            cmd = str(args.get("command", "")).strip().split()
            prefix = " ".join(cmd[:2]) if cmd else ""
            return f"Bash({prefix}:*)"
        if name in ("Write", "Edit", "MultiEdit"):
            return f"{name}(*)"
        return name

    @staticmethod
    def _tool_message(call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    # ------------------------------------------------------------- compaction
    def _maybe_compact(self) -> None:
        window = self.cfg.context_window
        used = estimate_tokens(self.ctx.messages) + self.cfg.max_tokens
        if used < window * self.cfg.auto_compact_at:
            return
        if len(self.ctx.messages) < 8:
            return
        if self.ui:
            self.ui.on_compact_start(used, window)
        self._compact()

    def _compact(self) -> None:
        msgs = self.ctx.messages
        system = msgs[0]
        # keep the last few exchanges verbatim, summarize the middle
        keep_tail = 6
        head = msgs[1:-keep_tail] if len(msgs) > keep_tail + 1 else []
        tail = msgs[-keep_tail:]
        if not head:
            return
        transcript = _render_for_summary(head)
        summary_prompt = [
            {"role": "system", "content": "Summarize this partial coding-agent "
             "transcript so work can continue without the full history. Preserve: "
             "the user's goal, decisions made, files created/edited and their "
             "purpose, commands run and their outcomes, current state, and the "
             "immediate next steps. Be specific about file paths and names. "
             "Output only the summary."},
            {"role": "user", "content": transcript[: self.cfg.context_window * 2]},
        ]
        try:
            model = self.cfg.small_model or self.cfg.model
            summary = self.provider.complete(summary_prompt, model=model, max_tokens=1500)
        except ProviderError:
            summary = _render_for_summary(head)[:4000]
        note = {"role": "user", "content":
                "[Earlier conversation was compacted to save context. Summary of "
                f"what happened so far:]\n\n{summary}"}
        changed = ""
        if self.ctx.changed_files:
            changed = "\n\nFiles changed so far: " + ", ".join(
                f"{Path(p).name} ({a})" for p, a in self.ctx.changed_files.items()
            )
            note["content"] += changed
        self.ctx.messages = [system, note, *tail]
        if self.ui:
            self.ui.on_compact_end(len(self.ctx.messages))

    # --------------------------------------------------------------- subagents
    def _spawn_subagent(self, *, description: str, prompt: str,
                        agent_type: str = "general") -> ToolResult:
        if self.ui:
            self.ui.on_subagent_start(description, agent_type)
        sub_cfg = self.cfg
        child_ctx = AgentContext(
            sub_cfg, depth=self.ctx.depth + 1, max_depth=self.ctx.max_depth,
            ui=None, parent=self.ctx,
        )
        child_ctx.shell_cwd = self.ctx.shell_cwd
        # explore subagents get a read-only tool set
        registry = self.registry
        if agent_type == "explore":
            registry = build_registry(sub_cfg)
            for t in list(registry._tools.values()):  # type: ignore[attr-defined]
                if not t.read_only:
                    registry._tools.pop(t.name, None)  # type: ignore[attr-defined]
        perms = PermissionEngine(sub_cfg)
        if agent_type != "explore":
            perms.mode = "acceptEdits" if self.perms.mode in ("yolo", "acceptEdits") else self.perms.mode
        sub = Agent(sub_cfg, ui=None, confirm=self.confirm, context=child_ctx,
                    provider=self.provider, registry=registry, perms=perms)
        sub.system_prompt = build_subagent_prompt(sub_cfg, agent_type)
        try:
            report = sub.run(prompt)
        finally:
            child_ctx.close()
        self.ctx.usage["input"] += child_ctx.usage["input"]
        self.ctx.usage["output"] += child_ctx.usage["output"]
        # merge the child's file knowledge up so the parent can edit them
        for p in child_ctx.changed_files:
            self.ctx._read_files[p] = time.time()  # type: ignore[attr-defined]
        self.ctx.changed_files.update(child_ctx.changed_files)
        if self.ui:
            self.ui.on_subagent_end(description, report)
        return ToolResult.ok(report or "[subagent returned no report]",
                             summary=f"subagent '{description}' done")


from pathlib import Path  # noqa: E402  (used in _compact)


def _render_for_summary(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append(f"[tool result] {(m.get('content') or '')[:1500]}")
        elif role == "assistant":
            if m.get("content"):
                out.append(f"[assistant] {m['content']}")
            for c in m.get("tool_calls") or []:
                out.append(f"[assistant calls {c['name']}] "
                           f"{json.dumps(c.get('arguments') or {})[:600]}")
        elif role == "user":
            out.append(f"[user] {(m.get('content') or '')[:2000]}")
    return "\n".join(out)
