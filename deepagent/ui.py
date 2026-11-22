"""Terminal UI built on `rich`, with a plain fallback if rich is absent."""
from __future__ import annotations

import json
from typing import Any

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


TOOL_ICON = {
    "Read": "読", "Write": "✎", "Edit": "✎", "MultiEdit": "✎", "Bash": "$",
    "Glob": "⿻", "Grep": "⌕", "LS": "☰", "WebSearch": "⌕", "WebFetch": "⤓",
    "TodoWrite": "☑", "Task": "⇲", "Think": "…", "BashOutput": "⟳", "KillBash": "✕",
}


class UI:
    def __init__(self, cfg, *, quiet: bool = False, show_reasoning: bool = False) -> None:
        self.cfg = cfg
        self.quiet = quiet
        self.show_reasoning = show_reasoning
        self._rich = _HAS_RICH and not cfg.__dict__.get("no_color")
        self.console = Console(highlight=False) if self._rich else None
        self._streaming = False
        self._buf: list[str] = []
        self._reason_open = False

    # ----------------------------------------------------------------- output
    def print(self, *a, **k) -> None:
        if self.console:
            self.console.print(*a, **k)
        else:
            print(*(str(x) for x in a))

    def rule(self, text: str = "") -> None:
        if self.console:
            self.console.rule(text) if text else self.console.print()
        else:
            print(f"--- {text} ---" if text else "")

    def system(self, text: str) -> None:
        if self.console:
            self.console.print(Text(text, style="dim"))
        else:
            print(text)

    def banner(self, cfg, env: dict) -> None:
        name = f"deepagent · {cfg.profile} · {cfg.model}"
        where = env.get("distro") or env.get("os", "")
        sub = f"{where} · {env.get('cwd','')}"
        if self.console:
            self.console.print(Panel.fit(
                f"[bold cyan]{name}[/]\n[dim]{sub}[/]\n"
                "[dim]/help for commands · Ctrl-C to interrupt · /exit to quit[/]",
                border_style="cyan"))
        else:
            print(f"{name}\n{sub}\n/help for commands")

    # ---------------------------------------------------------- agent callbacks
    def on_assistant_start(self) -> None:
        self._streaming = True
        self._buf = []

    def on_text(self, text: str) -> None:
        self._buf.append(text)
        if self.console:
            self.console.print(text, end="", soft_wrap=True, markup=False, highlight=False)
        else:
            print(text, end="", flush=True)

    def on_reasoning(self, text: str) -> None:
        if not self.show_reasoning:
            return
        if self.console:
            self.console.print(Text(text, style="dim italic"), end="", markup=False)
        else:
            print(text, end="", flush=True)

    def on_assistant_end(self, text: str, tool_calls: list[dict]) -> None:
        if self._buf:
            self.print()  # newline after streamed text
        self._streaming = False

    def on_tool_start(self, name: str, args: dict, decision) -> None:
        icon = TOOL_ICON.get(name, "•")
        subj = _subject(name, args)
        line = f"  [{icon}] {name}" + (f"  {subj}" if subj else "")
        if self.console:
            style = "yellow" if decision.verdict != "allow" else "cyan"
            self.console.print(Text(line, style=style))
        else:
            print(line)

    def on_tool_end(self, name: str, result, declined: bool = False) -> None:
        if declined:
            return
        summ = result.summary or ("error" if result.is_error else "ok")
        dur = result.meta.get("duration")
        tail = f"  ({dur}s)" if dur else ""
        if self.console:
            style = "red" if result.is_error else "green"
            self.console.print(Text(f"      → {summ}{tail}", style=f"dim {style}"))
        else:
            print(f"      -> {summ}{tail}")

    def on_file_change(self, path: str, action: str) -> None:
        pass  # already summarized by on_tool_end

    def on_todos(self, todos: list[dict]) -> None:
        if self.quiet:
            return
        glyph = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
        if self.console:
            t = Text()
            for td in todos:
                g = glyph.get(td.get("status"), "[ ]")
                style = {"completed": "dim green", "in_progress": "bold yellow"}.get(
                    td.get("status"), "white")
                t.append(f"  {g} {td.get('content','')}\n", style=style)
            self.console.print(Panel(t, title="tasks", border_style="dim", expand=False))
        else:
            for td in todos:
                print(f"  {glyph.get(td.get('status'),'[ ]')} {td.get('content','')}")

    def on_subagent_start(self, description: str, agent_type: str) -> None:
        self.print(f"  [⇲] subagent ({agent_type}): {description}")

    def on_subagent_end(self, description: str, report: str) -> None:
        self.print(f"      → subagent '{description}' finished")

    def on_compact_start(self, used: int, window: int) -> None:
        self.system(f"  … compacting context ({used} / {window} tokens)")

    def on_compact_end(self, n_messages: int) -> None:
        self.system(f"  … context compacted to {n_messages} messages")

    def on_error(self, message: str) -> None:
        if self.console:
            self.console.print(Text(f"  ⚠ {message}", style="bold red"))
        else:
            print(f"  ! {message}")

    def markdown(self, text: str) -> None:
        if self.console:
            self.console.print(Markdown(text))
        else:
            print(text)

    def usage_table(self, ctx) -> None:
        u = ctx.usage
        rows = [
            ("requests", str(u["requests"])),
            ("input tokens", f"{u['input']:,}"),
            ("output tokens", f"{u['output']:,}"),
            ("files changed", str(len(ctx.changed_files))),
        ]
        if self.console:
            t = Table(show_header=False, box=None)
            for k, v in rows:
                t.add_row(Text(k, style="dim"), v)
            self.console.print(t)
        else:
            for k, v in rows:
                print(f"{k}: {v}")


def _subject(name: str, args: dict) -> str:
    if name == "Bash":
        return (args.get("description") or args.get("command", ""))[:70]
    for k in ("file_path", "path", "pattern", "query", "url", "job_id", "description"):
        if args.get(k):
            return str(args[k])[:70]
    return ""


class NullUI(UI):
    """Used by subagents - swallows everything."""

    def __init__(self) -> None:  # noqa: D401
        self.console = None
        self.quiet = True
        self.show_reasoning = False
        self._buf = []

    def __getattribute__(self, item):
        if item.startswith("on_") or item in ("print", "system", "rule", "markdown",
                                               "banner", "usage_table", "error"):
            return lambda *a, **k: None
        return object.__getattribute__(self, item)
