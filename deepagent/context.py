"""Shared agent context: conversation state, file tracking, shell, jobs, usage."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import LOG_DIR, SESSIONS_DIR
from .providers import Message


def estimate_tokens(messages: list[Message]) -> int:
    """Cheap heuristic: ~4 chars per token over all serialized content."""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            total += len(json.dumps(c.get("arguments") or {}))
            total += len(c.get("name") or "")
    return total // 4 + 4 * len(messages)


class AgentContext:
    """Everything a tool handler and the loop need to share."""

    def __init__(self, cfg, *, depth: int = 0, max_depth: int = 3,
                 ui: Any = None, parent: "AgentContext | None" = None) -> None:
        self.cfg = cfg
        self.depth = depth
        self.max_depth = max_depth
        self.ui = ui
        self.parent = parent

        self.messages: list[Message] = []
        self.todos: list[dict] = []
        self._read_files: dict[str, float] = {}
        self.changed_files: dict[str, str] = {}
        self._backups: dict[str, str] = {}

        self.jobs: dict[str, Any] = {}
        self._shell = None
        self.shell_cwd: str = str(Path(cfg.workdir).resolve())
        self.log_dir: Path = LOG_DIR

        self.usage = {"input": 0, "output": 0, "requests": 0}
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        # injected by the Agent so tools can spawn subagents / render UI
        self.run_subagent: Callable[..., Any] = lambda **_: None  # type: ignore[assignment]

    # ----------------------------------------------------------- file tracking
    def mark_read(self, path: Path) -> None:
        self._read_files[str(path.resolve())] = time.time()

    def was_read(self, path: Path) -> bool:
        return str(path.resolve()) in self._read_files

    def backup(self, path: Path) -> None:
        try:
            self._backups.setdefault(str(path.resolve()),
                                     path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass

    def record_change(self, path: Path, action: str) -> None:
        self.changed_files[str(path.resolve())] = action
        if self.ui:
            self.ui.on_file_change(str(path), action)

    # ------------------------------------------------------------------- shell
    def get_shell(self):
        from .tools.shell import PersistentShell

        if self._shell is None or not self._shell.alive:
            self._shell = PersistentShell(self.shell_cwd or self.cfg.workdir)
        return self._shell

    def note_shell_cwd(self, cwd: str) -> None:
        if cwd:
            self.shell_cwd = cwd

    def close(self) -> None:
        if self._shell is not None:
            self._shell.close()
        for job in list(self.jobs.values()):
            try:
                if job.proc.poll() is None:
                    job.proc.terminate()
            except Exception:
                pass

    # ---------------------------------------------------------------------- UI
    def render_todos(self, todos: list[dict]) -> None:
        if self.ui:
            self.ui.on_todos(todos)

    # ------------------------------------------------------------- persistence
    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created": time.time(),
            "profile": self.cfg.profile,
            "model": self.cfg.model,
            "workdir": self.cfg.workdir,
            "messages": self.messages,
            "todos": self.todos,
            "usage": self.usage,
            "changed_files": self.changed_files,
        }

    def save_session(self, title: str | None = None) -> Path:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{self.session_id}.json"
        data = self.snapshot()
        if title:
            data["title"] = title
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def load_session(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.messages = data.get("messages", [])
        self.todos = data.get("todos", [])
        self.usage = data.get("usage", self.usage)
        self.changed_files = data.get("changed_files", {})
        self.session_id = data.get("session_id", self.session_id)
