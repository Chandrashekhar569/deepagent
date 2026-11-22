"""Shell tools: a persistent interactive shell plus background job control.

The shell is a long-lived `bash` process, so `cd`, `export`, activated
virtualenvs, and shell functions persist between calls - the way a human
terminal behaves across a multi-step task.
"""
from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .base import ToolRegistry, ToolResult, boolean, integer, schema, string

registry = ToolRegistry()

SENTINEL = "__DEEPAGENT_DONE__"


def _pick_shell() -> list[str]:
    for cand in (os.environ.get("DEEPAGENT_SHELL"), "/bin/bash", "bash", "/bin/sh", "sh"):
        if not cand:
            continue
        path = shutil.which(cand) if not os.path.isabs(cand) else cand
        if path and Path(path).exists():
            return [path, "-i"] if "bash" in path else [path]
    if os.name == "nt":
        return ["powershell", "-NoLogo", "-NoProfile", "-Command", "-"]
    raise RuntimeError("no POSIX shell found")


class PersistentShell:
    """One bash process, reused for every Bash call."""

    def __init__(self, cwd: str, env: dict[str, str] | None = None) -> None:
        argv = _pick_shell()
        self.is_posix = "powershell" not in argv[0].lower()
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {}), "TERM": "dumb", "PAGER": "cat",
                 "GIT_PAGER": "cat", "DEBIAN_FRONTEND": "noninteractive"},
        )
        self._q: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._lock = threading.Lock()
        if self.is_posix:
            self._send("export PS1=''; set +o histexpand 2>/dev/null; stty -echo 2>/dev/null")
            self.run("true", timeout=15)

    def _pump(self) -> None:
        assert self.proc.stdout
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put("")

    def _send(self, text: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(self, command: str, timeout: int = 180) -> tuple[str, "int | None", bool]:
        """Returns (output, exit_code, timed_out)."""
        if not self.alive:
            return ("shell has exited; it will be restarted on the next call", None, False)
        with self._lock:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            token = f"{SENTINEL}{uuid.uuid4().hex[:8]}"
            if self.is_posix:
                self._send(f"{command}\nprintf '\\n%s:%s\\n' '{token}' \"$?\"")
            else:
                self._send(f"{command}\nWrite-Output \"`n{token}:$LASTEXITCODE\"")

            deadline = time.monotonic() + timeout
            out: list[str] = []
            code: "int | None" = None
            timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = self._q.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if line == "":
                    break
                if token in line:
                    _, _, tail = line.partition(token + ":")
                    try:
                        code = int(tail.strip())
                    except ValueError:
                        code = None
                    break
                out.append(line)
            if timed_out:
                self.interrupt()
            return ("".join(out), code, timed_out)

    def interrupt(self) -> None:
        try:
            if self.is_posix and self.proc.pid:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)  # type: ignore[attr-defined]
        except Exception:
            try:
                self.proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    def cwd(self) -> str:
        out, _, _ = self.run("pwd", timeout=10)
        return out.strip().splitlines()[-1] if out.strip() else ""

    def close(self) -> None:
        try:
            if self.alive:
                self._send("exit")
                self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


@dataclass
class BackgroundJob:
    id: str
    command: str
    proc: subprocess.Popen
    log: Path
    started: float = field(default_factory=time.time)
    cursor: int = 0

    @property
    def status(self) -> str:
        rc = self.proc.poll()
        return "running" if rc is None else f"exited({rc})"

    def read_new(self, max_chars: int = 20000) -> str:
        try:
            with self.log.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.cursor)
                data = fh.read()
                self.cursor = fh.tell()
        except OSError:
            return ""
        return data[-max_chars:]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [output truncated, {len(text)} chars total] ...\n\n{tail}"


# ---------------------------------------------------------------------- Bash
@registry.add(
    "Bash",
    "Run a command in a persistent shell session. State such as the working "
    "directory, environment variables and activated virtualenvs carries over "
    "between calls. Set background=true for long-running servers or watchers "
    "and poll them with BashOutput. Prefer the Read/Glob/Grep tools over cat, "
    "find and grep.",
    schema(
        ["command"],
        command=string("The command line to run."),
        timeout=integer("Seconds before the command is interrupted (default 180)."),
        background=boolean("Run detached; returns a job id for BashOutput."),
        description=string("Short description of what the command does."),
    ),
)
def _bash(args: dict, ctx) -> ToolResult:
    command = str(args.get("command", "")).strip()
    if not command:
        return ToolResult.error("command is empty.")
    timeout = int(args.get("timeout") or ctx.cfg.bash_timeout)

    if args.get("background"):
        return _run_background(command, ctx)

    shell = ctx.get_shell()
    out, code, timed_out = shell.run(command, timeout=timeout)
    out = _truncate(out.rstrip("\n"), ctx.cfg.max_tool_output)
    ctx.note_shell_cwd(shell.cwd())

    if timed_out:
        body = out + f"\n\n[timed out after {timeout}s and was interrupted (SIGINT)]"
        return ToolResult(content=body or "[no output before timeout]",
                          is_error=True, summary=f"timeout: {command[:40]}")
    header = "" if code in (0, None) else f"[exit code {code}]\n"
    body = header + (out or "[no output]")
    summ = f"$ {command[:48]}" + (f" -> exit {code}" if code not in (0, None) else "")
    return ToolResult(content=body, is_error=bool(code), summary=summ)


def _run_background(command: str, ctx) -> ToolResult:
    jid = "job_" + uuid.uuid4().hex[:8]
    log = ctx.log_dir / f"{jid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w", encoding="utf-8")
    argv = _pick_shell()
    proc = subprocess.Popen(
        argv[:1] + (["-c", command] if argv[0].endswith(("sh", "bash")) else ["-Command", command]),
        stdout=fh, stderr=subprocess.STDOUT, cwd=ctx.shell_cwd or ctx.cfg.workdir, text=True,
        env={**os.environ, "TERM": "dumb"},
    )
    ctx.jobs[jid] = BackgroundJob(jid, command, proc, log)
    return ToolResult.ok(
        f"Started background job {jid} (pid {proc.pid}).\n"
        f"Command: {command}\nPoll it with BashOutput(job_id='{jid}').",
        summary=f"bg {jid}: {command[:40]}",
    )


@registry.add(
    "BashOutput",
    "Read new output from a background job started with Bash(background=true), "
    "and report whether it is still running.",
    schema(
        ["job_id"],
        job_id=string("The job id returned by Bash."),
    ),
    read_only=True,
)
def _bash_output(args: dict, ctx) -> ToolResult:
    job = ctx.jobs.get(args["job_id"])
    if not job:
        known = ", ".join(ctx.jobs) or "none"
        return ToolResult.error(f"no such job. Known jobs: {known}")
    new = job.read_new(ctx.cfg.max_tool_output)
    status = job.status
    body = f"[{job.id} status: {status}]\n" + (new or "[no new output]")
    return ToolResult.ok(body, summary=f"{job.id}: {status}")


@registry.add(
    "KillBash",
    "Terminate a background job started with Bash(background=true).",
    schema(["job_id"], job_id=string("The job id to kill.")),
)
def _kill_bash(args: dict, ctx) -> ToolResult:
    job = ctx.jobs.get(args["job_id"])
    if not job:
        return ToolResult.error("no such job.")
    if job.proc.poll() is None:
        try:
            job.proc.terminate()
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.proc.kill()
        except Exception as exc:
            return ToolResult.error(f"could not kill {job.id}: {exc}")
    return ToolResult.ok(f"Killed {job.id} ({job.command[:50]}).", summary=f"killed {job.id}")


__all__ = ["registry", "PersistentShell", "BackgroundJob"]
