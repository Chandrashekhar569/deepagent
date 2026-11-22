"""Command-line entry point: interactive REPL, one-shot and pipe modes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import (CONFIG_FILE, Config, SESSIONS_DIR, detect_environment,
                     dump_env_json, ensure_dirs)
from .permissions import Decision, PermissionEngine
from .providers import ProviderError, build_provider
from .ui import UI


# --------------------------------------------------------------------- confirm
def make_confirm(ui: UI, auto: bool):
    def confirm(tool: str, args: dict, decision: Decision) -> str:
        if decision.verdict == "deny":
            ui.on_error(f"denied: {decision.reason}")
            return "deny"
        if auto:
            return "allow"
        subj = _confirm_subject(tool, args)
        ui.print()
        ui.system(f"  permission: {tool} — {decision.reason}")
        if subj:
            ui.print(f"  {subj}")
        try:
            ans = input("  allow? [y]es / [n]o / [a]lways (this kind) : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if ans in ("a", "always"):
            return "always"
        return "allow" if ans in ("y", "yes", "") else "deny"
    return confirm


def _confirm_subject(tool: str, args: dict) -> str:
    if tool == "Bash":
        return "$ " + str(args.get("command", ""))[:300]
    if tool in ("Write", "Edit", "MultiEdit"):
        return "file: " + str(args.get("file_path", ""))
    return json.dumps(args)[:300]


# ----------------------------------------------------------------- slash cmds
HELP = """commands:
  /help              show this help
  /model [name]      show or switch the model
  /models            list models the provider offers
  /profile [name]    show or switch provider profile
  /mode [name]       permission mode: ask | acceptEdits | plan | yolo
  /tools             list available tools
  /env               show detected environment
  /usage             token / request usage this session
  /save [title]      save this session to disk
  /resume [id]       list or resume saved sessions
  /clear             clear the conversation (keep settings)
  /reasoning         toggle showing model reasoning
  /config            show effective configuration
  /exit, /quit       leave
anything else is sent to the agent."""


def handle_slash(line: str, agent: Agent, ui: UI) -> bool:
    """Returns True if the REPL should continue, False to exit."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    cfg = agent.cfg

    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        ui.print(HELP)
    elif cmd == "/model":
        if arg:
            cfg.model = arg
            ui.system(f"model -> {arg}")
        else:
            ui.print(f"model: {cfg.model}  (profile {cfg.profile})")
    elif cmd == "/models":
        try:
            prov = build_provider(cfg)
            models = prov.list_models() if hasattr(prov, "list_models") else []
            ui.print("\n".join(models) or "(provider does not list models)")
        except Exception as exc:
            ui.on_error(str(exc))
    elif cmd == "/profile":
        if arg:
            new = Config.load({"profile": arg})
            agent.cfg = new
            agent.provider = build_provider(new)
            agent.perms = PermissionEngine(new)
            ui.system(f"profile -> {arg} ({new.provider}:{new.model})")
        else:
            ui.print(f"profile: {cfg.profile}")
    elif cmd == "/mode":
        if arg in ("ask", "acceptEdits", "plan", "yolo"):
            cfg.permission_mode = arg
            agent.perms.mode = arg
            ui.system(f"permission mode -> {arg}")
        else:
            ui.print(f"mode: {agent.perms.mode}  (ask|acceptEdits|plan|yolo)")
    elif cmd == "/tools":
        ui.print(", ".join(agent.registry.names()))
    elif cmd == "/env":
        ui.print(dump_env_json())
    elif cmd == "/usage":
        ui.usage_table(agent.ctx)
    elif cmd == "/save":
        path = agent.ctx.save_session(arg or None)
        ui.system(f"saved -> {path}")
    elif cmd == "/resume":
        _resume(agent, ui, arg)
    elif cmd == "/clear":
        agent.ctx.messages = []
        agent.ctx.todos = []
        ui.system("conversation cleared")
    elif cmd == "/reasoning":
        ui.show_reasoning = not ui.show_reasoning
        ui.system(f"show reasoning: {ui.show_reasoning}")
    elif cmd == "/config":
        ui.print(json.dumps(cfg.as_dict(), indent=2))
    else:
        ui.on_error(f"unknown command {cmd}; try /help")
    return True


def _resume(agent: Agent, ui: UI, arg: str) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    if not arg:
        if not sessions:
            ui.print("no saved sessions")
            return
        for p in sessions[:20]:
            try:
                d = json.loads(p.read_text())
                ui.print(f"  {p.stem}  {d.get('title','(untitled)')}  "
                         f"[{len(d.get('messages',[]))} msgs, {d.get('model','')}]")
            except Exception:
                continue
        ui.system("resume with: /resume <id>")
        return
    match = SESSIONS_DIR / f"{arg}.json"
    if not match.exists():
        cand = [p for p in sessions if arg in p.stem]
        if not cand:
            ui.on_error(f"no session matching {arg!r}")
            return
        match = cand[0]
    agent.ctx.load_session(match)
    ui.system(f"resumed {match.stem} ({len(agent.ctx.messages)} messages)")


# ---------------------------------------------------------------------- modes
def run_repl(agent: Agent, ui: UI) -> int:
    env = detect_environment()
    ui.banner(agent.cfg, env)
    if not agent.cfg.api_key and agent.cfg.api_key_env:
        ui.on_error(
            f"{agent.cfg.api_key_env} is not set. Export it, or run "
            f"`deepagent --profile ollama` for a local model.")
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from .config import CONFIG_DIR
        psession = PromptSession(history=FileHistory(str(CONFIG_DIR / "history")))
        read = lambda: psession.prompt("\n› ")  # noqa: E731
    except Exception:
        read = lambda: input("\n› ")  # noqa: E731

    while True:
        try:
            line = read()
        except (EOFError, KeyboardInterrupt):
            ui.print()
            break
        if not line.strip():
            continue
        if line.strip().startswith("/"):
            if not handle_slash(line, agent, ui):
                break
            continue
        try:
            agent.run(line)
        except KeyboardInterrupt:
            agent.interrupt()
            ui.on_error("interrupted")
        except ProviderError as exc:
            ui.on_error(str(exc))
    agent.ctx.close()
    return 0


def run_once(agent: Agent, ui: UI, prompt: str, as_json: bool) -> int:
    try:
        result = agent.run(prompt)
    except ProviderError as exc:
        ui.on_error(str(exc))
        return 2
    finally:
        agent.ctx.close()
    if as_json:
        print(json.dumps({
            "result": result,
            "usage": agent.ctx.usage,
            "changed_files": agent.ctx.changed_files,
        }, indent=2))
    return 0


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepagent",
        description="A deep coding agent for your terminal (Kali-ready). "
                    "Works with NVIDIA NIM, OpenAI-compatible, Ollama, llama.cpp "
                    "and Anthropic models.")
    p.add_argument("prompt", nargs="*", help="run this prompt once and exit")
    p.add_argument("-p", "--print", dest="oneshot", action="store_true",
                   help="non-interactive: print the result and exit")
    p.add_argument("--profile", help="provider profile (nvidia, ollama, llamacpp, ...)")
    p.add_argument("--model", help="override the model id")
    p.add_argument("--base-url", dest="base_url", help="override the API base URL")
    p.add_argument("--provider", help="provider impl: openai|anthropic|ollama")
    p.add_argument("--api-key", dest="api_key", help="API key (else read from env)")
    p.add_argument("-C", "--cwd", dest="workdir", help="working directory")
    p.add_argument("--mode", dest="permission_mode",
                   choices=["ask", "acceptEdits", "plan", "yolo"],
                   help="permission mode")
    p.add_argument("--yolo", action="store_true", help="shortcut for --mode yolo")
    p.add_argument("--allow", action="append", default=[], help="allow rule (repeatable)")
    p.add_argument("--deny", action="append", default=[], help="deny rule (repeatable)")
    p.add_argument("--no-web", action="store_true", help="disable web tools")
    p.add_argument("--no-stream", action="store_true", help="disable streaming")
    p.add_argument("--reasoning", action="store_true", help="show model reasoning")
    p.add_argument("--json", action="store_true", help="JSON output in -p mode")
    p.add_argument("--resume", metavar="ID", help="resume a saved session")
    p.add_argument("--list-models", action="store_true", help="list models and exit")
    p.add_argument("--setup", action="store_true", help="write a starter config file")
    p.add_argument("--env", action="store_true", help="print detected environment")
    p.add_argument("--version", action="version", version=f"deepagent {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    args = build_parser().parse_args(argv)

    if args.env:
        print(dump_env_json())
        return 0

    overrides = {
        "profile": args.profile, "model": args.model, "base_url": args.base_url,
        "provider": args.provider, "api_key": args.api_key, "workdir": args.workdir,
        "permission_mode": "yolo" if args.yolo else args.permission_mode,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    cfg = Config.load(overrides)
    if args.allow:
        cfg.allow += args.allow
    if args.deny:
        cfg.deny += args.deny
    if args.no_web:
        cfg.enable_web = False
    if args.no_stream:
        cfg.stream = False
    if args.workdir:
        import os
        os.chdir(cfg.workdir)

    if args.setup:
        path = cfg.save()
        print(f"wrote {path}\nedit it to set your profile, model and API key env.")
        return 0

    if args.list_models:
        try:
            prov = build_provider(cfg)
            for m in (prov.list_models() if hasattr(prov, "list_models") else []):
                print(m)
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    ui = UI(cfg, show_reasoning=args.reasoning)
    confirm = make_confirm(ui, auto=(cfg.permission_mode == "yolo"))
    try:
        agent = Agent(cfg, ui=ui, confirm=confirm)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.resume:
        _resume(agent, ui, args.resume)

    # piped stdin becomes the prompt
    piped = ""
    if not sys.stdin.isatty():
        try:
            piped = sys.stdin.read()
        except Exception:
            piped = ""

    prompt = " ".join(args.prompt).strip()
    if piped.strip():
        prompt = (prompt + "\n\n" + piped).strip() if prompt else piped.strip()

    if prompt and (args.oneshot or not sys.stdin.isatty()):
        return run_once(agent, ui, prompt, args.json)
    if prompt:
        agent.run(prompt)
    return run_repl(agent, ui)


if __name__ == "__main__":
    raise SystemExit(main())
