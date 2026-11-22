"""Permission engine: decide allow / ask / deny for every tool call."""
from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

# Commands that are never run without an explicit interactive confirmation,
# regardless of mode (yolo excepted only where the user opted in via --allow).
HARD_DANGEROUS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*(-rf|-fr|-r\s+-f)\b.*\s/(\s|$)",
    r"\bmkfs(\.|\s)",
    r"\bdd\s+[^\n]*of=/dev/",
    r":\(\)\s*\{.*\};\s*:",            # fork bomb
    r"\bshred\b",
    r"\bchmod\s+(-R\s+)?777\s+/(\s|$)",
    r">\s*/dev/sd[a-z]",
    r"\bhistory\s+-c\b",
    r"\biptables\s+-F\b",
    r"\buserdel\b|\bpasswd\s+root\b",
    r"\bcryptsetup\b.*\b(luksFormat|erase)\b",
]

# Read-only shell commands that are safe to auto-allow.
SAFE_BINS = {
    "ls", "cat", "head", "tail", "wc", "file", "stat", "pwd", "which", "whoami",
    "id", "uname", "env", "date", "echo", "basename", "dirname", "realpath",
    "grep", "egrep", "rg", "find", "fd", "sort", "uniq", "cut", "tr", "sed",
    "awk", "diff", "jq", "yq", "tree", "du", "df", "ps", "top", "free", "nproc",
    "git", "python3", "python", "pip", "pip3", "node", "npm", "npx", "go",
    "cargo", "rustc", "gcc", "g++", "make", "cmake", "pytest", "ruff", "mypy",
    "black", "nvidia-smi", "ollama", "docker", "curl", "wget", "ss", "ip",
    "nmap", "dig", "host", "nslookup", "ping", "traceroute",
}
# git subcommands that write to the outside world need confirmation
GIT_WRITE = {"push", "reset", "clean", "rebase", "commit", "tag", "remote", "submodule"}


@dataclass
class Decision:
    verdict: str  # "allow" | "ask" | "deny"
    reason: str = ""


class PermissionEngine:
    """Rules look like `Tool`, `Tool(pattern)` e.g. `Bash(git status:*)`,
    `Edit(src/**)`, `WebFetch(domain:example.com)`."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mode = cfg.permission_mode
        self.allow = list(cfg.allow or [])
        self.deny = list(cfg.deny or [])
        self.session_allow: set[str] = set()
        self.root = Path(cfg.workdir).resolve()

    # ------------------------------------------------------------------- rules
    def add_session_allow(self, rule: str) -> None:
        self.session_allow.add(rule)

    @staticmethod
    def _match(rule: str, tool: str, subject: str) -> bool:
        if "(" not in rule:
            return rule == tool or rule == "*"
        rtool, _, rest = rule.partition("(")
        pattern = rest.rstrip(")")
        if rtool not in (tool, "*"):
            return False
        if pattern in ("*", ""):
            return True
        if pattern.startswith("domain:"):
            dom = pattern.split(":", 1)[1]
            return subject == dom or subject.endswith("." + dom)
        if pattern.endswith(":*"):
            return subject.startswith(pattern[:-2])
        return fnmatch.fnmatch(subject, pattern)

    def _rules_hit(self, rules: list[str] | set[str], tool: str, subject: str) -> bool:
        return any(self._match(r, tool, subject) for r in rules)

    # -------------------------------------------------------------------- main
    def check(self, tool: str, args: dict) -> Decision:
        subject = self.subject_for(tool, args)

        if self._rules_hit(self.deny, tool, subject):
            return Decision("deny", "matched a deny rule")

        if tool == "Bash":
            cmd = str(args.get("command", ""))
            for pat in HARD_DANGEROUS:
                if re.search(pat, cmd):
                    return Decision("ask", "command looks destructive")

        if self.mode == "plan" and self._is_write(tool):
            return Decision("deny", "plan mode: no changes until the plan is approved")

        if self._rules_hit(self.allow, tool, subject) or self._rules_hit(
            self.session_allow, tool, subject
        ):
            return Decision("allow", "allowed by rule")

        if self.mode == "yolo":
            return Decision("allow", "yolo mode")

        if tool in ("Read", "Glob", "Grep", "LS", "TodoWrite", "Task", "WebSearch",
                    "WebFetch", "Memory", "Think"):
            return Decision("allow", "read-only tool")

        if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit", "Patch"):
            if not self._inside_root(args):
                return Decision("ask", "path is outside the working directory")
            if self.mode == "acceptEdits":
                return Decision("allow", "acceptEdits mode")
            return Decision("ask", "file write")

        if tool == "Bash":
            if self._is_safe_command(str(args.get("command", ""))):
                return Decision("allow", "read-only command")
            return Decision("ask", "shell command")

        return Decision("ask", "needs confirmation")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def subject_for(tool: str, args: dict) -> str:
        if tool == "Bash":
            return str(args.get("command", ""))
        if tool in ("WebFetch",):
            url = str(args.get("url", ""))
            m = re.match(r"https?://([^/]+)", url)
            return m.group(1) if m else url
        for key in ("file_path", "path", "pattern", "notebook_path"):
            if args.get(key):
                return str(args[key])
        return ""

    @staticmethod
    def _is_write(tool: str) -> bool:
        return tool in {
            "Write", "Edit", "MultiEdit", "NotebookEdit", "Patch", "Bash", "Apply",
        }

    def _inside_root(self, args: dict) -> bool:
        p = args.get("file_path") or args.get("path") or args.get("notebook_path")
        if not p:
            return True
        try:
            return self.root in Path(str(p)).resolve().parents or Path(
                str(p)
            ).resolve() == self.root
        except OSError:
            return False

    def _is_safe_command(self, cmd: str) -> bool:
        if not cmd.strip():
            return False
        # any redirection, substitution or chaining -> ask
        if re.search(r"[>|`$]|&&|\|\||;|\$\(", cmd):
            return False
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return False
        if not parts:
            return False
        bin_ = Path(parts[0]).name
        if bin_ == "sudo":
            return False
        if bin_ not in SAFE_BINS:
            return False
        if bin_ == "git" and len(parts) > 1 and parts[1] in GIT_WRITE:
            return False
        if bin_ in ("pip", "pip3", "npm", "docker", "cargo", "go") and len(parts) > 1:
            return parts[1] in ("list", "show", "version", "--version", "ps", "images",
                                "info", "freeze", "env", "config")
        if bin_ in ("python", "python3") and "-c" in parts:
            return False
        if bin_ in ("sed", "awk") and "-i" in parts:
            return False
        if bin_ in ("curl", "wget") and any(
            p in ("-o", "-O", "--output") for p in parts
        ):
            return False
        if bin_ == "nmap":  # scanning is outward-facing: always confirm
            return False
        return True
