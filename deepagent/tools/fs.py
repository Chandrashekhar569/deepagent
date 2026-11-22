"""Filesystem tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS."""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import ToolRegistry, ToolResult, array, boolean, integer, schema, string

registry = ToolRegistry()

MAX_READ_LINES = 2000
MAX_LINE_LEN = 2000
BINARY_SNIFF = 8192
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", "target", ".ruff_cache", ".tox",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def _resolve(ctx, p: str) -> Path:
    path = Path(os.path.expanduser(str(p)))
    if not path.is_absolute():
        path = Path(ctx.cfg.workdir).resolve() / path
    return path


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(BINARY_SNIFF)
    except OSError:
        return False


# --------------------------------------------------------------------- Read
@registry.add(
    "Read",
    "Read a file from disk. Returns the contents with 1-indexed line numbers "
    "(cat -n style). Use offset/limit for large files. Always read a file "
    "before editing it.",
    schema(
        ["file_path"],
        file_path=string("Absolute or workdir-relative path to read."),
        offset=integer("1-indexed line to start at.", minimum=1),
        limit=integer("How many lines to read (default 2000)."),
    ),
    read_only=True,
)
def _read(args: dict, ctx) -> ToolResult:
    path = _resolve(ctx, args["file_path"])
    if not path.exists():
        # help the model recover from a near miss
        sibs = []
        if path.parent.is_dir():
            sibs = [
                p.name
                for p in path.parent.iterdir()
                if p.name.lower().startswith(path.name[:3].lower())
            ][:10]
        hint = f" Similar names here: {', '.join(sibs)}" if sibs else ""
        return ToolResult.error(f"{path} does not exist.{hint}")
    if path.is_dir():
        return ToolResult.error(f"{path} is a directory - use LS or Glob.")
    if path.suffix.lower() in IMAGE_EXT:
        return ToolResult.ok(
            f"[image file {path} - {path.stat().st_size} bytes; this agent cannot "
            "view images, inspect it with a shell tool instead]",
            summary=f"image {path.name}",
        )
    if _is_binary(path):
        return ToolResult.ok(
            f"[binary file {path} - {path.stat().st_size} bytes]",
            summary=f"binary {path.name}",
        )

    offset = int(args.get("offset") or 1)
    limit = int(args.get("limit") or MAX_READ_LINES)
    out: list[str] = []
    total = 0
    truncated_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            total = i
            if i < offset:
                continue
            if len(out) >= limit:
                continue
            line = line.rstrip("\n")
            if len(line) > MAX_LINE_LEN:
                line = line[:MAX_LINE_LEN] + " ... [line truncated]"
                truncated_lines += 1
            out.append(f"{i:6d}\t{line}")

    ctx.mark_read(path)
    if not out:
        return ToolResult.ok(
            f"[{path} has {total} lines; offset {offset} is past the end]",
            summary=f"{path.name}: empty slice",
        )
    body = "\n".join(out)
    tail = ""
    end = offset + len(out) - 1
    if end < total:
        tail = (
            f"\n\n[showing lines {offset}-{end} of {total}. "
            f"Use offset={end + 1} to continue.]"
        )
    return ToolResult.ok(body + tail, summary=f"{path.name} ({len(out)} lines)")


# -------------------------------------------------------------------- Write
@registry.add(
    "Write",
    "Create a file, or completely overwrite an existing one. Read an existing "
    "file before overwriting it. Prefer Edit for partial changes.",
    schema(
        ["file_path", "content"],
        file_path=string("Path to write."),
        content=string("Full file contents."),
    ),
)
def _write(args: dict, ctx) -> ToolResult:
    path = _resolve(ctx, args["file_path"])
    content = args.get("content", "")
    existed = path.exists()
    if existed and not ctx.was_read(path):
        return ToolResult.error(
            f"{path} already exists and has not been read in this session. "
            "Read it first so you do not discard content."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        ctx.backup(path)
    path.write_text(content, encoding="utf-8", newline="\n")
    ctx.mark_read(path)
    ctx.record_change(path, "overwrote" if existed else "created")
    n = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    verb = "Overwrote" if existed else "Created"
    return ToolResult.ok(f"{verb} {path} ({n} lines).", summary=f"{verb.lower()} {path.name}")


# --------------------------------------------------------------------- Edit
def _apply_edit(text: str, old: str, new: str, replace_all: bool) -> tuple[str, int]:
    count = text.count(old)
    if count == 0:
        return text, 0
    if replace_all:
        return text.replace(old, new), count
    if count > 1:
        return text, -count
    return text.replace(old, new, 1), 1


@registry.add(
    "Edit",
    "Replace an exact string in a file. old_string must appear exactly once "
    "unless replace_all is true. Include surrounding context to make it unique. "
    "Strip the line-number prefix that Read adds before matching.",
    schema(
        ["file_path", "old_string", "new_string"],
        file_path=string("File to edit."),
        old_string=string("Exact text to replace, including indentation."),
        new_string=string("Replacement text."),
        replace_all=boolean("Replace every occurrence (default false)."),
    ),
)
def _edit(args: dict, ctx) -> ToolResult:
    path = _resolve(ctx, args["file_path"])
    if not path.exists():
        return ToolResult.error(f"{path} does not exist.")
    if not ctx.was_read(path):
        return ToolResult.error(f"Read {path} before editing it.")
    old, new = args["old_string"], args.get("new_string", "")
    if old == new:
        return ToolResult.error("old_string and new_string are identical.")
    text = path.read_text(encoding="utf-8", errors="replace")
    result, n = _apply_edit(text, old, new, bool(args.get("replace_all")))
    if n == 0:
        return ToolResult.error(
            "old_string not found. Re-read the file; whitespace or indentation "
            "probably differs."
        )
    if n < 0:
        return ToolResult.error(
            f"old_string appears {abs(n)} times. Add surrounding context to make "
            "it unique, or pass replace_all=true."
        )
    ctx.backup(path)
    path.write_text(result, encoding="utf-8", newline="\n")
    ctx.record_change(path, "edited")
    return ToolResult.ok(
        f"Edited {path} ({n} replacement{'s' if n != 1 else ''}).\n"
        + _diff_preview(text, result, path),
        summary=f"edited {path.name}",
    )


@registry.add(
    "MultiEdit",
    "Apply several exact-string edits to one file atomically and in order. If "
    "any edit fails, nothing is written.",
    schema(
        ["file_path", "edits"],
        file_path=string("File to edit."),
        edits=array(
            "Edits applied in sequence.",
            {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["old_string", "new_string"],
            },
        ),
    ),
)
def _multiedit(args: dict, ctx) -> ToolResult:
    path = _resolve(ctx, args["file_path"])
    edits = args.get("edits") or []
    if not edits:
        return ToolResult.error("edits is empty.")
    if not path.exists():
        return ToolResult.error(f"{path} does not exist.")
    if not ctx.was_read(path):
        return ToolResult.error(f"Read {path} before editing it.")
    original = path.read_text(encoding="utf-8", errors="replace")
    text = original
    applied = 0
    for i, e in enumerate(edits, start=1):
        text, n = _apply_edit(
            text, e["old_string"], e.get("new_string", ""), bool(e.get("replace_all"))
        )
        if n == 0:
            return ToolResult.error(f"edit {i}: old_string not found. Nothing written.")
        if n < 0:
            return ToolResult.error(
                f"edit {i}: old_string appears {abs(n)} times. Nothing written."
            )
        applied += n
    ctx.backup(path)
    path.write_text(text, encoding="utf-8", newline="\n")
    ctx.record_change(path, "edited")
    return ToolResult.ok(
        f"Applied {len(edits)} edits ({applied} replacements) to {path}.\n"
        + _diff_preview(original, text, path),
        summary=f"edited {path.name} x{len(edits)}",
    )


def _diff_preview(before: str, after: str, path: Path, limit: int = 60) -> str:
    import difflib

    d = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{path.name}", tofile=f"b/{path.name}", lineterm="", n=2,
        )
    )
    if not d:
        return ""
    if len(d) > limit:
        d = d[:limit] + [f"... ({len(d) - limit} more diff lines)"]
    return "\n".join(d)


# --------------------------------------------------------------------- Glob
@registry.add(
    "Glob",
    "Find files by glob pattern (e.g. '**/*.py', 'src/**/test_*.go'), newest "
    "first. Fast; use it instead of `find`.",
    schema(
        ["pattern"],
        pattern=string("Glob pattern."),
        path=string("Directory to search (default: working directory)."),
        limit=integer("Max results (default 200)."),
    ),
    read_only=True,
)
def _glob(args: dict, ctx) -> ToolResult:
    root = _resolve(ctx, args.get("path") or ".")
    if not root.is_dir():
        return ToolResult.error(f"{root} is not a directory.")
    limit = int(args.get("limit") or 200)
    hits: list[Path] = []
    for p in root.glob(args["pattern"]):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.is_file():
            hits.append(p)
        if len(hits) > 5000:
            break
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shown = hits[:limit]
    if not shown:
        return ToolResult.ok("No files matched.", summary="0 matches")
    body = "\n".join(str(p) for p in shown)
    if len(hits) > limit:
        body += f"\n... ({len(hits) - limit} more)"
    return ToolResult.ok(body, summary=f"{len(hits)} match(es)")


# --------------------------------------------------------------------- Grep
@registry.add(
    "Grep",
    "Search file contents with a regular expression. Uses ripgrep when it is "
    "installed and falls back to a pure-python scan. Prefer this over "
    "`grep` in Bash.",
    schema(
        ["pattern"],
        pattern=string("Regular expression."),
        path=string("File or directory to search (default: working directory)."),
        glob=string("Only search files matching this glob, e.g. '*.py'."),
        output_mode=string(
            "'content' (matching lines), 'files_with_matches' (default), or 'count'.",
            enum=["content", "files_with_matches", "count"],
        ),
        case_insensitive=boolean("Case-insensitive match."),
        context=integer("Lines of context around each match (content mode)."),
        limit=integer("Max lines/files returned (default 120)."),
    ),
    read_only=True,
)
def _grep(args: dict, ctx) -> ToolResult:
    pattern = args["pattern"]
    target = _resolve(ctx, args.get("path") or ".")
    mode = args.get("output_mode") or "files_with_matches"
    limit = int(args.get("limit") or 120)
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--no-heading", "--color", "never"]
        if args.get("case_insensitive"):
            cmd.append("-i")
        if mode == "files_with_matches":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")
        else:
            cmd.append("-n")
            if args.get("context"):
                cmd += ["-C", str(int(args["context"]))]
        if args.get("glob"):
            cmd += ["--glob", args["glob"]]
        cmd += ["-e", pattern, str(target)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            lines = [l for l in proc.stdout.splitlines() if l]
            if proc.returncode not in (0, 1) and not lines:
                return ToolResult.error(proc.stderr.strip()[:800] or "ripgrep failed")
            return _grep_output(lines, limit)
        except subprocess.TimeoutExpired:
            return ToolResult.error("search timed out after 120s")

    # pure-python fallback
    try:
        rx = re.compile(pattern, re.IGNORECASE if args.get("case_insensitive") else 0)
    except re.error as exc:
        return ToolResult.error(f"bad regex: {exc}")
    files = [target] if target.is_file() else [
        p for p in target.rglob(args.get("glob") or "*")
        if p.is_file() and not any(x in IGNORE_DIRS for x in p.parts)
    ]
    lines: list[str] = []
    for f in files:
        if _is_binary(f):
            continue
        try:
            matched = 0
            for i, line in enumerate(
                f.open("r", encoding="utf-8", errors="replace"), start=1
            ):
                if rx.search(line):
                    matched += 1
                    if mode == "content":
                        lines.append(f"{f}:{i}:{line.rstrip()}")
                    elif mode == "files_with_matches":
                        lines.append(str(f))
                        break
            if mode == "count" and matched:
                lines.append(f"{f}:{matched}")
        except OSError:
            continue
        if len(lines) > limit * 4:
            break
    return _grep_output(lines, limit)


def _grep_output(lines: list[str], limit: int) -> ToolResult:
    if not lines:
        return ToolResult.ok("No matches found.", summary="0 matches")
    shown = lines[:limit]
    body = "\n".join(shown)
    if len(lines) > limit:
        body += f"\n... ({len(lines) - limit} more; narrow the pattern or raise limit)"
    return ToolResult.ok(body, summary=f"{len(lines)} result(s)")


# ----------------------------------------------------------------------- LS
@registry.add(
    "LS",
    "List a directory with sizes and types, optionally recursive.",
    schema(
        [],
        path=string("Directory (default: working directory)."),
        all=boolean("Include dotfiles."),
        depth=integer("Recursion depth, 1 = just this directory (default 1)."),
    ),
    read_only=True,
)
def _ls(args: dict, ctx) -> ToolResult:
    root = _resolve(ctx, args.get("path") or ".")
    if not root.exists():
        return ToolResult.error(f"{root} does not exist.")
    if root.is_file():
        st = root.stat()
        return ToolResult.ok(f"{root}  {st.st_size} bytes")
    depth = max(1, int(args.get("depth") or 1))
    show_all = bool(args.get("all"))
    out: list[str] = [str(root)]

    def walk(d: Path, level: int, prefix: str) -> None:
        try:
            entries = sorted(
                d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            out.append(prefix + "  [permission denied]")
            return
        for p in entries:
            if not show_all and p.name.startswith("."):
                continue
            if p.is_dir():
                skip = p.name in IGNORE_DIRS
                out.append(f"{prefix}  {p.name}/" + ("  [skipped]" if skip else ""))
                if level < depth and not skip:
                    walk(p, level + 1, prefix + "  ")
            else:
                try:
                    out.append(f"{prefix}  {p.name}  ({p.stat().st_size}b)")
                except OSError:
                    out.append(f"{prefix}  {p.name}")
            if len(out) > 1500:
                return

    walk(root, 1, "")
    return ToolResult.ok("\n".join(out[:1500]), summary=f"{len(out) - 1} entries")


__all__ = ["registry"]
