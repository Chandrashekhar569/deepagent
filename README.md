# deepagent

A deep, terminal-native coding agent — the same shape as Claude Code / Codex,
but **provider-agnostic** and built to run on **Kali Linux** with full access to
the shell and every tool on the box.

It drives any of:

- **NVIDIA NIM** (`build.nvidia.com` or a self-hosted NIM container)
- Any **OpenAI-compatible** endpoint (vLLM, llama.cpp `llama-server`, LM Studio,
  OpenRouter, Groq, DeepSeek, OpenAI itself)
- **Ollama** (native `/api/chat`, local models)
- **Anthropic** Messages API

Switch between a cloud API key and a fully local model with one flag — the agent
loop, tools, permissions and UI are identical.

## What it can do

- **Agentic tool loop** — the model reads, searches, writes, edits and runs code,
  then verifies its own work, iterating until the task is done.
- **Full filesystem tools** — `Read` (line-numbered), `Write`, `Edit`,
  `MultiEdit` (atomic multi-edit with diff preview), `Glob`, `Grep`
  (ripgrep-backed), `LS`.
- **Persistent shell** — `Bash` runs in one long-lived shell, so `cd`, `export`,
  virtualenvs and shell state carry across calls, just like a real terminal.
  Background jobs via `Bash(background=true)` + `BashOutput` / `KillBash` for
  servers and long-running processes.
- **Web** — `WebSearch` (DuckDuckGo) and `WebFetch` (page → Markdown).
- **Subagents** — `Task` spawns a fresh context to handle wide searches or
  independent subtasks without polluting the main context.
- **Planning** — `TodoWrite` task list, `Think` scratchpad.
- **MCP** — pull in external stdio MCP tool servers from config.
- **Permission engine** — every action is classified allow / ask / deny.
  Modes: `ask`, `acceptEdits`, `plan`, `yolo`. Destructive commands always ask.
- **Context management** — automatic summarization/compaction as the window fills.
- **Sessions** — save/resume full conversations; one-shot (`-p`) and pipe modes
  for scripting.
- **Kali aware** — detects the distro and the security toolset on PATH and makes
  it available through the shell for authorized, defensive and CTF work.

## Install on Kali

```bash
git clone <this-repo> deepagent && cd deepagent
./install-kali.sh
```

This creates an isolated virtualenv and puts `deepagent` (alias `dag`) on your
PATH. Then pick a model:

```bash
# hosted (NVIDIA NIM)
export NVIDIA_API_KEY=nvapi-xxxxxxxx
deepagent

# or fully local, no API key
ollama serve & ; ollama pull qwen2.5-coder:14b
deepagent --profile ollama
```

### Manual install (any OS)

```bash
pip install -e ".[all]"
deepagent --setup        # writes ~/.deepagent/config.toml
```

## Usage

```bash
deepagent                        # interactive REPL in the current project
deepagent -C /path/to/project    # set working directory
deepagent -p "add a --json flag to cli.py and run the tests"   # one-shot
cat error.log | deepagent -p "diagnose this"                   # piped input
deepagent --profile ollama --model qwen2.5-coder:32b
deepagent --mode acceptEdits     # auto-approve in-workdir edits
deepagent --list-models          # ask the provider what it serves
```

### Slash commands (in the REPL)

```
/help  /model  /models  /profile  /mode  /tools  /env  /usage
/save  /resume  /clear  /reasoning  /config  /exit
```

## Configuration

`~/.deepagent/config.toml` (user) and `./.deepagent.toml` (per project, wins).
Everything is also overridable by env vars (`DEEPAGENT_MODEL`,
`DEEPAGENT_BASE_URL`, `DEEPAGENT_PROFILE`, …) and CLI flags.

```toml
profile = "nvidia"

[profiles.nvidia]
base_url = "https://integrate.api.nvidia.com/v1"
model = "qwen/qwen3-coder-480b-a35b-instruct"
api_key_env = "NVIDIA_API_KEY"

[profiles.my-nim]            # self-hosted NIM container
provider = "openai"
base_url = "http://localhost:8000/v1"
model = "meta/llama-3.3-70b-instruct"

permission_mode = "ask"

# optional MCP stdio servers -> tools named mcp__<name>__<tool>
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/kali"]
```

Built-in profiles: `nvidia`, `nvidia-local`, `ollama`, `llamacpp`, `vllm`,
`lmstudio`, `openai`, `openrouter`, `groq`, `deepseek`, `anthropic`.

### Project / agent instructions

deepagent reads `DEEPAGENT.md`, `AGENTS.md` or `CLAUDE.md` from the working
directory and folds them into the system prompt — so existing repo conventions
are respected.

## Permissions & safety

- Read-only tools (Read/Glob/Grep/LS/WebSearch/WebFetch) run without prompting.
- Edits inside the working directory prompt once (or auto-approve in
  `acceptEdits`); edits outside it always prompt.
- Read-only shell commands auto-run; anything that writes, chains, redirects,
  uses `sudo`, or scans remote hosts prompts for confirmation.
- A hard-coded list of destructive patterns (`rm -rf /`, `mkfs`, `dd of=/dev/…`,
  fork bombs, disk wipes) always prompts, even in `yolo`.
- `plan` mode makes no changes at all until you approve.
- Allow/deny rules: `--allow 'Bash(git status:*)' --deny 'Bash(rm:*)'`.

Use it for authorized security testing, CTF, and defensive work. It is a
developer tool, not a weapon — it won't help with clearly malicious use.

## Requirements

Python ≥ 3.10. Core deps: `httpx`, `rich`, `prompt_toolkit`, `pyyaml`. Optional
`[all]` adds `beautifulsoup4` + `markdownify` (better WebFetch) and `tiktoken`.
`ripgrep` is used for `Grep` when present (there's a pure-python fallback).

## License

MIT
