"""Configuration: layered defaults -> config file -> env vars -> CLI flags."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CONFIG_DIR = Path(os.environ.get("DEEPAGENT_HOME", Path.home() / ".deepagent"))
CONFIG_FILE = CONFIG_DIR / "config.toml"
SESSIONS_DIR = CONFIG_DIR / "sessions"
MEMORY_DIR = CONFIG_DIR / "memory"
LOG_DIR = CONFIG_DIR / "logs"

# Built-in profiles. Each maps to a provider implementation plus sane defaults.
PRESETS: dict[str, dict[str, Any]] = {
    "nvidia": {
        "provider": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "max_tokens": 8192,
    },
    "nvidia-local": {  # NIM container running on the box / in the lab
        "provider": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model": "meta/llama-3.3-70b-instruct",
        "max_tokens": 8192,
    },
    "ollama": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "api_key_env": None,
        "model": "qwen2.5-coder:14b",
        "max_tokens": 8192,
    },
    "llamacpp": {  # llama.cpp / llama-server --api
        "provider": "openai",
        "base_url": "http://localhost:8080/v1",
        "api_key_env": None,
        "model": "local-model",
        "max_tokens": 8192,
    },
    "vllm": {
        "provider": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "max_tokens": 8192,
    },
    "lmstudio": {
        "provider": "openai",
        "base_url": "http://localhost:1234/v1",
        "api_key_env": None,
        "model": "local-model",
        "max_tokens": 8192,
    },
    "openai": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1",
        "max_tokens": 8192,
    },
    "openrouter": {
        "provider": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3-coder",
        "max_tokens": 8192,
    },
    "groq": {
        "provider": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "moonshotai/kimi-k2-instruct",
        "max_tokens": 8192,
    },
    "deepseek": {
        "provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "max_tokens": 8192,
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5",
        "max_tokens": 8192,
    },
}


@dataclass
class Config:
    # model / provider
    profile: str = "nvidia"
    provider: str = "openai"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "qwen/qwen3-coder-480b-a35b-instruct"
    api_key: str | None = None
    api_key_env: str | None = "NVIDIA_API_KEY"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    context_window: int = 128_000
    stream: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # a cheaper/faster model used for subagents and compaction
    small_model: str | None = None

    # behaviour
    workdir: str = "."
    permission_mode: str = "ask"  # ask | acceptEdits | plan | yolo
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    max_turns: int = 200
    max_tool_output: int = 60_000
    bash_timeout: int = 180
    auto_compact_at: float = 0.80  # fraction of context window
    system_prompt_append: str = ""
    enable_web: bool = True
    search_engine: str = "duckduckgo"
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    theme: str = "dark"

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, overrides: dict[str, Any] | None = None) -> "Config":
        data: dict[str, Any] = {}
        file_data: dict[str, Any] = {}

        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("rb") as fh:
                file_data = tomllib.load(fh)

        # project-level config overrides user-level
        proj = Path(".deepagent.toml")
        if proj.exists():
            with proj.open("rb") as fh:
                _deep_merge(file_data, tomllib.load(fh))

        profile = (
            (overrides or {}).get("profile")
            or os.environ.get("DEEPAGENT_PROFILE")
            or file_data.get("profile")
            or cls.profile
        )
        preset = PRESETS.get(profile, {})
        data.update(preset)
        data["profile"] = profile

        # [profiles.<name>] blocks in the config file extend/override presets
        prof_block = (file_data.get("profiles") or {}).get(profile)
        if isinstance(prof_block, dict):
            data.update(prof_block)

        for key, value in file_data.items():
            if key in ("profiles", "mcp"):
                continue
            data[key] = value
        if "mcp" in file_data:
            data["mcp_servers"] = file_data["mcp"]

        # environment
        env_map = {
            "DEEPAGENT_MODEL": "model",
            "DEEPAGENT_BASE_URL": "base_url",
            "DEEPAGENT_PROVIDER": "provider",
            "DEEPAGENT_API_KEY": "api_key",
            "DEEPAGENT_SMALL_MODEL": "small_model",
            "DEEPAGENT_PERMISSION_MODE": "permission_mode",
            "DEEPAGENT_CONTEXT_WINDOW": "context_window",
            "DEEPAGENT_MAX_TOKENS": "max_tokens",
            "DEEPAGENT_TEMPERATURE": "temperature",
        }
        for env, key in env_map.items():
            if os.environ.get(env):
                data[key] = os.environ[env]

        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})

        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in known})

        # coerce types that may have arrived as strings (env/toml)
        cfg.max_tokens = int(cfg.max_tokens)
        cfg.context_window = int(cfg.context_window)
        cfg.temperature = float(cfg.temperature)
        cfg.bash_timeout = int(cfg.bash_timeout)

        if not cfg.api_key and cfg.api_key_env:
            cfg.api_key = os.environ.get(cfg.api_key_env)
        return cfg

    def save(self) -> Path:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        lines = ["# deepagent config", f'profile = "{self.profile}"', ""]
        lines.append(f'[profiles.{self.profile}]')
        for key in ("provider", "base_url", "model", "api_key_env", "small_model"):
            val = getattr(self, key)
            if val:
                lines.append(f'{key} = "{val}"')
        lines += [
            f"max_tokens = {self.max_tokens}",
            f"context_window = {self.context_window}",
            "",
            f'permission_mode = "{self.permission_mode}"',
        ]
        CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return CONFIG_FILE

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("api_key"):
            d["api_key"] = d["api_key"][:6] + "..." + d["api_key"][-4:]
        return d


def _deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, SESSIONS_DIR, MEMORY_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def detect_environment() -> dict[str, Any]:
    """Facts about the host, injected into the system prompt."""
    import platform

    info: dict[str, Any] = {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "shell": os.environ.get("SHELL", "/bin/bash"),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "is_root": hasattr(os, "geteuid") and os.geteuid() == 0,  # type: ignore[attr-defined]
    }
    osr = Path("/etc/os-release")
    if osr.exists():
        try:
            kv = dict(
                line.split("=", 1)
                for line in osr.read_text().splitlines()
                if "=" in line
            )
            info["distro"] = kv.get("PRETTY_NAME", "").strip('"')
            info["is_kali"] = "kali" in kv.get("ID", "").lower()
        except Exception:
            pass
    info["git"] = bool(shutil.which("git"))
    if info["git"] and (Path(".git").exists()):
        info["in_git_repo"] = True
    # which notable toolchains / kali tooling are actually installed
    probes = [
        "nmap", "masscan", "nikto", "sqlmap", "metasploit-framework", "msfconsole",
        "hydra", "john", "hashcat", "aircrack-ng", "gobuster", "ffuf", "wpscan",
        "burpsuite", "tcpdump", "wireshark", "tshark", "netcat", "nc", "socat",
        "docker", "nvidia-smi", "ollama", "node", "npm", "go", "rustc", "cargo",
        "gcc", "make", "ripgrep", "rg", "fd", "jq", "curl", "wget", "pip3",
    ]
    info["tools"] = sorted({p for p in probes if shutil.which(p)})
    return info


def dump_env_json() -> str:
    return json.dumps(detect_environment(), indent=2)
