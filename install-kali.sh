#!/usr/bin/env bash
# deepagent installer for Kali Linux / Debian-based systems.
# Installs deepagent into an isolated virtualenv and drops a launcher on PATH.
set -euo pipefail

say()  { printf '\033[1;36m[deepagent]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deepagent]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[deepagent]\033[0m %s\n' "$*" >&2; exit 1; }

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${DEEPAGENT_PREFIX:-$HOME/.local/share/deepagent}"
BIN_DIR="${DEEPAGENT_BIN:-$HOME/.local/bin}"
VENV="$PREFIX/venv"

[ -f "$SRC_DIR/pyproject.toml" ] || die "run this script from the deepagent source tree"

command -v python3 >/dev/null || die "python3 is required (apt install python3 python3-venv)"
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
say "python $PYV detected"

if ! python3 -c 'import venv' 2>/dev/null; then
  warn "python3-venv missing; attempting to install"
  sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip || \
    die "could not install python3-venv; run: sudo apt install python3-venv"
fi

say "creating virtualenv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel
say "installing deepagent and dependencies"
pip install --quiet "$SRC_DIR[all]"
deactivate

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/deepagent" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/deepagent" "\$@"
EOF
chmod +x "$BIN_DIR/deepagent"
ln -sf "$BIN_DIR/deepagent" "$BIN_DIR/dag"

say "installed launchers: $BIN_DIR/deepagent  (alias: dag)"

case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) warn "$BIN_DIR is not on PATH. Add this to your ~/.bashrc or ~/.zshrc:"
     printf '    export PATH="%s:$PATH"\n' "$BIN_DIR" ;;
esac

# starter config
CFG_DIR="${DEEPAGENT_HOME:-$HOME/.deepagent}"
if [ ! -f "$CFG_DIR/config.toml" ]; then
  mkdir -p "$CFG_DIR"
  cat > "$CFG_DIR/config.toml" <<'EOF'
# deepagent configuration
# Pick a default profile. Built-ins: nvidia, nvidia-local, ollama, llamacpp,
# vllm, lmstudio, openai, openrouter, groq, deepseek, anthropic.
profile = "nvidia"

[profiles.nvidia]
# NVIDIA NIM / build.nvidia.com. Export NVIDIA_API_KEY in your shell.
base_url = "https://integrate.api.nvidia.com/v1"
model = "qwen/qwen3-coder-480b-a35b-instruct"
api_key_env = "NVIDIA_API_KEY"

[profiles.ollama]
base_url = "http://localhost:11434"
model = "qwen2.5-coder:14b"

permission_mode = "ask"
EOF
  say "wrote starter config: $CFG_DIR/config.toml"
fi

echo
say "done. Next steps:"
cat <<EOF
  1. Set an API key for a hosted model, e.g.:
         export NVIDIA_API_KEY=nvapi-xxxxxxxx
     ...or run a local model instead:
         ollama serve &  &&  ollama pull qwen2.5-coder:14b
         deepagent --profile ollama
  2. Start it in your project:
         cd /path/to/project && deepagent
  3. One-shot / scripting:
         deepagent -p "explain what this repo does"
EOF
