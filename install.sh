#!/bin/bash
# Karl installer — sets up the local coding agent on macOS.
#
# Idempotent: safe to re-run. It will
#   1. ensure Homebrew, Ollama, and a Python 3.11+ are present
#   2. create the .venv and install Python deps + Piper voice
#   3. pull the chat + embedding models (skip with --no-models)
#   4. link the `karl` command onto your PATH
#   5. connect Google (Gmail + Calendar) if credentials.json is present (skip with --no-google)
#
# Usage:
#   ./install.sh                # full install (prompts to connect Google at the end)
#   ./install.sh --no-models    # skip the (large) model downloads
#   ./install.sh --no-google    # don't prompt to connect Google

set -euo pipefail

# --- locate the repo (this script's directory) ------------------------------
KARL_HOME="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VENV="$KARL_HOME/.venv"
PULL_MODELS=1
SKIP_GOOGLE=0
for a in "$@"; do
  case "$a" in
    --no-models) PULL_MODELS=0 ;;
    --no-google) SKIP_GOOGLE=1 ;;
    *) ;;
  esac
done

say()  { printf "\033[1;36m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$1" >&2; }
die()  { printf "\033[1;31mxx\033[0m %s\n" "$1" >&2; exit 1; }

[ "$(uname)" = "Darwin" ] || die "This installer targets macOS. On Linux, install Ollama + Python 3.11 manually, then run bin/karl."

# --- 1. Homebrew ------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  die "Homebrew not found. Install it from https://brew.sh then re-run ./install.sh"
fi

# --- 2. Ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1 && [ ! -x /opt/homebrew/opt/ollama/bin/ollama ]; then
  say "Installing Ollama..."
  brew install ollama
else
  say "Ollama already installed."
fi
OLLAMA_BIN="$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)"

# --- 3. Python 3.11+ --------------------------------------------------------
find_python() {
  for c in python3.13 python3.12 python3.11; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  if command -v python3 >/dev/null 2>&1 && \
     python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)'; then
    echo python3; return 0
  fi
  return 1
}
if ! PYBIN="$(find_python)"; then
  say "Installing Python 3.12..."
  brew install python@3.12
  PYBIN="$(find_python)" || die "Python 3.11+ still not found after install."
fi
say "Using Python: $($PYBIN --version) ($PYBIN)"

# --- 4. venv + deps ---------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtualenv at .venv ..."
  "$PYBIN" -m venv "$VENV"
fi
say "Installing Python dependencies..."
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$KARL_HOME/requirements.txt"

# Piper neural TTS voice (for `karl --voice`) — download once if missing.
PIPER_VOICE="$KARL_HOME/voices/en_US-amy-medium.onnx"
if [ ! -f "$PIPER_VOICE" ]; then
  say "Downloading Piper neural voice (~63 MB)..."
  mkdir -p "$KARL_HOME/voices"
  PV="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
  curl -fsSL "$PV" -o "$PIPER_VOICE" && curl -fsSL "$PV.json" -o "$PIPER_VOICE.json" \
    || warn "Piper voice download failed — voice mode will fall back to macOS \`say\`."
else
  say "Piper voice already present."
fi

# --- 5. models --------------------------------------------------------------
# Read the model names straight from config so this stays in sync.
read -r CHAT_MODEL EMBED_MODEL < <(
  PYTHONPATH="$KARL_HOME/assistant" "$VENV/bin/python" -c \
    "import config; print(config.CHAT_MODEL, config.EMBED_MODEL)"
)
if [ "$PULL_MODELS" = "1" ]; then
  if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    say "Starting Ollama daemon..."
    "$OLLAMA_BIN" serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  for m in "$CHAT_MODEL" "$EMBED_MODEL"; do
    if "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -q "^${m%%:*}"; then
      say "Model '$m' already present."
    else
      say "Pulling model '$m' (this can take a while)..."
      "$OLLAMA_BIN" pull "$m"
    fi
  done
else
  warn "Skipping model downloads (--no-models). Pull later: ollama pull $CHAT_MODEL && ollama pull $EMBED_MODEL"
fi

# --- 6. link the `karl` command ---------------------------------------------
chmod +x "$KARL_HOME/bin/karl"
link_into() {  # $1 = target dir on PATH
  ln -sf "$KARL_HOME/bin/karl" "$1/karl" && say "Linked 'karl' -> $1/karl"
}
if [ -d /opt/homebrew/bin ] && [ -w /opt/homebrew/bin ]; then
  link_into /opt/homebrew/bin
elif [ -w /usr/local/bin ]; then
  link_into /usr/local/bin
else
  mkdir -p "$HOME/.local/bin"
  link_into "$HOME/.local/bin"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) warn "Add this to your ~/.zshrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
  esac
fi

# --- 7. Google (Gmail + Calendar) — optional --------------------------------
if [ "$SKIP_GOOGLE" = "0" ]; then
  CREDS="$KARL_HOME/credentials.json"
  AUTH="$KARL_HOME/assistant/tools/google_auth.py"
  if [ -f "$CREDS" ]; then
    # already connected? (any token present)
    if ls "$KARL_HOME"/token*.json >/dev/null 2>&1; then
      say "Google already connected."
    elif [ -t 0 ]; then
      printf "\033[1;36m==>\033[0m Connect a Google account now (Gmail + Calendar)? Opens a browser. [y/N] "
      read -r ans
      if [[ "$ans" =~ ^[Yy] ]]; then
        "$VENV/bin/python" "$AUTH" \
          || warn "Authorization didn't finish — run later: .venv/bin/python assistant/tools/google_auth.py"
      else
        say "Skipped. Connect later: .venv/bin/python assistant/tools/google_auth.py"
      fi
    else
      say "credentials.json found. Connect later: .venv/bin/python assistant/tools/google_auth.py"
    fi
  else
    warn "Google (Gmail/Calendar) is optional and not set up. To enable it:"
    echo "    1. Google Cloud console: create a project; enable the Gmail API + Calendar API."
    echo "    2. OAuth consent screen -> External -> publish to production."
    echo "    3. Create an OAuth client ID (Desktop app) -> download JSON -> save as"
    echo "       $CREDS"
    echo "    4. Re-run ./install.sh, or: .venv/bin/python assistant/tools/google_auth.py"
  fi
fi

echo
say "Done. Start Karl from any project directory:"
echo "    cd /path/to/your/project && karl"
