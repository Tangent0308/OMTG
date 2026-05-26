#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$RL_ROOT/.." && pwd)"
TRANSFORMERS_REPO_URL="https://github.com/huggingface/transformers.git"
TRANSFORMERS_COMMIT="f2738ee3756483538c20e10e4f104324675fb406"
TRANSFORMERS_SRC_DIR="$PROJECT_ROOT/.cache/transformers-src"

cd "$SCRIPT_DIR"
export UV_LINK_MODE=copy

uv venv "$RL_ROOT/.venv" --python 3.12 --allow-existing
source "$RL_ROOT/.venv/bin/activate"

PYTHONPATH="" uv sync --project "$RL_ROOT" --active --no-install-project
uv pip install -r "$RL_ROOT/verl/requirements.txt"

mkdir -p "$(dirname "$TRANSFORMERS_SRC_DIR")"
if [ ! -d "$TRANSFORMERS_SRC_DIR/.git" ]; then
    git clone "$TRANSFORMERS_REPO_URL" "$TRANSFORMERS_SRC_DIR"
fi

git -C "$TRANSFORMERS_SRC_DIR" fetch --all --tags
git -C "$TRANSFORMERS_SRC_DIR" checkout "$TRANSFORMERS_COMMIT"

pushd "$TRANSFORMERS_SRC_DIR" >/dev/null
uv pip install '.[torch]'
popd >/dev/null

uv pip install -e "$RL_ROOT/verl"

echo "Install complete. Virtual environment: ${RL_ROOT}/.venv"
