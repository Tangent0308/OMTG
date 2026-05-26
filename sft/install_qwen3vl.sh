#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"
export UV_LINK_MODE=copy

uv venv .venv --python 3.12 --allow-existing
source .venv/bin/activate

uv sync --active

echo "Install complete. Virtual environment: ${SCRIPT_DIR}/.venv"
