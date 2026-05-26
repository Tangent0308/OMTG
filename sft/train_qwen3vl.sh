#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

cd "$SCRIPT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi

CONFIG_INPUT="${1:-qwen3vl_4b_omtg_wcot.yaml}"
if [[ "$CONFIG_INPUT" == *.yaml ]]; then
  CONFIG_REL="$CONFIG_INPUT"
else
  CONFIG_REL="${CONFIG_INPUT}.yaml"
fi

if [[ -f "$CONFIG_REL" ]]; then
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_REL"
else
  CONFIG_PATH="$PROJECT_ROOT/$CONFIG_REL"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-23333}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${ARNOLD_ID:-0}"
export NNODES="${ARNOLD_WORKER_NUM:-1}"
export NPROC_PER_NODE="${ARNOLD_WORKER_GPU:-8}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export VIDEO_MIN_TOKEN_NUM="${VIDEO_MIN_TOKEN_NUM:-2}"
export MODEL_SEQ_LEN="${MODEL_SEQ_LEN:-8192}"
export WANDB_PROJECT="${WANDB_PROJECT:-sft_qwen3vl_temporal_grounding}"
export ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-$PROJECT_ROOT/data}"

if [[ -n "${OMTG_DATA_SOURCE:-}" && ! -e "$PROJECT_ROOT/data" ]]; then
  ln -sfn "$OMTG_DATA_SOURCE" "$PROJECT_ROOT/data"
fi

if [[ ! -d "$ROOT_IMAGE_DIR" ]]; then
  echo "Error: ROOT_IMAGE_DIR $ROOT_IMAGE_DIR does not exist." >&2
  exit 1
fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Working directory: $PROJECT_ROOT"
echo "Using config: $CONFIG_PATH"

cd "$PROJECT_ROOT"
python -m swift.cli.main sft "$CONFIG_PATH"
