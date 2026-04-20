#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Fixed paths and script
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/infer-extract-judge.py"

# ------------------------
# Fixed parameters
# ------------------------
## FULL DATASET
INPUT_FILE="../MaskProof-Bench.jsonl"
WORK_DIR_BASE="./outputs"

# ------------------------
# Inference model list (supports sweeps)
# ------------------------
INF_MODELS=(
  # Input model names here like "Qwen/Qwen3-8B"
)

# ------------------------
# Optional extraction / evaluation models
# Use an empty string to skip the argument
# ------------------------
EXT_MODEL="gpt-oss-120b"
EVAL_MODEL="gpt-oss-120b"

# ------------------------
# API configuration
# Set `API_KEY` or all of `INF_API_KEY`, `EXT_API_KEY`, `EVAL_API_KEY`
# before running this script.
# ------------------------
COMMON_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
INF_API_KEY="${INF_API_KEY:-$COMMON_API_KEY}"
EXT_API_KEY="${EXT_API_KEY:-$COMMON_API_KEY}"
EVAL_API_KEY="${EVAL_API_KEY:-$COMMON_API_KEY}"
COMMON_BASE_URL="${BASE_URL:-https://aihubmix.com/v1}"
INF_BASE_URL="${INF_BASE_URL:-$COMMON_BASE_URL}"
EXT_BASE_URL="${EXT_BASE_URL:-$COMMON_BASE_URL}"
EVAL_BASE_URL="${EVAL_BASE_URL:-$COMMON_BASE_URL}"
INF_REASONING_EFFORT="${INF_REASONING_EFFORT:-high}"

: "${INF_API_KEY:?Set API_KEY or INF_API_KEY before running this script.}"
: "${EXT_API_KEY:?Set API_KEY or EXT_API_KEY before running this script.}"
: "${EVAL_API_KEY:?Set API_KEY or EVAL_API_KEY before running this script.}"

# ------------------------
# Main loop
# ------------------------

for INF_MODEL in "${INF_MODELS[@]}"; do
  SAFE_MODEL_NAME="$(echo "$INF_MODEL" | tr '/:' '__')"
  WORK_DIR="${WORK_DIR_BASE}_${SAFE_MODEL_NAME}"

  CMD=(
    python3 "$PY_SCRIPT"
    --input-file "$INPUT_FILE"
    --work-dir "$WORK_DIR"
    --api-provider "openai"
    --inf-base-url "$INF_BASE_URL"
    --ext-base-url "$EXT_BASE_URL"
    --eval-base-url "$EVAL_BASE_URL"
    --inf-api-key "$INF_API_KEY"
    --ext-api-key "$EXT_API_KEY"
    --eval-api-key "$EVAL_API_KEY"
    --inf-model-path "$INF_MODEL"
    --inf-reasoning-effort "$INF_REASONING_EFFORT"
    --global-concurrency 32
    --inference-workers 32
    --extraction-workers 64
    --evaluation-workers 64
    --n-responses 4
    --resume
    --rerun-rate-limit
    # --rerun-http-status 403
  )

  if [ -n "$EXT_MODEL" ]; then
    CMD+=(--ext-model-path "$EXT_MODEL")
  fi

  if [ -n "$EVAL_MODEL" ]; then
    CMD+=(--eval-model-path "$EVAL_MODEL")
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
done
