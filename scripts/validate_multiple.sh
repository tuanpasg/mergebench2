#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/validated_results}"

MODELS=(
  "/workspace/outs/wudi_cold_start_10"
  "/workspace/outs/wudi_cold_start_20"
  "/workspace/outs/wudi_cold_start_50"
  "/workspace/outs/wudi_warm_start_10"
  "/workspace/outs/wudi_warm_start_20"
  "/workspace/outs/wudi_warm_start_50"
)

for MODEL in "${MODELS[@]}"; do
  echo "Evaluating: $MODEL"
  "$SCRIPT_DIR/validate.sh" "$MODEL" "$GPU_ID" "$OUTPUT_PATH"
done
