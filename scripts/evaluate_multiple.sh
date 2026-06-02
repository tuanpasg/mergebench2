#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/tested_results}"

MODELS=(
"tuanpasg/mb_gemma_wudi_iter_0"
"tuanpasg/mb_gemma_wudi_iter_10"
"tuanpasg/mb_gemma_wudi_iter_25"
"tuanpasg/mb_gemma_wudi_iter_100"
"tuanpasg/mb_gemma_wudi_iter_300"
"google/gemma-2-2b"
"MergeBench/gemma-2-2b_instruction"
"MergeBench/gemma-2-2b_math"
"MergeBench/gemma-2-2b_coding"
)

for MODEL in "${MODELS[@]}"; do
  echo "Evaluating: $MODEL"
  "$SCRIPT_DIR/evaluate_3t.sh" "$MODEL" "$GPU_ID" "$OUTPUT_PATH"
done
