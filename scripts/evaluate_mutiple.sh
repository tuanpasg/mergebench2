#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/tested_results}"

MODELS=(
 "tuanpasg/mb_llama_ta_0.33"
 "tuanpasg/mb_llama_carbs_64_256_1.0"
  "MergeBench/Llama-3.2-3B_instruction"
  "MergeBench/Llama-3.2-3B_math"
  "MergeBench/Llama-3.2-3B_coding"
  "meta-llama/Llama-3.2-3B"
)

for MODEL in "${MODELS[@]}"; do
  echo "Evaluating: $MODEL"
  "$SCRIPT_DIR/evaluate_3t.sh" "$MODEL" "$GPU_ID" "$OUTPUT_PATH"
done
