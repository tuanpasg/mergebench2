#!/bin/bash
set -euo pipefail  # safer

MODEL="$1"
GPU_ID="$2"
OUTPUT_PATH="$3"

echo "MODEL:        $MODEL"
echo "GPU_ID:       $GPU_ID"
echo "OUTPUT_PATH:  $OUTPUT_PATH"

# Normalize OUTPUT_PATH to absolute (optional but safer)
OUTPUT_PATH="$(readlink -f "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_PATH"

# Initialize conda for this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Use only the requested GPU and refer to it as cuda:0
export CUDA_VISIBLE_DEVICES="$GPU_ID"
DEVICE="cuda:0"

conda activate lmeval

# Math (GSM8K)
lm_eval --model hf \
  --model_args "pretrained=$MODEL" \
  --tasks gsm8k_cot \
  --device "$DEVICE" \
  --batch_size 16 \
  --output_path "$OUTPUT_PATH"

# Instruction (IFEval)
lm_eval --model hf \
  --model_args "pretrained=$MODEL" \
  --tasks ifeval \
  --device "$DEVICE" \
  --batch_size 8 \
  --output_path "$OUTPUT_PATH"

# Generalization Retention (MMLU subset)
lm_eval --model hf \
  --model_args "pretrained=$MODEL" \
  --tasks mmlu \
  --device "$DEVICE" \
  --batch_size 8 \
  --output_path "$OUTPUT_PATH" \
  --num_fewshot 5 \
  # --limit 100

conda deactivate

# BigCode eval
conda activate bigcode
cd bigcode-evaluation-harness

accelerate launch main.py \
  --model "$MODEL" \
  --max_length_generation 512 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --metric_output_path "$OUTPUT_PATH/code_eval.json" \
  --use_auth_token

cd ..
conda deactivate
