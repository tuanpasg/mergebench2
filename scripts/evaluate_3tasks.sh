#!/bin/bash
set -e  # exit on first error

MODEL=$1
GPU_ID=$2
OUTPUT_PATH=$3

echo $MODEL
echo $GPU_ID
echo $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=$GPU_ID

source $(conda info --base)/etc/profile.d/conda.sh
mkdir -p $OUTPUT_PATH

conda activate lmeval

# Math
lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks gsm8k_cot \
    --device cuda:$GPU_ID \
    --batch_size 16 \
    --output_path $OUTPUT_PATH

# Instruction
lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks ifeval \
    --device cuda:$GPU_ID \
    --batch_size 8 \
    --output_path $OUTPUT_PATH

# Generalization Retention
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct \
  --tasks mmlu_high_school_math,mmlu_physics,mmlu_computer_science,\
  mmlu_economics,mmlu_psychology,mmlu_political_science,\
  mmlu_history,mmlu_philosophy,mmlu_law,mmlu_linguistics \
  --device cuda:$GPU_ID \
  --batch_size 8 \
  --output_path $OUTPUT_PATH\
  --num_fewshot 5 \
  --limit 100

conda deactivate
conda activate bigcode
cd bigcode-evaluation-harness

#Coding
accelerate launch  main.py \
  --model $MODEL \
  --max_length_generation 512 \
  --precision bf16 \
  --tasks humanevalplus,mbppplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --metric_output_path $OUTPUT_PATH/code_eval.json \
  --use_auth_token

cd ..
conda deactivate
