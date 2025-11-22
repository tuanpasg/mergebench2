#!/bin/bash
set -e  # exit on first error

# Make sure 'conda activate' works in non-interactive shell
eval "$(conda shell.bash hook)"

conda create -n lmeval python=3.10.9
conda activate lmeval

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install langdetect
pip install immutabledict