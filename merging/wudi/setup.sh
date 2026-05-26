#!/bin/bash

# Initialize conda for this shell
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "/opt/miniforge3/bin/conda" ]; then
  CONDA_BIN="/opt/miniforge3/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
echo "[setup_eval] Sourcing conda from: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda --version

conda create -y -n wudi python=3.10.9
conda activate wudi 

cd /workspace/
git clone https://github.com/tuanpasg/wudi-merging-llama.git

cd /workspace/wudi-merging-llama/llama
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd /workspace/wudi-merging-llama/nlp_roberta/llama
python3 main.py \
 --out /workspace/outs/wudi \
 --merge_method wudi_merge \
 --wudi_variant wudi_last_14_layers\
 --scaling 1 \
 --wudi_device cuda \
 --exclude ".*embed.*" ".*lm_head.*"