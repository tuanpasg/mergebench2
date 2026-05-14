#!/bin/bash
set -e  # exit on first error

# Conda note:
# In non-interactive bash scripts, `conda` commands may fail until conda.sh is sourced.

CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "/opt/miniforge3/bin/conda" ]; then
  CONDA_BIN="/opt/miniforge3/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
fi

if [ -n "$CONDA_BIN" ]; then
  echo "[setup_eval] Conda detected at: $CONDA_BIN"
else
  echo "[setup_eval] Conda not found. Installing Miniconda3..."

  MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
  MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$MINICONDA_SCRIPT" "$MINICONDA_URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_SCRIPT"
  else
    echo "[setup_eval] Error: neither wget nor curl is available to download Miniconda."
    exit 1
  fi

  bash "$MINICONDA_SCRIPT" -b -p "$HOME/miniconda3"
  rm -f "$MINICONDA_SCRIPT"
  CONDA_BIN="$HOME/miniconda3/bin/conda"
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
echo "[setup_eval] Sourcing conda from: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda --version

# Make sure 'conda activate' works in non-interactive shell
eval "$(conda shell.bash hook)"

# Install MergeBench
conda create -y -n merging python=3.10.9
conda activate merging
cd /workspace/mergebench2/merging
pip install -r requirements.txt
conda deactivate

# Install BigCode
conda create -y -n bigcode python=3.10.9
conda activate bigcode

cd /workspace
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install hf_transfer
conda deactivate
# Install LmEval

conda create -y -n lmeval python=3.10.9
conda activate lmeval

cd /workspace
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness

cd lm-evaluation-harness
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install transformers accelerate
pip install langdetect immutabledict hf_transfer

conda deactivate
