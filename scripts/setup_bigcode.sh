#!/bin/bash
set -e  # exit on first error

# Make sure 'conda activate' works in non-interactive shell
eval "$(conda shell.bash hook)"

# Create env (add -y so it doesn't wait for confirmation)
conda create -n bigcode python=3.10.9 -y

# Activate the env
conda activate bigcode

# Sanity check: show which python we're using
python --version
which python
which pip

# Clone repo
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

# Install harness
pip install -e .

# Install PyTorch: see note below about versions
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install numpy
pip install numpy==1.24.1
