#!/bin/bash
set -e  # exit on first error

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version

# Make sure 'conda activate' works in non-interactive shell
eval "$(conda shell.bash hook)"

# Install MergeBench
conda create -n merging
conda activate merging
git clone https://github.com/tuanpasg/mergebench2
cd MergeBench
pip install -r requirements.txt

conda deactivate
# Install BigCode

conda create -n bigcode python=3.10.9
conda activate bigcode

git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

pip install -e .
pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.1

conda deactivate
# Install LmEval

conda create -n lmeval python=3.10.9
conda activate lmeval

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install langdetect
pip install immutabledict

conda deactivate