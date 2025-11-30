#!/bin/bash
set -e  # exit on first error

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Install MergeBench
conda create -n merging
conda activate merging
cd /workspace
git clone https://github.com/tuanpasg/mergebench2
cd mergebench2
pip install -r requirements.txt

conda deactivate