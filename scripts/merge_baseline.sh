# First and foremost login to HF
hf auth login

# Script to push merged model to HF, MUST modifying --repo_id and --folder_path 
python ./scripts/hf_publish.py --repo_id tuanpasg/mb_llama_ta_0.1 --folder_path /workspace/mergebench2/merged_models/Llama-3.2-3B_merged/TaskArithmetic_scaling_coef_0.1

# Model soup
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.33 --base-model meta-llama/Llama-3.2-3B

# Task arithmetic
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.33 --base-model meta-llama/Llama-3.2-3B

# Dataless Localize-and-Stitch
python ./merging/main.py --algo LocalizeAndStitch --base-model meta-llama/Llama-3.2-3B --sparsity 0.1 --dataless

# LineS
python ./merging/main.py --algo LiNeS --beta-coef 0.5 --base-model meta-llama/Llama-3.2-3B