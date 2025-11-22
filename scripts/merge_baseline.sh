# First and foremost login to HF
hf auth login

# Model soup
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.33 --base-model meta-llama/Llama-3.2-3B

# Task arithmetic
python ./merging/main.py --algo TaskArithmetic --scaling-coef 0.33 --base-model meta-llama/Llama-3.2-3B

# Dataless Localize-and-Stitch
python ./merging/main.py --algo LocalizeAndStitch --base-model meta-llama/Llama-3.2-3B --sparsity 0.1 --dataless
