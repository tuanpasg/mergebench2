Checkpoint of Roberta: https://drive.google.com/drive/folders/1pArQjN8f0DIdLcmv-YDoIYnKeNG0HJV2

Procedure to merging Llama-3.2-3B checkpoints by WUDI-merging
Step 0: clone

git clone https://github.com/tuanpasg/wudi-merging-llama.git

Step 1: Install library
pip install -r requirements_llama.txt

Step 2: Create environmental variable for HF_TOKEN

Step 3: Run merging script
python merge_llama.py \
 --out /workspace/outs/wudi \
 --merge_method wudi_merge \
 --scaling 0.3333333333 \
 --wudi_device cuda \
 --exclude ".*embed.*" ".*lm_head.*"
".*embed.*" ".*lm_head.*"
Step 4: Upload model to HF (hf_publish.sh)
