#!/usr/bin/env python3

import argparse
from huggingface_hub import HfApi, upload_folder
import numpy as np
def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model",
        help="Commit message for this upload"
    )
    
    args = parser.parse_args()

    # Initialize API
    api = HfApi()

    repo_base="tuanpasg/mb_llama_iso_"
    folder_path_base="/workspace/mergebench2/merged_models/Llama-3.2-3B_merged/IsoCTS_scaling_coef_"
    # Create repo if it doesn't already exist
    for alpha in np.arange(2.6,3.4,0.2):
        repo_id = repo_base + f"{alpha:.1f}"
        folder_path = folder_path_base + f"{alpha:.1f}"
        print(f"🔧 Creating repo {repo_id} (if not exists)...")
        api.create_repo(repo_id=repo_id, exist_ok=True)

        # Upload folder
        print(f"⬆️ Uploading folder '{folder_path}' to {repo_id} ...")
        upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

        print("✅ Upload completed successfully!")

if __name__ == "__main__":
    main()