#!/usr/bin/env python3

import argparse
from huggingface_hub import HfApi, upload_folder

def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face Hub (e.g., username/my-model)"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Local folder containing model files"
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

    # Create repo if it doesn't already exist
    print(f"🔧 Creating repo {args.repo_id} (if not exists)...")
    api.create_repo(repo_id=args.repo_id, exist_ok=True)

    # Upload folder
    print(f"⬆️ Uploading folder '{args.folder_path}' to {args.repo_id} ...")
    upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print("✅ Upload completed successfully!")

if __name__ == "__main__":
    main()
