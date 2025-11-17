#!/usr/bin/env python3
"""
Utility script to push the SER checkpoint and accompanying artifacts to Hugging Face Hub.

Usage:
    HF_TOKEN=hf_xxx python upload_to_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    repo_id = "Kananmittal/unitree_audio_to_emotion"
    folder_path = Path(__file__).parent / "hf_hub_upload"

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder to upload not found: {folder_path}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Create a write token at https://huggingface.co/settings/tokens "
            "and run `export HF_TOKEN=hf_xxx` before executing this script."
        )

    api = HfApi(token=token)

    # Make sure the repository exists (no-op if already created).
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message="Upload SER checkpoint and config",
        allow_patterns=["*"],
    )

    print(f"✅ Successfully uploaded {folder_path} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

