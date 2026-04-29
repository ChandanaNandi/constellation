#!/usr/bin/env python3
"""Download sample BDD100K images from Hugging Face."""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

DATA_DIR = Path("./data/bdd100k")
IMAGES_DIR = DATA_DIR / "images" / "100k" / "train"
LABELS_DIR = DATA_DIR / "labels" / "det_20"
SAMPLE_SIZE = 100

def main():
    print(f"Downloading {SAMPLE_SIZE} sample images...")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    repo_id = "karthik789338/BDD100K-Images"

    print("Fetching file list...")
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Filter for train images
    image_files = [f for f in files if f.endswith('.jpg') and 'train' in f][:SAMPLE_SIZE]
    print(f"Found {len(image_files)} images to download")

    labels = []
    downloaded = 0

    for i, file_path in enumerate(image_files):
        try:
            if i % 10 == 0:
                print(f"  {i}/{len(image_files)}...")

            local = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset")
            name = Path(file_path).name
            shutil.copy2(local, IMAGES_DIR / name)
            labels.append({"name": name, "labels": []})
            downloaded += 1
        except Exception as e:
            print(f"  Skip {file_path}: {e}")

    # Save labels
    with open(LABELS_DIR / "det_train.json", "w") as f:
        json.dump(labels, f)

    print(f"\n✅ Downloaded {downloaded} images to {IMAGES_DIR}")
    print(f"✅ Labels file: {LABELS_DIR}/det_train.json")

if __name__ == "__main__":
    main()
