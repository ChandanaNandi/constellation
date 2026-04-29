#!/bin/bash
# Download BDD100K dataset (5K subset) for Constellation
#
# BDD100K official source requires registration at: https://bdd-data.berkeley.edu/
# This script provides multiple download options.

set -e

DATA_DIR="${DATA_DIR:-./data}"
BDD100K_DIR="$DATA_DIR/bdd100k"
SUBSET_SIZE="${SUBSET_SIZE:-5000}"

echo "🌌 Constellation - BDD100K Dataset Setup"
echo "=========================================="
echo ""
echo "Target: $SUBSET_SIZE image subset"
echo "Location: $BDD100K_DIR"
echo ""

# Create directory structure
mkdir -p "$BDD100K_DIR"/{images/100k/{train,val},labels/det_20}

# Check if data already exists
if [ -d "$BDD100K_DIR/images/100k/train" ] && [ "$(ls -A $BDD100K_DIR/images/100k/train 2>/dev/null | head -1)" ]; then
    TRAIN_COUNT=$(ls -1 "$BDD100K_DIR/images/100k/train"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Found existing data: $TRAIN_COUNT training images"

    if [ "$TRAIN_COUNT" -ge "$SUBSET_SIZE" ]; then
        echo "Dataset ready. Skipping download."
        exit 0
    fi
fi

echo ""
echo "📥 Download Options"
echo "==================="
echo ""

# Option 1: Hugging Face mirror (easiest)
echo "Option 1: Hugging Face (Recommended)"
echo "-------------------------------------"
echo "The BDD100K dataset is available on Hugging Face:"
echo ""
echo "  pip install datasets"
echo "  python -c \"from datasets import load_dataset; ds = load_dataset('bdd100k', split='train[:$SUBSET_SIZE]')\""
echo ""

# Option 2: Official portal
echo "Option 2: Official BDD100K Portal"
echo "----------------------------------"
echo "1. Register at: https://bdd-data.berkeley.edu/"
echo "2. Download:"
echo "   - bdd100k_images_100k.zip (Images)"
echo "   - bdd100k_labels_release.zip (Labels)"
echo "3. Extract to: $BDD100K_DIR"
echo ""

# Option 3: Kaggle
echo "Option 3: Kaggle"
echo "-----------------"
echo "Search 'BDD100K' on Kaggle for community uploads."
echo ""

# Try automated download via Hugging Face
echo ""
read -p "Attempt automated download via Hugging Face? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Downloading via Hugging Face datasets..."
    echo ""

    # Create Python download script
    cat > "$DATA_DIR/download_hf.py" << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path

# Configuration
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", 5000))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
BDD100K_DIR = DATA_DIR / "bdd100k"

print(f"Downloading {SUBSET_SIZE} images from BDD100K...")

try:
    from datasets import load_dataset
    from PIL import Image
    import json

    # Load subset
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("bdd100k", "detection", split=f"train[:{SUBSET_SIZE}]", trust_remote_code=True)

    # Create directories
    images_dir = BDD100K_DIR / "images" / "100k" / "train"
    labels_dir = BDD100K_DIR / "labels" / "det_20"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Save images and collect labels
    all_labels = []

    for i, item in enumerate(ds):
        if i % 100 == 0:
            print(f"Processing {i}/{len(ds)}...")

        # Save image
        image = item["image"]
        image_name = f"{item['name']}"
        if not image_name.endswith('.jpg'):
            image_name = f"image_{i:06d}.jpg"

        image_path = images_dir / image_name
        image.save(image_path)

        # Collect labels
        if "labels" in item and item["labels"]:
            label_entry = {
                "name": image_name,
                "labels": item["labels"]
            }
            all_labels.append(label_entry)

    # Save labels
    labels_file = labels_dir / "det_train.json"
    with open(labels_file, "w") as f:
        json.dump(all_labels, f)

    print(f"✅ Downloaded {len(ds)} images to {images_dir}")
    print(f"✅ Saved labels to {labels_file}")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install datasets pillow")
    sys.exit(1)
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("")
    print("Please download manually from https://bdd-data.berkeley.edu/")
    sys.exit(1)
PYTHON_SCRIPT

    # Run download script
    SUBSET_SIZE=$SUBSET_SIZE DATA_DIR=$DATA_DIR python "$DATA_DIR/download_hf.py"

    # Cleanup
    rm -f "$DATA_DIR/download_hf.py"

else
    echo ""
    echo "Manual download required. Follow the instructions above."
    echo ""
    echo "After downloading, run this script again to verify the data."
fi

# Verify download
echo ""
echo "📊 Verifying dataset..."

if [ -d "$BDD100K_DIR/images/100k/train" ]; then
    TRAIN_COUNT=$(ls -1 "$BDD100K_DIR/images/100k/train"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
    echo "Training images: $TRAIN_COUNT"
else
    echo "Training images: 0 (directory not found)"
fi

if [ -f "$BDD100K_DIR/labels/det_20/det_train.json" ]; then
    echo "Detection labels: ✅ Found"
else
    echo "Detection labels: ❌ Not found"
fi

echo ""
echo "Setup complete. Run the data exploration notebook:"
echo "  jupyter notebook notebooks/01_data_exploration.ipynb"
