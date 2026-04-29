#!/bin/bash
# Download BDD100K dataset subset for Constellation
#
# BDD100K requires registration at: https://bdd-data.berkeley.edu/
# After registration, download the following:
#   - 100K Images (train/val/test)
#   - Detection 2020 Labels
#   - Lane Marking Labels
#   - Drivable Area Labels
#
# This script provides instructions and creates the expected directory structure.

set -e

DATA_DIR="${DATA_DIR:-./data}"
BDD100K_DIR="$DATA_DIR/bdd100k"

echo "🌌 Constellation - BDD100K Dataset Setup"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p "$BDD100K_DIR"/{images,labels}/{train,val,test}
mkdir -p "$BDD100K_DIR"/labels/{det_20,lane,drivable}

echo "📁 Created directory structure at: $BDD100K_DIR"
echo ""
echo "⚠️  BDD100K requires manual download due to license restrictions."
echo ""
echo "Steps to download:"
echo "1. Register at: https://bdd-data.berkeley.edu/"
echo "2. Download the following files:"
echo "   - bdd100k_images_100k.zip (100K Images)"
echo "   - bdd100k_labels_images_det_20.zip (Detection Labels)"
echo "   - bdd100k_labels_images_lane.zip (Lane Labels)"
echo "   - bdd100k_labels_images_drivable.zip (Drivable Area Labels)"
echo ""
echo "3. Extract to the following locations:"
echo "   - Images → $BDD100K_DIR/images/"
echo "   - Detection labels → $BDD100K_DIR/labels/det_20/"
echo "   - Lane labels → $BDD100K_DIR/labels/lane/"
echo "   - Drivable labels → $BDD100K_DIR/labels/drivable/"
echo ""
echo "4. For development, you can start with just the validation set (~10K images)"
echo ""

# Check if any images exist
if [ -d "$BDD100K_DIR/images/train" ] && [ "$(ls -A $BDD100K_DIR/images/train 2>/dev/null)" ]; then
    TRAIN_COUNT=$(ls -1 "$BDD100K_DIR/images/train" 2>/dev/null | wc -l)
    VAL_COUNT=$(ls -1 "$BDD100K_DIR/images/val" 2>/dev/null | wc -l)
    echo "✅ Found existing data:"
    echo "   - Train images: $TRAIN_COUNT"
    echo "   - Val images: $VAL_COUNT"
else
    echo "📥 No images found yet. Follow the steps above to download."
fi

echo ""
echo "After downloading, run the data exploration notebook:"
echo "  jupyter notebook notebooks/01_data_exploration.ipynb"
