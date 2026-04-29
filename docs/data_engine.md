# Data Engine

## Overview

The data engine handles data ingestion, auto-labeling, and the closed-loop feedback system.

## Components

### Auto-Labeler
- Uses SAM (Segment Anything Model) for segmentation masks
- Uses YOLOv8 for bounding box detection
- Outputs labels in COCO format

### Shadow Mode
- Runs baseline and candidate models simultaneously
- Detects disagreements between model versions
- Mines hard cases for retraining

### Hard Case Mining
- Embeds failed predictions using CLIP
- Clusters failures with HDBSCAN
- Surfaces systematic failure modes

---

*TODO: Add implementation details*
