# Constellation

**Multi-task vision system for autonomous driving, inspired by Tesla's HydraNet architecture.**

Built with PyTorch. Trained on BDD100K. Designed for real-world deployment.

---

## What This Project Demonstrates

- **HydraNet-style Architecture**: Shared EfficientNet-B0 backbone with task-specific heads, enabling efficient multi-task inference
- **FCOS Anchor-Free Detection**: Modern anchor-free object detection with multi-scale feature processing (P3/P4/P5)
- **Production Training Pipeline**: Focal loss with proper normalization, mAP evaluation, W&B experiment tracking
- **Cloud GPU Workflow**: Full training infrastructure for NVIDIA H100 on RunPod with Docker and persistent storage
- **Data Engine**: Auto-labeling pipeline with YOLOv8 + MobileSAM, shadow mode evaluation, hard case mining

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Input Image (640×640)         │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     EfficientNet-B0 Backbone (Frozen)   │
                    │         Pretrained on ImageNet          │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │   P3 Features     │   │   P4 Features     │   │   P5 Features     │
    │   (80×80, 40ch)   │   │   (40×40, 112ch)  │   │   (20×20, 320ch)  │
    └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │      FCOS Detection Head (256ch)        │
                    │  Classification + BBox + Centerness     │
                    └─────────────────────────────────────────┘
```

**Model Stats:**
- Parameters: 8.5M (efficient for edge deployment)
- Backbone: EfficientNet-B0 (pretrained, frozen)
- Detection: FCOS-style anchor-free with focal loss
- Classes: 10 (person, car, truck, bus, bike, motor, rider, traffic light, traffic sign, train)

---

## Training Pipeline

### Loss Function
- **Classification**: Focal loss (alpha=0.25, gamma=2.0) with proper normalization
- **Bounding Box**: GIoU loss for accurate localization
- **Centerness**: Binary cross-entropy for center-ness prediction

### FCOS Target Assignment
Multi-scale assignment based on object size:
- P3 (stride 8): Objects < 32px
- P4 (stride 16): Objects 32-64px
- P5 (stride 32): Objects > 64px

### Evaluation
- mAP@50 and mAP@50:95 via torchmetrics
- Per-class precision/recall analysis
- W&B logging for experiment tracking

---

## Results

Training on BDD100K (70K images, 10 classes):

| Metric | Value |
|--------|-------|
| Training Loss | 1.45 → decreasing |
| Classification Loss | 0.28 |
| BBox Loss | 0.57 |
| Centerness Loss | 0.60 |

*Loss curves show healthy convergence with proper normalization.*

---

## Quick Start

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA (for GPU training)

### Installation

```bash
git clone https://github.com/ChandanaNandi/constellation.git
cd constellation
pip install -r requirements.txt
```

### Training

```bash
# Local training (M4 Pro / CPU)
python train.py --subset 1000 --epochs 5 --batch-size 8 --device mps

# Full training (GPU)
python train.py --epochs 50 --batch-size 64 --device cuda --use-wandb
```

### Inference

```bash
python inference.py --checkpoint checkpoints/best.pt --num-images 20
```

---

## Project Structure

```
constellation/
├── model/
│   ├── hydranet_v1.py      # Main model architecture
│   ├── fcos_targets.py     # FCOS target assignment
│   ├── backbones/          # EfficientNet backbone
│   └── heads/              # Detection, segmentation heads
├── data_engine/
│   ├── data_loader.py      # BDD100K YOLO format loader
│   ├── auto_labeler.py     # YOLOv8 + MobileSAM pipeline
│   └── shadow_mode.py      # Prediction comparison
├── train.py                # Training script with W&B
├── inference.py            # Visualization and evaluation
└── deployment/
    ├── export_onnx.py      # ONNX export
    └── quantize.py         # INT8 quantization
```

---

## Technical Highlights

### Debugging & Validation
- Identified and fixed focal loss normalization bug (cls_loss was 2 orders of magnitude too low)
- Corrected FCOS size ranges for BDD100K object distribution
- Validated with overfit test on single batch

### Cloud Infrastructure
- RunPod deployment with NVIDIA H100 (80GB HBM3)
- tmux-based training for persistent sessions
- W&B integration for remote monitoring

### Design Decisions
- Frozen backbone for faster training and regularization
- Anchor-free detection for better small object handling
- Multi-scale features for objects from 8px to 512px

---

## Dataset

**BDD100K** - Berkeley Deep Drive dataset
- 70K training images
- 10K validation images
- 10 object classes relevant to autonomous driving
- YOLO format labels

---

## Roadmap

- [x] Phase 1: Data engine with auto-labeling
- [x] Phase 2: HydraNet architecture with FCOS detection
- [x] Phase 3: Training pipeline with cloud GPU support
- [ ] Phase 4: Lane segmentation + drivable area heads
- [ ] Phase 5: Shadow mode evaluation + INT8 quantization

---

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [BDD100K: A Diverse Driving Dataset](https://arxiv.org/abs/1805.04687)
- Tesla AI Day presentations on HydraNet architecture

---

## License

MIT - see [LICENSE](LICENSE)

---

**Built by Chandana Reddy** | [GitHub](https://github.com/ChandanaNandi)
