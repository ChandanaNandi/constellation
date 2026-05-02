# Constellation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**Multi-task vision system for autonomous driving, inspired by Tesla's HydraNet architecture.**

Built with PyTorch. Trained on BDD100K. Designed for real-world deployment.

![Constellation Detection Demo](assets/demo.gif)

*Multi-task object detection on BDD100K driving scenes — cars, traffic lights, signs, and pedestrians.*

---

## Highlights

- **HydraNet-style Architecture** — Shared EfficientNet-B0 backbone with task-specific heads
- **FCOS Anchor-Free Detection** — Modern detection without anchor boxes, multi-scale (P3/P4/P5)
- **Production Pipeline** — Focal loss, mAP evaluation, W&B experiment tracking
- **Cloud GPU Ready** — Full training infrastructure for NVIDIA H100 on RunPod
- **Auto-Labeling** — YOLOv8 + MobileSAM pipeline for automated annotation

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 4.0% (5 epochs, 1K subset) |
| Training Loss | 2.78 → 1.45 |
| Classification Loss | 0.43 → 0.24 |
| Model Size | 8.5M parameters |

*Trained on a 1,000-image subset of BDD100K (10 object classes) to validate the architecture and training pipeline. Cloud GPU infrastructure validated on NVIDIA H100; full-scale training deprioritized to focus on multi-task expansion (Phase 4).*

![Detection Example](assets/detection_example.jpg)

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

**Model Specs:**
- **Parameters:** 8.5M (efficient for edge deployment)
- **Backbone:** EfficientNet-B0 (ImageNet pretrained, frozen)
- **Detection:** FCOS anchor-free with focal loss
- **Classes:** 10 (person, car, truck, bus, bike, motor, rider, traffic light, traffic sign, train)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/ChandanaNandi/constellation.git
cd constellation
pip install -r requirements.txt

# Train locally (M4 Pro / CPU)
python train.py --subset 1000 --epochs 5 --batch-size 8 --device mps

# Train on GPU
python train.py --epochs 50 --batch-size 64 --device cuda --use-wandb

# Run inference
python inference.py --checkpoint checkpoints/best.pt --num-images 20
```

---

## Technical Deep Dive

### Training Pipeline
- **Classification:** Focal loss (α=0.25, γ=2.0) with proper positive sample normalization
- **Bounding Box:** GIoU loss for accurate localization
- **Centerness:** Binary cross-entropy for center-ness prediction

### FCOS Target Assignment
Multi-scale assignment based on object size:
| Scale | Stride | Object Size |
|-------|--------|-------------|
| P3 | 8 | < 32px |
| P4 | 16 | 32-64px |
| P5 | 32 | > 64px |

### Debugging Journey
- Fixed focal loss normalization bug (cls_loss was 2 orders of magnitude too low)
- Corrected FCOS size ranges for BDD100K object distribution
- Validated training with overfit test on single batch

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
│   └── shadow_mode.py      # Model comparison
├── train.py                # Training with W&B
├── inference.py            # Visualization
└── deployment/
    ├── export_onnx.py      # ONNX export
    └── quantize.py         # INT8 quantization
```

---

## What I Learned

Building Constellation taught me:

1. **Multi-task architectures** — How shared backbones enable efficient inference across tasks
2. **FCOS target assignment** — The math behind anchor-free detection and why proper size ranges matter
3. **Loss debugging** — Finding normalization bugs requires systematic overfit testing
4. **Cloud GPU workflow** — RunPod, tmux sessions, W&B remote monitoring
5. **Data engineering** — Auto-labeling pipelines and shadow mode evaluation

These are the skills Tesla's Autopilot team uses daily.

---

## Roadmap

- [x] Phase 1: Data engine with auto-labeling
- [x] Phase 2: HydraNet architecture with FCOS detection
- [x] Phase 3: Training pipeline with cloud GPU support
- [ ] Phase 4: Lane segmentation + drivable area heads
- [ ] Phase 5: Shadow mode evaluation + INT8 quantization

---

## Dataset

**BDD100K** — Berkeley Deep Drive
- 70K training images, 10K validation
- 10 object classes for autonomous driving
- YOLO format labels

---

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [BDD100K: A Diverse Driving Dataset](https://arxiv.org/abs/1805.04687)
- Tesla AI Day presentations on HydraNet architecture

---

## License

MIT — see [LICENSE](LICENSE)

---

**Built by Chandana Reddy** | [GitHub](https://github.com/ChandanaNandi) | Open to AI/ML opportunities
