# Constellation Architecture

## Overview

Constellation is a multi-task vision system inspired by Tesla's HydraNet architecture. It uses a shared backbone with multiple task-specific heads to efficiently process driving scene images in a single forward pass.

**Design Principles:**
1. **Multi-task learning** — one model, multiple outputs
2. **Shared backbone** — parameter efficiency, shared representations
3. **Multi-scale features** — different tasks consume different scales
4. **Uncertainty-weighted loss** — automatic task balancing (Kendall et al. 2018)

## System Architecture

```
                     Input Image (3, 640, 640)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Shared Backbone (EfficientNet-B0)                 │
│                         5.3M parameters, pretrained                  │
│                                                                      │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│    │     P3       │    │     P4       │    │     P5       │        │
│    │  (40, 80×80) │    │ (112, 40×40) │    │ (320, 20×20) │        │
│    │   1/8 scale  │    │  1/16 scale  │    │  1/32 scale  │        │
│    └──────────────┘    └──────────────┘    └──────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
           │                    │                    │
     ┌─────┴────────────────────┼────────────────────┼────────┐
     │                          │                    │        │
     ▼                          ▼                    ▼        ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Detection│  │  Lane    │  │ Drivable │  │  Depth   │  │ Traffic  │
│   Head   │  │   Seg    │  │   Area   │  │   Head   │  │  Light   │
│  (FCOS)  │  │  (U-Net) │  │  (U-Net) │  │  (Reg)   │  │  (Cls)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │             │
     ▼             ▼             ▼             ▼             ▼
  Boxes +      Lane Mask     Drive Mask    Depth Map      4-class
  Classes      (640×640)     (640×640)     (640×640)      logits
```

## Components

### Backbone: EfficientNet-B0

**File:** `model/backbones/efficientnet.py`

- **Architecture:** EfficientNet-B0 from `timm` (PyTorch Image Models)
- **Parameters:** 3.6M (pretrained on ImageNet)
- **Feature scales:** P3, P4, P5 at indices [2, 3, 4]
- **Output channels:** P3=40, P4=112, P5=320
- **Training mode:** Initially frozen, unfrozen for fine-tuning

```python
backbone = EfficientNetBackbone(pretrained=True)
features = backbone(image)  # Returns {'p3': ..., 'p4': ..., 'p5': ...}
```

### Detection Head: FCOS-style Anchor-Free

**File:** `model/heads/detection_head.py`

- **Architecture:** FCOS (Fully Convolutional One-Stage Object Detection)
- **Parameters:** ~4.9M
- **Input:** Multi-scale features P3, P4, P5
- **Outputs per scale:**
  - `cls_logits`: (B, 9, H, W) — class probabilities
  - `bbox_pred`: (B, 4, H, W) — distances to box edges (l, t, r, b)
  - `centerness`: (B, 1, H, W) — prediction quality score

**BDD100K Classes (9):**
| ID | Class |
|----|-------|
| 0 | background |
| 1 | car |
| 2 | truck |
| 3 | bus |
| 4 | person |
| 5 | bicycle |
| 6 | motorcycle |
| 7 | traffic_light |
| 8 | traffic_sign |

### Lane Segmentation Head: U-Net Decoder

**File:** `model/heads/segmentation_head.py`

- **Input:** P3 features (highest resolution)
- **Architecture:** Transposed convolutions for 8× upsampling
- **Output:** (B, num_classes, 640, 640)
- **Classes:** background, lane line
- **Loss:** Cross-entropy + Dice loss

### Drivable Area Head: U-Net Decoder

**File:** `model/heads/segmentation_head.py` (shared class)

- **Input:** P3 features
- **Output:** (B, 2, 640, 640) binary mask
- **Loss:** BCE + Dice loss

### Depth Estimation Head: Regression Decoder

**File:** `model/heads/depth_head.py`

- **Input:** P3 features
- **Output:** (B, 1, 640, 640) — depth in meters per pixel
- **Training:** Knowledge distillation from Depth Anything
- **Loss:** Smooth L1 or scale-invariant log loss

### Traffic Light Classification Head

**File:** `model/heads/traffic_light_head.py`

- **Input:** P5 features (most semantic)
- **Architecture:** Global average pooling → Linear(320, 4)
- **Output:** (B, 4) logits
- **Classes:** red, yellow, green, none
- **Loss:** Cross-entropy

## Multi-Task Loss

**File:** `model/losses/multi_task_loss.py`

Uses uncertainty-weighted loss from [Kendall et al. 2018](https://arxiv.org/abs/1705.07115):

```
L_total = Σ (1/(2σ²ᵢ)) × Lᵢ + log(σᵢ)
```

Where σᵢ is a learned parameter for each task, automatically balancing:
- Tasks with noisy labels → lower weight
- Tasks that are "easy" → higher weight

**Per-task losses:**
| Task | Loss Function |
|------|--------------|
| Detection | Focal loss + IoU loss + BCE |
| Lane | Cross-entropy + Dice |
| Drivable | BCE + Dice |
| Depth | Smooth L1 |
| Traffic Light | Cross-entropy |

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| Backbone (EfficientNet-B0) | 3.6M |
| Detection Head | 4.9M |
| Lane Segmentation | ~1.5M |
| Drivable Segmentation | ~1.5M |
| Depth Head | ~1.5M |
| Traffic Light Head | ~50K |
| **Total** | **~13M** |

## Implementation Status

- [x] **Backbone:** EfficientNet-B0 with multi-scale extraction
- [x] **Detection Head:** FCOS-style anchor-free
- [ ] **Lane Segmentation:** U-Net decoder
- [ ] **Drivable Area:** U-Net decoder
- [ ] **Depth Head:** Regression decoder
- [ ] **Traffic Light:** Classification head
- [ ] **Multi-Task Loss:** Uncertainty weighting
- [ ] **Full HydraNet:** Assembly + forward pass

## Usage

```python
from model.hydranet import HydraNet

model = HydraNet()
outputs = model(image)  # (B, 3, 640, 640)

# Access individual task outputs
boxes = outputs['detection']      # Multi-scale FCOS outputs
lanes = outputs['lane']           # (B, 2, 640, 640)
drivable = outputs['drivable']    # (B, 2, 640, 640)
depth = outputs['depth']          # (B, 1, 640, 640)
traffic = outputs['traffic_light'] # (B, 4)
```

## References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) (Tian et al., 2019)
- [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) (Kendall et al., 2018)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- [Tesla AI Day 2021: HydraNet Architecture](https://www.youtube.com/watch?v=j0z4FweCy4M)

---

*Last updated: Phase 2 - Day 3*
