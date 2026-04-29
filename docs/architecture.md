# Constellation Architecture

## Overview

Constellation is a multi-task vision system inspired by Tesla's HydraNet architecture. It uses a shared backbone with multiple task-specific heads to efficiently process driving scene images.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image (640x640)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Backbone (EfficientNet-B0)             │
│                                                                  │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│    │   P3     │    │   P4     │    │   P5     │                │
│    │ (80x80)  │    │ (40x40)  │    │ (20x20)  │                │
│    └──────────┘    └──────────┘    └──────────┘                │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  Detection Head   │ │  Segmentation     │ │  Classification   │
│  (FCOS-style)     │ │  Heads (U-Net)    │ │  Head             │
└───────────────────┘ └───────────────────┘ └───────────────────┘
        │                      │                     │
        ▼                      ▼                     ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  Bounding Boxes   │ │  Lane Lines       │ │  Traffic Light    │
│  + Classes        │ │  Drivable Area    │ │  State            │
│                   │ │  Depth Map        │ │                   │
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

## Task Heads

### 1. Object Detection (FCOS-style)
- Anchor-free detection
- Outputs: class scores, box coordinates, centerness
- Classes: vehicle, pedestrian, cyclist, etc.

### 2. Lane Segmentation
- U-Net decoder architecture
- Binary segmentation of lane markings
- Output: 640x640 binary mask

### 3. Drivable Area Segmentation
- U-Net decoder architecture
- Binary segmentation of drivable regions
- Output: 640x640 binary mask

### 4. Depth Estimation
- Decoder predicting per-pixel depth in meters
- Regression with smooth L1 loss
- Output: 640x640 depth map

### 5. Traffic Light Classification
- Global average pooling + linear classifier
- 4 classes: red, yellow, green, none
- Output: class probabilities

## Multi-Task Loss

Uses uncertainty-weighted loss from Kendall et al. 2018:

```
L_total = Σ (1/2σᵢ²) * Lᵢ + log(σᵢ)
```

Where σᵢ is a learned parameter for each task.

## Data Pipeline

<!-- TODO: Add data pipeline diagram -->

## Deployment

<!-- TODO: Add deployment architecture -->

---

*Last updated: Phase 1*
