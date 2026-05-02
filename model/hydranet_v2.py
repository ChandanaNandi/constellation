"""HydraNet v2: Multi-task model with Detection + Segmentation.

Phase 4: True multi-task learning with shared backbone.
- Detection head: FCOS-style anchor-free detection
- Segmentation head: Drivable area (road/sidewalk)
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import EfficientNetBackbone
from .heads import DetectionHead, SegmentationHead


class HydraNetV2(nn.Module):
    """HydraNet v2 with detection + segmentation heads.

    Multi-task architecture:
        - Shared EfficientNet-B0 backbone (frozen or trainable)
        - FCOS detection head for bounding boxes
        - Segmentation head for drivable area

    One forward pass produces both detection and segmentation outputs.

    Args:
        num_det_classes: Number of detection classes (8 for Cityscapes)
        num_seg_classes: Number of segmentation classes (3 for drivable area)
        pretrained_backbone: Whether to use pretrained EfficientNet
        freeze_backbone: Whether to freeze backbone initially
    """

    def __init__(
        self,
        num_det_classes: int = 8,
        num_seg_classes: int = 3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes

        # Shared backbone
        self.backbone = EfficientNetBackbone(
            model_name="efficientnet_b0",
            pretrained=pretrained_backbone,
            frozen=freeze_backbone,
        )

        channels = self.backbone.out_channels

        # Task 1: Detection head (FCOS-style)
        self.detection_head = DetectionHead(
            in_channels_p3=channels['p3'],
            in_channels_p4=channels['p4'],
            in_channels_p5=channels['p5'],
            num_classes=num_det_classes,
        )

        # Task 2: Segmentation head (drivable area)
        # Uses P3 features for highest resolution
        self.segmentation_head = SegmentationHead(
            in_channels=channels['p3'],
            hidden_channels=128,
            num_classes=num_seg_classes,
        )

        self._print_summary()

    def _print_summary(self):
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"HydraNet V2 - Multi-Task Model")
        print(f"{'='*60}")
        print(f"  Detection classes: {self.num_det_classes}")
        print(f"  Segmentation classes: {self.num_seg_classes}")
        print(f"  Backbone: EfficientNet-B0")
        print(f"  {'='*40}")
        print(f"  Parameters:")
        print(f"    Backbone:     {self._count_params(self.backbone):>10,}")
        print(f"    Detection:    {self._count_params(self.detection_head):>10,}")
        print(f"    Segmentation: {self._count_params(self.segmentation_head):>10,}")
        print(f"    {'─'*30}")
        print(f"    Total:        {self.get_num_parameters():>10,}")
        print(f"    Trainable:    {self.get_trainable_parameters():>10,}")
        print(f"{'='*60}\n")

    def _count_params(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary with:
                - 'detection': dict with p3/p4/p5 outputs
                - 'segmentation': (B, num_seg_classes, H, W) logits
        """
        # Shared feature extraction
        features = self.backbone(x)

        # Task 1: Detection
        detection_out = self.detection_head(
            features['p3'],
            features['p4'],
            features['p5'],
        )

        # Task 2: Segmentation (uses P3 for high resolution)
        segmentation_out = self.segmentation_head(features['p3'])

        return {
            'detection': detection_out,
            'segmentation': segmentation_out,
        }

    def forward_detection_only(self, x: torch.Tensor) -> dict:
        """Forward pass for detection only (faster inference)."""
        features = self.backbone(x)
        return {
            'detection': self.detection_head(
                features['p3'],
                features['p4'],
                features['p5'],
            ),
        }

    def forward_segmentation_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation only."""
        features = self.backbone(x)
        return self.segmentation_head(features['p3'])

    def freeze_backbone(self) -> None:
        """Freeze backbone weights."""
        self.backbone.freeze()
        print("[HydraNetV2] Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights."""
        self.backbone.unfreeze()
        print("[HydraNetV2] Backbone unfrozen")

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_hydranet_v2():
    """Test HydraNet v2 multi-task model."""
    print("\n" + "="*60)
    print("Testing HydraNet V2 (Multi-Task)")
    print("="*60 + "\n")

    # Create model
    model = HydraNetV2(
        num_det_classes=8,
        num_seg_classes=3,
        pretrained_backbone=True,
        freeze_backbone=True,
    )
    model.eval()

    # Test input (Cityscapes aspect ratio: 2:1)
    x = torch.randn(2, 3, 512, 1024)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        outputs = model(x)

    # Check detection outputs
    det = outputs['detection']
    print(f"\nDetection outputs:")
    for scale in ['p3', 'p4', 'p5']:
        print(f"  {scale}: cls={det[scale].cls_logits.shape}, bbox={det[scale].bbox_pred.shape}")

    # Check segmentation output
    seg = outputs['segmentation']
    print(f"\nSegmentation output: {seg.shape}")
    assert seg.shape == (2, 3, 512, 1024), f"Expected (2, 3, 512, 1024), got {seg.shape}"

    # Test single-task forwards
    print(f"\nTesting single-task forwards...")
    with torch.no_grad():
        det_only = model.forward_detection_only(x)
        seg_only = model.forward_segmentation_only(x)

    print(f"  Detection-only: {det_only['detection']['p3'].cls_logits.shape}")
    print(f"  Segmentation-only: {seg_only.shape}")

    print("\n" + "="*60)
    print("HydraNet V2 test passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_hydranet_v2()
