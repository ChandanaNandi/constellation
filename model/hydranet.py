"""HydraNet: Multi-task vision model for autonomous driving.

Architecture inspired by Tesla's HydraNet as presented at AI Day 2021/2022.
Single shared backbone with multiple task-specific heads for efficient
multi-task inference.

Tasks:
- Object detection (FCOS-style anchor-free)
- Lane segmentation
- Drivable area segmentation
- Depth estimation
- Traffic light classification
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import EfficientNetBackbone
from .heads import (
    DetectionHead,
    SegmentationHead,
    DepthHead,
    TrafficLightHead,
)


class HydraNet(nn.Module):
    """Multi-task vision model with shared backbone and multiple heads.

    One forward pass produces outputs for all 5 tasks, enabling efficient
    inference for autonomous driving perception.

    Args:
        backbone_name: Name of backbone architecture
        num_det_classes: Number of detection classes (9 for BDD100K)
        num_lane_classes: Number of lane classes (2 for binary)
        num_drivable_classes: Number of drivable area classes (2 for binary)
        num_tl_classes: Number of traffic light classes (4)
        pretrained_backbone: Whether to use pretrained backbone
        freeze_backbone: Whether to freeze backbone initially

    Example:
        >>> model = HydraNet()
        >>> image = torch.randn(1, 3, 640, 640)
        >>> outputs = model(image)
        >>> print(outputs.keys())  # dict_keys(['detection', 'lane', 'drivable', 'depth', 'traffic_light'])
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_det_classes: int = 9,
        num_lane_classes: int = 2,
        num_drivable_classes: int = 2,
        num_tl_classes: int = 4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Shared backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            frozen=freeze_backbone,
        )

        # Get backbone output channels
        channels = self.backbone.out_channels

        # Task-specific heads
        self.detection_head = DetectionHead(
            in_channels_p3=channels['p3'],
            in_channels_p4=channels['p4'],
            in_channels_p5=channels['p5'],
            num_classes=num_det_classes,
        )

        self.lane_head = SegmentationHead(
            in_channels=channels['p3'],
            num_classes=num_lane_classes,
        )

        self.drivable_head = SegmentationHead(
            in_channels=channels['p3'],
            num_classes=num_drivable_classes,
        )

        self.depth_head = DepthHead(
            in_channels=channels['p3'],
        )

        self.traffic_light_head = TrafficLightHead(
            in_channels=channels['p5'],
            num_classes=num_tl_classes,
        )

        print(f"\n[HydraNet] Model initialized")
        print(f"  Backbone: {backbone_name}")
        print(f"  Detection classes: {num_det_classes}")
        print(f"  Lane classes: {num_lane_classes}")
        print(f"  Drivable classes: {num_drivable_classes}")
        print(f"  Traffic light classes: {num_tl_classes}")
        print(f"  Total parameters: {self.get_num_parameters():,}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all heads.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with outputs for each task:
            - 'detection': dict with 'p3', 'p4', 'p5' DetectionOutputs
            - 'lane': (B, num_classes, H, W) segmentation logits
            - 'drivable': (B, num_classes, H, W) segmentation logits
            - 'depth': (B, 1, H, W) depth map in meters
            - 'traffic_light': (B, num_classes) classification logits
        """
        # Extract multi-scale features
        features = self.backbone(x)

        # Run all heads
        return {
            'detection': self.detection_head(
                features['p3'],
                features['p4'],
                features['p5'],
            ),
            'lane': self.lane_head(features['p3']),
            'drivable': self.drivable_head(features['p3']),
            'depth': self.depth_head(features['p3']),
            'traffic_light': self.traffic_light_head(features['p5']),
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone weights for transfer learning."""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for fine-tuning."""
        self.backbone.unfreeze()

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Get parameter count for each component."""
        return {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'detection': sum(p.numel() for p in self.detection_head.parameters()),
            'lane': sum(p.numel() for p in self.lane_head.parameters()),
            'drivable': sum(p.numel() for p in self.drivable_head.parameters()),
            'depth': sum(p.numel() for p in self.depth_head.parameters()),
            'traffic_light': sum(p.numel() for p in self.traffic_light_head.parameters()),
        }


def get_device() -> torch.device:
    """Get the best available device for training/inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_hydranet():
    """Test full HydraNet forward pass."""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    print("\n" + "="*60)
    print("Testing HydraNet Full Forward Pass")
    print("="*60 + "\n")

    device = torch.device('cpu')  # Use CPU for reliable testing

    # Create model
    model = HydraNet(pretrained_backbone=True)
    model = model.to(device)
    model.eval()

    # Create dummy input (640x640 driving image)
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # Print output shapes
    print("\n" + "-"*40)
    print("Output Shapes:")
    print("-"*40)

    # Detection (multi-scale)
    det = outputs['detection']
    for scale in ['p3', 'p4', 'p5']:
        print(f"Detection {scale}:")
        print(f"  cls_logits: {det[scale].cls_logits.shape}")
        print(f"  bbox_pred: {det[scale].bbox_pred.shape}")
        print(f"  centerness: {det[scale].centerness.shape}")

    print(f"\nLane: {outputs['lane'].shape}")
    print(f"Drivable: {outputs['drivable'].shape}")
    print(f"Depth: {outputs['depth'].shape}")
    print(f"Traffic Light: {outputs['traffic_light'].shape}")

    # Verify shapes
    assert det['p3'].cls_logits.shape == (batch_size, 9, 80, 80)
    assert det['p4'].cls_logits.shape == (batch_size, 9, 40, 40)
    assert det['p5'].cls_logits.shape == (batch_size, 9, 20, 20)
    assert outputs['lane'].shape == (batch_size, 2, 640, 640)
    assert outputs['drivable'].shape == (batch_size, 2, 640, 640)
    assert outputs['depth'].shape == (batch_size, 1, 640, 640)
    assert outputs['traffic_light'].shape == (batch_size, 4)

    # Parameter breakdown
    print("\n" + "-"*40)
    print("Parameter Breakdown:")
    print("-"*40)
    for component, count in model.get_parameter_breakdown().items():
        print(f"  {component}: {count:,}")
    print(f"  TOTAL: {model.get_num_parameters():,}")

    print("\n" + "="*60)
    print("✅ HydraNet forward pass successful!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_hydranet()
