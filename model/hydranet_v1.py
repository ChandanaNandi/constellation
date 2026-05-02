"""HydraNet v1: Detection-focused multi-task model.

Simplified version focusing on detection training first.
Lane and drivable heads are included but can be disabled during training
if ground truth masks aren't available.
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import EfficientNetBackbone
from .heads import DetectionHead


class HydraNetV1(nn.Module):
    """HydraNet v1 with detection head.

    Simplified architecture for initial training with BDD100K detection labels.
    Other heads (lane, drivable) can be added back once masks are available.

    Args:
        num_classes: Number of detection classes (10 for BDD100K YOLO format)
        pretrained_backbone: Whether to use pretrained EfficientNet
        freeze_backbone: Whether to freeze backbone initially
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Shared backbone
        self.backbone = EfficientNetBackbone(
            model_name="efficientnet_b0",
            pretrained=pretrained_backbone,
            frozen=freeze_backbone,
        )

        channels = self.backbone.out_channels

        # Detection head (FCOS-style)
        self.detection_head = DetectionHead(
            in_channels_p3=channels['p3'],
            in_channels_p4=channels['p4'],
            in_channels_p5=channels['p5'],
            num_classes=num_classes,
        )

        print(f"\n[HydraNet-v1] Detection model initialized")
        print(f"  Detection classes: {num_classes}")
        print(f"  Total parameters: {self.get_num_parameters():,}")

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary with detection outputs per scale
        """
        features = self.backbone(x)

        return {
            'detection': self.detection_head(
                features['p3'],
                features['p4'],
                features['p5'],
            ),
        }

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_hydranet_v1():
    """Test HydraNet v1."""
    print("\n" + "="*60)
    print("Testing HydraNet V1")
    print("="*60 + "\n")

    model = HydraNetV1(num_classes=10, pretrained_backbone=True)
    model.eval()

    x = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        outputs = model(x)

    det = outputs['detection']
    for scale in ['p3', 'p4', 'p5']:
        print(f"{scale}: cls={det[scale].cls_logits.shape}, bbox={det[scale].bbox_pred.shape}")

    print(f"\nTotal params: {model.get_num_parameters():,}")
    print("\n✅ HydraNet V1 test passed!")


if __name__ == "__main__":
    test_hydranet_v1()
