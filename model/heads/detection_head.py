"""FCOS-style anchor-free detection head.

Anchor-free detection following the FCOS paper:
"FCOS: Fully Convolutional One-Stage Object Detection" (Tian et al., 2019)

Each location predicts:
- Classification: probability of each class
- Box regression: distances to left, top, right, bottom edges
- Centerness: how centered the prediction is (helps suppress low-quality detections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class DetectionOutput(NamedTuple):
    """Output from detection head at each scale."""
    cls_logits: torch.Tensor  # (B, num_classes, H, W)
    bbox_pred: torch.Tensor   # (B, 4, H, W) - l, t, r, b distances
    centerness: torch.Tensor  # (B, 1, H, W)


class ScaleHead(nn.Module):
    """Shared convolution head applied at each feature scale.

    Produces classification, box regression, and centerness outputs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 9,
        num_convs: int = 4,
    ):
        super().__init__()

        # Classification branch
        cls_layers = []
        for i in range(num_convs):
            cls_layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            cls_layers.append(nn.GroupNorm(32, hidden_channels))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(hidden_channels, num_classes, kernel_size=3, padding=1)

        # Box regression branch
        reg_layers = []
        for i in range(num_convs):
            reg_layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            reg_layers.append(nn.GroupNorm(32, hidden_channels))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_layers)
        self.bbox_pred = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)

        # Centerness branch (shares regression tower)
        self.centerness = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

        # Scale parameter for box regression (learned per scale)
        self.scale = nn.Parameter(torch.ones(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper initialization."""
        for module in [self.cls_tower, self.reg_tower]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)

        # Classification head - bias initialization for focal loss
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -4.6)  # -log((1-0.01)/0.01)

        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)

        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.zeros_(self.centerness.bias)

    def forward(self, x: torch.Tensor) -> DetectionOutput:
        """Forward pass for one scale.

        Args:
            x: Feature map of shape (B, C, H, W)

        Returns:
            DetectionOutput with cls_logits, bbox_pred, centerness
        """
        cls_tower_out = self.cls_tower(x)
        reg_tower_out = self.reg_tower(x)

        cls_logits = self.cls_logits(cls_tower_out)
        bbox_pred = F.relu(self.bbox_pred(reg_tower_out)) * self.scale
        centerness = self.centerness(reg_tower_out)

        return DetectionOutput(
            cls_logits=cls_logits,
            bbox_pred=bbox_pred,
            centerness=centerness,
        )


class DetectionHead(nn.Module):
    """Multi-scale FCOS-style detection head.

    Processes features at P3, P4, P5 scales and produces detection outputs
    at each scale. During inference, outputs from all scales are combined.

    Args:
        in_channels_p3: Number of channels in P3 features (default: 40 for EfficientNet-B0)
        in_channels_p4: Number of channels in P4 features (default: 112)
        in_channels_p5: Number of channels in P5 features (default: 320)
        hidden_channels: Number of channels in head convolutions
        num_classes: Number of object classes (including background)
        num_convs: Number of convolutions in each branch

    BDD100K classes (9 total):
        0: background
        1: car
        2: truck
        3: bus
        4: person
        5: bicycle
        6: motorcycle
        7: traffic_light
        8: traffic_sign
    """

    def __init__(
        self,
        in_channels_p3: int = 40,
        in_channels_p4: int = 112,
        in_channels_p5: int = 320,
        hidden_channels: int = 256,
        num_classes: int = 9,
        num_convs: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Project all scales to same channel dimension
        self.proj_p3 = nn.Conv2d(in_channels_p3, hidden_channels, kernel_size=1)
        self.proj_p4 = nn.Conv2d(in_channels_p4, hidden_channels, kernel_size=1)
        self.proj_p5 = nn.Conv2d(in_channels_p5, hidden_channels, kernel_size=1)

        # Shared head (applied to projected features at each scale)
        self.head = ScaleHead(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_convs=num_convs,
        )

        print(f"[DetectionHead] FCOS-style anchor-free head")
        print(f"  Input channels: P3={in_channels_p3}, P4={in_channels_p4}, P5={in_channels_p5}")
        print(f"  Hidden channels: {hidden_channels}")
        print(f"  Classes: {num_classes}")

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> dict[str, list[DetectionOutput]]:
        """Process multi-scale features and produce detection outputs.

        Args:
            p3: P3 features (B, 40, H/8, W/8)
            p4: P4 features (B, 112, H/16, W/16)
            p5: P5 features (B, 320, H/32, W/32)

        Returns:
            Dictionary with 'outputs' key containing list of DetectionOutput
            for each scale (P3, P4, P5)
        """
        # Project to common channel dimension
        p3_proj = self.proj_p3(p3)
        p4_proj = self.proj_p4(p4)
        p5_proj = self.proj_p5(p5)

        # Apply head at each scale
        out_p3 = self.head(p3_proj)
        out_p4 = self.head(p4_proj)
        out_p5 = self.head(p5_proj)

        return {
            'p3': out_p3,
            'p4': out_p4,
            'p5': out_p5,
        }

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def test_detection_head():
    """Test detection head with expected shapes."""
    print("\n=== Testing DetectionHead ===\n")

    device = torch.device('cpu')

    # Create head
    head = DetectionHead(
        in_channels_p3=40,
        in_channels_p4=112,
        in_channels_p5=320,
        num_classes=9,
    ).to(device)

    # Create dummy features (matching EfficientNet-B0 output for 640x640 input)
    batch_size = 2
    p3 = torch.randn(batch_size, 40, 80, 80, device=device)
    p4 = torch.randn(batch_size, 112, 40, 40, device=device)
    p5 = torch.randn(batch_size, 320, 20, 20, device=device)

    # Forward pass
    outputs = head(p3, p4, p5)

    # Check outputs
    for scale_name, out in outputs.items():
        print(f"\n{scale_name}:")
        print(f"  cls_logits: {out.cls_logits.shape}")
        print(f"  bbox_pred: {out.bbox_pred.shape}")
        print(f"  centerness: {out.centerness.shape}")

    # Verify shapes
    # P3: 80x80
    assert outputs['p3'].cls_logits.shape == (batch_size, 9, 80, 80)
    assert outputs['p3'].bbox_pred.shape == (batch_size, 4, 80, 80)
    assert outputs['p3'].centerness.shape == (batch_size, 1, 80, 80)

    # P4: 40x40
    assert outputs['p4'].cls_logits.shape == (batch_size, 9, 40, 40)
    assert outputs['p4'].bbox_pred.shape == (batch_size, 4, 40, 40)
    assert outputs['p4'].centerness.shape == (batch_size, 1, 40, 40)

    # P5: 20x20
    assert outputs['p5'].cls_logits.shape == (batch_size, 9, 20, 20)
    assert outputs['p5'].bbox_pred.shape == (batch_size, 4, 20, 20)
    assert outputs['p5'].centerness.shape == (batch_size, 1, 20, 20)

    params = head.get_num_parameters()
    print(f"\nTotal parameters: {params:,}")

    print("\n✅ All detection head tests passed!\n")


if __name__ == '__main__':
    test_detection_head()
