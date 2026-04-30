"""U-Net style segmentation heads for lane and drivable area.

Both lane segmentation and drivable area are pixel-wise classification tasks.
They share the same architecture but are instantiated separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """U-Net style decoder for semantic segmentation.

    Takes P3 features (highest resolution from backbone) and upsamples
    8× back to input resolution using transposed convolutions.

    Args:
        in_channels: Number of input channels (40 for EfficientNet-B0 P3)
        hidden_channels: Number of channels in decoder layers
        num_classes: Number of output classes (2 for binary, 3+ for multi-class)
        upsample_factor: Total upsampling factor (8 for P3 -> full res)

    Example:
        >>> head = SegmentationHead(in_channels=40, num_classes=2)
        >>> p3_feat = torch.randn(1, 40, 80, 80)
        >>> output = head(p3_feat)
        >>> print(output.shape)  # (1, 2, 640, 640)
    """

    def __init__(
        self,
        in_channels: int = 40,
        hidden_channels: int = 128,
        num_classes: int = 2,
        upsample_factor: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Calculate intermediate sizes for 8× upsampling (3 stages of 2×)
        # Stage 1: 80 -> 160
        # Stage 2: 160 -> 320
        # Stage 3: 320 -> 640

        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks with upsampling
        self.decoder1 = self._make_decoder_block(hidden_channels, hidden_channels)
        self.decoder2 = self._make_decoder_block(hidden_channels, hidden_channels // 2)
        self.decoder3 = self._make_decoder_block(hidden_channels // 2, hidden_channels // 4)

        # Final classification head
        self.classifier = nn.Conv2d(hidden_channels // 4, num_classes, kernel_size=1)

        self._init_weights()

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,  # 2× upsample
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            p3: P3 features of shape (B, 40, H/8, W/8)

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        x = self.proj(p3)         # (B, 128, 80, 80)
        x = self.decoder1(x)      # (B, 128, 160, 160)
        x = self.decoder2(x)      # (B, 64, 320, 320)
        x = self.decoder3(x)      # (B, 32, 640, 640)
        x = self.classifier(x)    # (B, num_classes, 640, 640)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def test_segmentation_head():
    """Test segmentation head shapes."""
    print("\n=== Testing SegmentationHead ===\n")

    device = torch.device('cpu')

    # Test lane segmentation (2 classes: background, lane)
    lane_head = SegmentationHead(in_channels=40, num_classes=2).to(device)
    p3 = torch.randn(2, 40, 80, 80, device=device)
    lane_out = lane_head(p3)
    print(f"Lane head output: {lane_out.shape} (expected: [2, 2, 640, 640])")
    assert lane_out.shape == (2, 2, 640, 640), f"Shape mismatch: {lane_out.shape}"
    print(f"Lane head parameters: {lane_head.get_num_parameters():,}")

    # Test drivable area (2 classes: background, drivable)
    drivable_head = SegmentationHead(in_channels=40, num_classes=2).to(device)
    drivable_out = drivable_head(p3)
    print(f"Drivable head output: {drivable_out.shape} (expected: [2, 2, 640, 640])")
    assert drivable_out.shape == (2, 2, 640, 640)
    print(f"Drivable head parameters: {drivable_head.get_num_parameters():,}")

    print("\n✅ All segmentation head tests passed!\n")


if __name__ == '__main__':
    test_segmentation_head()
