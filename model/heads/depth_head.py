
"""Depth estimation head for monocular depth prediction.

Predicts per-pixel depth (distance from camera) in meters.
Uses knowledge distillation from Depth Anything / MiDaS for supervision.
"""

import torch
import torch.nn as nn


class DepthHead(nn.Module):
    """Regression decoder for monocular depth estimation.

    Similar to segmentation head but outputs continuous depth values
    instead of class logits. Uses sigmoid activation to output
    normalized depth (0-1) which is then scaled to actual meters.

    Args:
        in_channels: Number of input channels (40 for EfficientNet-B0 P3)
        hidden_channels: Number of channels in decoder layers
        max_depth: Maximum depth in meters for scaling (default: 80m)

    Example:
        >>> head = DepthHead(in_channels=40)
        >>> p3_feat = torch.randn(1, 40, 80, 80)
        >>> depth = head(p3_feat)
        >>> print(depth.shape)  # (1, 1, 640, 640)
    """

    def __init__(
        self,
        in_channels: int = 40,
        hidden_channels: int = 128,
        max_depth: float = 80.0,
    ):
        super().__init__()
        self.max_depth = max_depth

        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks with upsampling (3 stages of 2× = 8× total)
        self.decoder1 = self._make_decoder_block(hidden_channels, hidden_channels)
        self.decoder2 = self._make_decoder_block(hidden_channels, hidden_channels // 2)
        self.decoder3 = self._make_decoder_block(hidden_channels // 2, hidden_channels // 4)

        # Final depth prediction (single channel, normalized 0-1)
        self.depth_pred = nn.Sequential(
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output 0-1, scale to meters
        )

        self._init_weights()

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
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
            Depth map of shape (B, 1, H, W) with values in [0, max_depth] meters
        """
        x = self.proj(p3)           # (B, 128, 80, 80)
        x = self.decoder1(x)        # (B, 128, 160, 160)
        x = self.decoder2(x)        # (B, 64, 320, 320)
        x = self.decoder3(x)        # (B, 32, 640, 640)
        depth = self.depth_pred(x)  # (B, 1, 640, 640) in [0, 1]

        # Scale to actual depth in meters
        depth = depth * self.max_depth

        return depth

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def test_depth_head():
    """Test depth head shapes."""
    print("\n=== Testing DepthHead ===\n")

    device = torch.device('cpu')

    head = DepthHead(in_channels=40, max_depth=80.0).to(device)
    p3 = torch.randn(2, 40, 80, 80, device=device)

    depth = head(p3)
    print(f"Depth output: {depth.shape} (expected: [2, 1, 640, 640])")
    print(f"Depth range: [{depth.min().item():.2f}, {depth.max().item():.2f}] meters")

    assert depth.shape == (2, 1, 640, 640), f"Shape mismatch: {depth.shape}"
    assert depth.min() >= 0, "Depth should be non-negative"
    assert depth.max() <= 80.0, "Depth should be <= max_depth"

    print(f"Depth head parameters: {head.get_num_parameters():,}")

    print("\n✅ All depth head tests passed!\n")


if __name__ == '__main__':
    test_depth_head()
