"""Traffic light classification head.

Classifies the dominant traffic light state in the image.
Uses P5 features (most semantic, lowest resolution) for global context.
"""

import torch
import torch.nn as nn


class TrafficLightHead(nn.Module):
    """Classification head for traffic light state.

    Uses global average pooling over P5 features followed by
    a linear classifier. This captures the dominant traffic light
    state in the image (not per-light classification).

    Args:
        in_channels: Number of input channels (320 for EfficientNet-B0 P5)
        hidden_channels: Number of hidden units in classifier
        num_classes: Number of traffic light classes (4: red, yellow, green, none)

    Example:
        >>> head = TrafficLightHead(in_channels=320, num_classes=4)
        >>> p5_feat = torch.randn(1, 320, 20, 20)
        >>> logits = head(p5_feat)
        >>> print(logits.shape)  # (1, 4)
    """

    # Traffic light classes
    CLASSES = ['red', 'yellow', 'green', 'none']

    def __init__(
        self,
        in_channels: int = 320,
        hidden_channels: int = 256,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Global average pooling + MLP classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p5: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            p5: P5 features of shape (B, 320, H/32, W/32)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        return self.classifier(p5)

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def test_traffic_light_head():
    """Test traffic light head shapes."""
    print("\n=== Testing TrafficLightHead ===\n")

    device = torch.device('cpu')

    head = TrafficLightHead(in_channels=320, num_classes=4).to(device)
    p5 = torch.randn(2, 320, 20, 20, device=device)

    logits = head(p5)
    print(f"Traffic light output: {logits.shape} (expected: [2, 4])")

    assert logits.shape == (2, 4), f"Shape mismatch: {logits.shape}"

    # Test softmax
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities sum: {probs.sum(dim=1)} (should be ~1.0)")

    print(f"Traffic light head parameters: {head.get_num_parameters():,}")

    print("\n✅ All traffic light head tests passed!\n")


if __name__ == '__main__':
    test_traffic_light_head()
