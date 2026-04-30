"""EfficientNet backbone for feature extraction.

Uses timm (PyTorch Image Models) for pretrained weights.
Extracts multi-scale features at P3, P4, P5 for downstream heads.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 backbone with multi-scale feature extraction.

    Extracts features at 3 scales for multi-task heads:
    - P3: 1/8 resolution (80x80 for 640x640 input) - high res, good for small objects
    - P4: 1/16 resolution (40x40) - medium res
    - P5: 1/32 resolution (20x20) - low res, semantic features

    Args:
        model_name: EfficientNet variant ('efficientnet_b0', 'efficientnet_b1', etc.)
        pretrained: Whether to load ImageNet pretrained weights
        frozen: Whether to freeze backbone weights initially

    Example:
        >>> backbone = EfficientNetBackbone(pretrained=True)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> features = backbone(x)
        >>> print(features['p3'].shape)  # (1, 40, 80, 80)
        >>> print(features['p4'].shape)  # (1, 112, 40, 40)
        >>> print(features['p5'].shape)  # (1, 320, 20, 20)
    """

    # Feature indices for P3, P4, P5 extraction
    # EfficientNet-B0 feature_info.channels() = [16, 24, 40, 112, 320]
    # We use indices 2, 3, 4 for P3, P4, P5
    FEATURE_INDICES = [2, 3, 4]

    # Expected output channels for each scale (EfficientNet-B0)
    CHANNELS = {
        'efficientnet_b0': {'p3': 40, 'p4': 112, 'p5': 320},
        'efficientnet_b1': {'p3': 40, 'p4': 112, 'p5': 320},
        'efficientnet_b2': {'p3': 48, 'p4': 120, 'p5': 352},
    }

    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        frozen: bool = False,
    ):
        super().__init__()
        self.model_name = model_name

        # Create backbone with feature extraction at multiple scales
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.FEATURE_INDICES,
        )

        # Get actual channel counts from the model
        self.channels = self.backbone.feature_info.channels()
        self.out_channels = {
            'p3': self.channels[0],
            'p4': self.channels[1],
            'p5': self.channels[2],
        }

        # Print info for debugging
        print(f"[EfficientNetBackbone] Loaded {model_name}")
        print(f"  Feature channels: P3={self.out_channels['p3']}, "
              f"P4={self.out_channels['p4']}, P5={self.out_channels['p5']}")
        print(f"  Pretrained: {pretrained}, Frozen: {frozen}")

        if frozen:
            self.freeze()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features from input image.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with feature maps:
            - 'p3': (B, 40, H/8, W/8) - highest resolution
            - 'p4': (B, 112, H/16, W/16)
            - 'p5': (B, 320, H/32, W/32) - lowest resolution, most semantic
        """
        features = self.backbone(x)

        return {
            'p3': features[0],  # 1/8 scale
            'p4': features[1],  # 1/16 scale
            'p5': features[2],  # 1/32 scale
        }

    def freeze(self) -> None:
        """Freeze all backbone parameters (for transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[EfficientNetBackbone] Backbone frozen")

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[EfficientNetBackbone] Backbone unfrozen")

    def get_num_parameters(self) -> dict[str, int]:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }


def test_backbone():
    """Quick test to verify backbone shapes."""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    print("\n=== Testing EfficientNetBackbone ===\n")

    # Use CPU for testing to avoid MPS issues
    device = torch.device('cpu')

    backbone = EfficientNetBackbone(pretrained=True, frozen=False)
    backbone = backbone.to(device)
    backbone.eval()

    # Test with 640x640 input
    x = torch.randn(1, 3, 640, 640, device=device)

    with torch.no_grad():
        features = backbone(x)

    print(f"\nInput shape: {x.shape}")
    print(f"P3 shape: {features['p3'].shape} (expected: [1, 40, 80, 80])")
    print(f"P4 shape: {features['p4'].shape} (expected: [1, 112, 40, 40])")
    print(f"P5 shape: {features['p5'].shape} (expected: [1, 320, 20, 20])")

    # Verify shapes
    assert features['p3'].shape == (1, 40, 80, 80), f"P3 shape mismatch: {features['p3'].shape}"
    assert features['p4'].shape == (1, 112, 40, 40), f"P4 shape mismatch: {features['p4'].shape}"
    assert features['p5'].shape == (1, 320, 20, 20), f"P5 shape mismatch: {features['p5'].shape}"

    params = backbone.get_num_parameters()
    print(f"\nParameters: {params['total']:,} total, {params['trainable']:,} trainable")

    print("\n✅ All backbone tests passed!\n")


if __name__ == '__main__':
    test_backbone()
