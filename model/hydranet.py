"""HydraNet: Multi-task vision model for autonomous driving.

Architecture inspired by Tesla's HydraNet as presented at AI Day 2021/2022.
Single shared backbone with multiple task-specific heads.
"""

import torch
import torch.nn as nn

# Placeholder - will be implemented in Phase 2


class HydraNet(nn.Module):
    """Multi-task vision model with shared backbone and multiple heads.

    Tasks:
        - Object detection (FCOS-style anchor-free)
        - Lane segmentation
        - Drivable area segmentation
        - Depth estimation
        - Traffic light classification

    Args:
        backbone: Name of backbone architecture ('efficientnet_b0', 'resnet50')
        num_classes: Number of detection classes
        pretrained: Whether to use pretrained backbone weights
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # TODO: Implement in Phase 2
        # self.backbone = ...
        # self.detection_head = ...
        # self.lane_head = ...
        # self.drivable_head = ...
        # self.depth_head = ...
        # self.traffic_light_head = ...

        # Placeholder layer for testing
        self.placeholder = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all heads.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with outputs for each task
        """
        # TODO: Implement in Phase 2
        batch_size = x.shape[0]

        return {
            "detection": torch.zeros(batch_size, 100, 5 + self.num_classes),
            "lane": torch.zeros(batch_size, 1, x.shape[2], x.shape[3]),
            "drivable": torch.zeros(batch_size, 1, x.shape[2], x.shape[3]),
            "depth": torch.zeros(batch_size, 1, x.shape[2], x.shape[3]),
            "traffic_light": torch.zeros(batch_size, 4),
        }


def get_device() -> torch.device:
    """Get the best available device for training/inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
