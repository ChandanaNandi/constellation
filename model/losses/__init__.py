"""Loss functions for multi-task learning."""

from .multi_task_loss import (
    MultiTaskLoss,
    DetectionLoss,
    SegmentationLoss,
)

__all__ = [
    'MultiTaskLoss',
    'DetectionLoss',
    'SegmentationLoss',
]
