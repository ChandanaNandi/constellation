"""Task-specific heads for multi-task learning."""

from .detection_head import DetectionHead, DetectionOutput
from .segmentation_head import SegmentationHead
from .depth_head import DepthHead
from .traffic_light_head import TrafficLightHead

__all__ = [
    'DetectionHead',
    'DetectionOutput',
    'SegmentationHead',
    'DepthHead',
    'TrafficLightHead',
]
