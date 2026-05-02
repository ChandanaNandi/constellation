"""Data augmentations for multi-task training.

Simple augmentations that work with both detection boxes and segmentation masks.
"""

import random
import numpy as np
import torch
from PIL import Image


def horizontal_flip(image: np.ndarray, boxes: list, seg_mask: np.ndarray) -> tuple:
    """Horizontal flip augmentation.

    Args:
        image: (H, W, 3) numpy array
        boxes: List of [x1, y1, x2, y2] normalized coordinates
        seg_mask: (H, W) numpy array

    Returns:
        Flipped image, boxes, and mask
    """
    image = np.fliplr(image).copy()
    seg_mask = np.fliplr(seg_mask).copy()

    # Flip box coordinates
    flipped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        flipped_boxes.append([1 - x2, y1, 1 - x1, y2])

    return image, flipped_boxes, seg_mask


def random_brightness(image: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
    """Random brightness adjustment."""
    delta = random.uniform(-max_delta, max_delta)
    image = image.astype(np.float32) + delta * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_contrast(image: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
    """Random contrast adjustment."""
    factor = random.uniform(lower, upper)
    mean = image.mean()
    image = (image.astype(np.float32) - mean) * factor + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_saturation(image: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
    """Random saturation adjustment."""
    factor = random.uniform(lower, upper)
    gray = np.mean(image, axis=2, keepdims=True)
    image = image.astype(np.float32)
    image = gray + factor * (image - gray)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class TrainAugmentation:
    """Training augmentation pipeline for multi-task learning."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        color_jitter: bool = True,
    ):
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter

    def __call__(
        self,
        image: np.ndarray,
        boxes: list,
        labels: list,
        seg_mask: np.ndarray,
    ) -> dict:
        """Apply augmentations.

        Args:
            image: (H, W, 3) numpy array
            boxes: List of [x1, y1, x2, y2] normalized coords
            labels: List of class indices
            seg_mask: (H, W) numpy array

        Returns:
            Dict with augmented image, boxes, labels, seg_mask
        """
        # Horizontal flip
        if random.random() < self.flip_prob:
            image, boxes, seg_mask = horizontal_flip(image, boxes, seg_mask)

        # Color jitter (only affects image, not masks)
        if self.color_jitter:
            if random.random() < 0.5:
                image = random_brightness(image)
            if random.random() < 0.5:
                image = random_contrast(image)
            if random.random() < 0.5:
                image = random_saturation(image)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'seg_mask': seg_mask,
        }


class ValAugmentation:
    """Validation augmentation (no augmentation, just formatting)."""

    def __call__(
        self,
        image: np.ndarray,
        boxes: list,
        labels: list,
        seg_mask: np.ndarray,
    ) -> dict:
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'seg_mask': seg_mask,
        }


def test_augmentations():
    """Test augmentation pipeline."""
    print("\n=== Testing Augmentations ===\n")

    # Create dummy data
    image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    boxes = [[0.1, 0.2, 0.3, 0.4], [0.6, 0.5, 0.8, 0.9]]
    labels = [0, 2]
    seg_mask = np.random.randint(0, 3, (512, 1024), dtype=np.uint8)

    # Test train augmentation
    aug = TrainAugmentation(flip_prob=1.0)  # Force flip
    result = aug(image, boxes, labels, seg_mask)

    print(f"Original box 0: {boxes[0]}")
    print(f"Flipped box 0:  {result['boxes'][0]}")
    print(f"Image shape: {result['image'].shape}")
    print(f"Mask shape:  {result['seg_mask'].shape}")

    # Verify flip
    orig_x1 = boxes[0][0]
    flip_x2 = result['boxes'][0][2]
    assert abs(orig_x1 - (1 - flip_x2)) < 1e-6, "Flip failed"

    print("\n✅ Augmentation tests passed!")


if __name__ == "__main__":
    test_augmentations()
