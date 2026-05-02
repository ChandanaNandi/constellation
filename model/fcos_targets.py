"""FCOS target assignment for anchor-free object detection.

Converts ground truth boxes to per-pixel targets for FCOS-style detection.
Each pixel in the feature map is assigned:
- A class label (or background)
- Regression targets (l, t, r, b) = distances to box edges
- Centerness score

Reference: https://arxiv.org/abs/1904.01355
"""

import torch
import torch.nn.functional as F
from typing import Optional


class FCOSTargetAssigner:
    """Assigns ground truth boxes to feature map locations for FCOS training.

    For each pixel location (x, y) in feature map:
    1. Find all GT boxes where (x, y) falls inside the box
    2. Assign the smallest box (if multiple matches)
    3. Compute regression targets: l, t, r, b = distances to edges
    4. Compute centerness = sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))

    Args:
        num_classes: Number of detection classes (excluding background)
        strides: Feature map strides for each scale (e.g., [8, 16, 32])
        size_ranges: Min/max object sizes per scale [(min, max), ...]
    """

    def __init__(
        self,
        num_classes: int = 10,
        strides: list[int] = [8, 16, 32],
        size_ranges: Optional[list[tuple[float, float]]] = None,
    ):
        self.num_classes = num_classes
        self.strides = strides

        # Default size ranges for multi-scale assignment
        # Adjusted for BDD100K where 81% of objects are < 64px
        # P3 (stride 8): smallest objects
        # P4 (stride 16): small-medium objects
        # P5 (stride 32): medium-large objects
        if size_ranges is None:
            self.size_ranges = [
                (0, 32),      # P3: very small objects (< 32px)
                (32, 64),     # P4: small objects (32-64px)
                (64, 1e8),    # P5: medium to large objects (64px+)
            ]
        else:
            self.size_ranges = size_ranges

    def assign_targets_single_image(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        image_size: tuple[int, int],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Assign targets for a single image across all scales.

        Args:
            boxes: (N, 4) tensor of [x1, y1, x2, y2] in normalized coords (0-1)
            labels: (N,) tensor of class indices
            image_size: (H, W) of input image

        Returns:
            Dictionary with targets for each scale:
            {
                'p3': {'cls': (H/8, W/8), 'reg': (4, H/8, W/8), 'centerness': (H/8, W/8)},
                'p4': {'cls': (H/16, W/16), 'reg': (4, H/16, W/16), 'centerness': (H/16, W/16)},
                'p5': {'cls': (H/32, W/32), 'reg': (4, H/32, W/32), 'centerness': (H/32, W/32)},
            }
        """
        H, W = image_size
        device = boxes.device if len(boxes) > 0 else torch.device('cpu')

        # Convert normalized coords to pixel coords
        if len(boxes) > 0:
            boxes_pixel = boxes.clone()
            boxes_pixel[:, [0, 2]] *= W
            boxes_pixel[:, [1, 3]] *= H
        else:
            boxes_pixel = torch.zeros((0, 4), device=device)

        targets = {}
        scale_names = ['p3', 'p4', 'p5']

        for scale_idx, (stride, scale_name) in enumerate(zip(self.strides, scale_names)):
            feat_h = H // stride
            feat_w = W // stride

            # Create grid of pixel locations (center of each cell)
            shifts_x = (torch.arange(feat_w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(feat_h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

            # Initialize targets
            cls_targets = torch.zeros(feat_h, feat_w, dtype=torch.long, device=device)
            reg_targets = torch.zeros(4, feat_h, feat_w, dtype=torch.float32, device=device)
            centerness_targets = torch.zeros(feat_h, feat_w, dtype=torch.float32, device=device)

            if len(boxes) == 0:
                targets[scale_name] = {
                    'cls': cls_targets,
                    'reg': reg_targets,
                    'centerness': centerness_targets,
                }
                continue

            # Get size range for this scale
            min_size, max_size = self.size_ranges[scale_idx]

            # For each GT box, assign to pixels inside it
            for box_idx in range(len(boxes_pixel)):
                x1, y1, x2, y2 = boxes_pixel[box_idx]
                cls = labels[box_idx].item()

                # Check if box size is appropriate for this scale
                box_size = max(x2 - x1, y2 - y1)
                if box_size < min_size or box_size >= max_size:
                    continue

                # Find pixels inside the box
                inside_x = (shift_x >= x1) & (shift_x <= x2)
                inside_y = (shift_y >= y1) & (shift_y <= y2)
                inside = inside_x & inside_y

                if not inside.any():
                    continue

                # Compute regression targets (l, t, r, b)
                l = shift_x - x1
                t = shift_y - y1
                r = x2 - shift_x
                b = y2 - shift_y

                # Compute centerness
                lr_min = torch.minimum(l, r)
                lr_max = torch.maximum(l, r)
                tb_min = torch.minimum(t, b)
                tb_max = torch.maximum(t, b)

                centerness = torch.sqrt(
                    (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
                )

                # Only update pixels inside the box
                # If a pixel already has a target, keep the smaller box (higher centerness)
                update_mask = inside & (
                    (cls_targets == 0) |  # Background
                    (centerness > centerness_targets)  # Better match
                )

                # Update targets
                cls_targets[update_mask] = cls + 1  # +1 because 0 is background

                reg_targets[0][update_mask] = l[update_mask]
                reg_targets[1][update_mask] = t[update_mask]
                reg_targets[2][update_mask] = r[update_mask]
                reg_targets[3][update_mask] = b[update_mask]

                centerness_targets[update_mask] = centerness[update_mask]

            targets[scale_name] = {
                'cls': cls_targets,
                'reg': reg_targets,
                'centerness': centerness_targets,
            }

        return targets

    def assign_targets_batch(
        self,
        boxes_batch: list[torch.Tensor],
        labels_batch: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Assign targets for a batch of images.

        Args:
            boxes_batch: List of (N_i, 4) tensors
            labels_batch: List of (N_i,) tensors
            image_size: (H, W) of input images

        Returns:
            Dictionary with batched targets for each scale
        """
        batch_size = len(boxes_batch)
        all_targets = [
            self.assign_targets_single_image(boxes, labels, image_size)
            for boxes, labels in zip(boxes_batch, labels_batch)
        ]

        # Stack into batched tensors
        result = {}
        for scale_name in ['p3', 'p4', 'p5']:
            result[scale_name] = {
                'cls': torch.stack([t[scale_name]['cls'] for t in all_targets]),
                'reg': torch.stack([t[scale_name]['reg'] for t in all_targets]),
                'centerness': torch.stack([t[scale_name]['centerness'] for t in all_targets]),
            }

        return result


def test_fcos_targets():
    """Test FCOS target assignment."""
    print("\n" + "="*60)
    print("Testing FCOS Target Assignment")
    print("="*60 + "\n")

    assigner = FCOSTargetAssigner(num_classes=10)

    # Create sample boxes (normalized coords)
    boxes = torch.tensor([
        [0.1, 0.1, 0.3, 0.3],   # Small box
        [0.4, 0.4, 0.8, 0.8],   # Large box
        [0.2, 0.5, 0.35, 0.7],  # Medium box
    ])
    labels = torch.tensor([2, 0, 5])  # car, person, bike

    # Assign targets
    targets = assigner.assign_targets_single_image(
        boxes, labels, image_size=(640, 640)
    )

    print("Target shapes:")
    for scale_name, scale_targets in targets.items():
        print(f"  {scale_name}:")
        print(f"    cls: {scale_targets['cls'].shape}")
        print(f"    reg: {scale_targets['reg'].shape}")
        print(f"    centerness: {scale_targets['centerness'].shape}")

    # Check some values
    print("\nPositive samples per scale:")
    for scale_name, scale_targets in targets.items():
        num_pos = (scale_targets['cls'] > 0).sum().item()
        print(f"  {scale_name}: {num_pos} positive pixels")

    # Test batch
    print("\nTesting batch assignment...")
    boxes_batch = [boxes, boxes[:1], torch.zeros((0, 4))]
    labels_batch = [labels, labels[:1], torch.zeros((0,), dtype=torch.long)]

    batch_targets = assigner.assign_targets_batch(
        boxes_batch, labels_batch, image_size=(640, 640)
    )

    print("Batch target shapes:")
    for scale_name, scale_targets in batch_targets.items():
        print(f"  {scale_name}: cls={scale_targets['cls'].shape}")

    print("\n" + "="*60)
    print("✅ FCOS target assignment test passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_fcos_targets()
