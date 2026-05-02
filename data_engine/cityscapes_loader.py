"""PyTorch Dataset for Cityscapes with multi-task support.

Supports: detection (bounding boxes), semantic segmentation (drivable area).
Phase 4: Multi-task learning with shared backbone.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CityscapesDataset(Dataset):
    """PyTorch Dataset for Cityscapes with detection + segmentation.

    Expected directory structure (after extracting downloads):
        root_dir/
            leftImg8bit/
                train/
                    aachen/
                        aachen_000000_000019_leftImg8bit.png
                        ...
                    bochum/
                    ...
                val/
                    ...
            gtFine/
                train/
                    aachen/
                        aachen_000000_000019_gtFine_labelIds.png
                        aachen_000000_000019_gtFine_instanceIds.png
                        ...
                val/
                    ...

    Args:
        root_dir: Path to Cityscapes data directory
        split: 'train' or 'val'
        image_size: Target image size (height, width)
        transforms: Optional albumentations transforms
    """

    # Cityscapes class IDs we care about
    # Full list: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    # Detection classes (things with instances)
    DETECTION_CLASSES = {
        24: "person",
        25: "rider",
        26: "car",
        27: "truck",
        28: "bus",
        31: "train",
        32: "motorcycle",
        33: "bicycle",
    }

    # Map Cityscapes IDs to our detection class indices (0-7)
    DETECTION_ID_MAP = {
        24: 0,  # person
        25: 1,  # rider
        26: 2,  # car
        27: 3,  # truck
        28: 4,  # bus
        31: 5,  # train
        32: 6,  # motorcycle
        33: 7,  # bicycle
    }
    NUM_DETECTION_CLASSES = 8

    # Segmentation classes for drivable area
    # 0 = background/ignore, 1 = road, 2 = sidewalk
    SEGMENTATION_CLASSES = {
        7: 1,   # road -> drivable
        8: 2,   # sidewalk -> alternative
        # everything else -> 0 (background)
    }
    NUM_SEG_CLASSES = 3  # background, road, sidewalk

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (512, 1024),  # H, W (keep aspect ratio)
        transforms: Any = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms

        # Setup paths
        self.images_dir = self.root_dir / "leftImg8bit" / split
        self.labels_dir = self.root_dir / "gtFine" / split

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        # Collect all image files (recursive through city folders)
        self.image_files = sorted(list(self.images_dir.glob("*/*_leftImg8bit.png")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"[Cityscapes] Loaded {len(self.image_files)} images from {split} split")

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_label_paths(self, image_path: Path) -> tuple[Path, Path]:
        """Get corresponding label file paths for an image.

        Image: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        Labels: gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
                gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png
        """
        # Extract city and file stem
        city = image_path.parent.name
        stem = image_path.stem.replace("_leftImg8bit", "")

        label_dir = self.labels_dir / city
        labelIds_path = label_dir / f"{stem}_gtFine_labelIds.png"
        instanceIds_path = label_dir / f"{stem}_gtFine_instanceIds.png"

        return labelIds_path, instanceIds_path

    def _extract_boxes_from_instances(
        self,
        instance_mask: np.ndarray,
        label_mask: np.ndarray,
        orig_size: tuple[int, int],
    ) -> tuple[list, list]:
        """Extract bounding boxes from instance segmentation mask.

        Cityscapes instance IDs encode: class_id * 1000 + instance_id
        For class_id >= 24 (person and above), instances are labeled.

        Args:
            instance_mask: (H, W) array of instance IDs
            label_mask: (H, W) array of class IDs
            orig_size: Original image size (H, W) for normalization

        Returns:
            boxes: List of [x1, y1, x2, y2] normalized coordinates
            class_ids: List of our detection class indices
        """
        boxes = []
        class_ids = []

        orig_h, orig_w = orig_size

        # Get unique instance IDs (excluding 0 = background)
        unique_instances = np.unique(instance_mask)

        for inst_id in unique_instances:
            if inst_id == 0:
                continue

            # Decode class from instance ID
            # instance_id = class_id * 1000 + instance_number
            if inst_id >= 1000:
                class_id = inst_id // 1000
            else:
                # For non-instance classes, use the label directly
                continue

            # Check if this is a detection class we care about
            if class_id not in self.DETECTION_ID_MAP:
                continue

            # Get mask for this instance
            mask = (instance_mask == inst_id)

            if mask.sum() < 10:  # Skip tiny instances
                continue

            # Get bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any() or not cols.any():
                continue

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Normalize coordinates
            x1_norm = x1 / orig_w
            y1_norm = y1 / orig_h
            x2_norm = (x2 + 1) / orig_w
            y2_norm = (y2 + 1) / orig_h

            # Clip to valid range
            x1_norm = max(0.0, min(1.0, x1_norm))
            y1_norm = max(0.0, min(1.0, y1_norm))
            x2_norm = max(0.0, min(1.0, x2_norm))
            y2_norm = max(0.0, min(1.0, y2_norm))

            # Only add valid boxes
            if x2_norm > x1_norm and y2_norm > y1_norm:
                boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                class_ids.append(self.DETECTION_ID_MAP[class_id])

        return boxes, class_ids

    def _create_seg_mask(self, label_mask: np.ndarray) -> np.ndarray:
        """Create simplified segmentation mask for drivable area.

        Args:
            label_mask: (H, W) array of Cityscapes class IDs

        Returns:
            seg_mask: (H, W) array with 0=background, 1=road, 2=sidewalk
        """
        seg_mask = np.zeros_like(label_mask, dtype=np.uint8)

        for cityscapes_id, our_id in self.SEGMENTATION_CLASSES.items():
            seg_mask[label_mask == cityscapes_id] = our_id

        return seg_mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample with image, detection boxes, and segmentation mask.

        Returns:
            Dictionary with:
                - image: (3, H, W) tensor, normalized to [0, 1]
                - boxes: (N, 4) tensor of [x1, y1, x2, y2] normalized coords
                - labels: (N,) tensor of detection class indices
                - seg_mask: (H, W) tensor of segmentation class indices
                - image_id: string identifier
                - image_path: full path to image
        """
        image_path = self.image_files[idx]
        image_name = image_path.stem.replace("_leftImg8bit", "")

        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Get label paths
        labelIds_path, instanceIds_path = self._get_label_paths(image_path)

        # Load labels
        if labelIds_path.exists():
            label_mask = np.array(Image.open(labelIds_path))
        else:
            label_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if instanceIds_path.exists():
            instance_mask = np.array(Image.open(instanceIds_path).convert("I"))  # 32-bit
        else:
            instance_mask = np.zeros((orig_h, orig_w), dtype=np.int32)

        # Extract detection boxes from instances
        boxes, class_ids = self._extract_boxes_from_instances(
            instance_mask, label_mask, (orig_h, orig_w)
        )

        # Create segmentation mask
        seg_mask = self._create_seg_mask(label_mask)

        # Resize everything to target size
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        seg_mask = np.array(
            Image.fromarray(seg_mask).resize(
                (self.image_size[1], self.image_size[0]),
                Image.NEAREST  # Use nearest for masks to preserve class IDs
            )
        )

        image_np = np.array(image)

        # Apply transforms if provided
        if self.transforms is not None:
            transformed = self.transforms(
                image=image_np,
                mask=seg_mask,
                bboxes=boxes,
                labels=class_ids,
            )
            image_np = transformed["image"]
            seg_mask = transformed["mask"]
            boxes = transformed["bboxes"]
            class_ids = transformed["labels"]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        seg_tensor = torch.from_numpy(seg_mask).long()

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_ids, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "seg_mask": seg_tensor,
            "image_id": image_name,
            "image_path": str(image_path),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for variable-length boxes."""
        images = torch.stack([item["image"] for item in batch])
        seg_masks = torch.stack([item["seg_mask"] for item in batch])
        image_ids = [item["image_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        # Keep boxes/labels as lists (variable length per image)
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "seg_masks": seg_masks,
            "image_ids": image_ids,
            "image_paths": image_paths,
        }


def get_cityscapes_dataloader(
    root_dir: str | Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: tuple[int, int] = (512, 1024),
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for Cityscapes.

    Args:
        root_dir: Path to Cityscapes data (containing leftImg8bit/ and gtFine/)
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (height, width)

    Returns:
        DataLoader instance
    """
    dataset = CityscapesDataset(
        root_dir,
        split=split,
        image_size=image_size,
        **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=CityscapesDataset.collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def test_dataloader():
    """Test the Cityscapes dataloader."""
    print("\n" + "="*60)
    print("Testing Cityscapes DataLoader")
    print("="*60 + "\n")

    root_dir = Path(__file__).parent.parent / "data" / "cityscapes"

    if not root_dir.exists():
        print(f"Dataset not found at {root_dir}")
        print("Download from: https://www.cityscapes-dataset.com/")
        return False

    # Test loading
    try:
        dataloader = get_cityscapes_dataloader(
            root_dir,
            split="val",
            batch_size=2,
            num_workers=0,
            image_size=(512, 1024),
        )
    except ValueError as e:
        print(f"Error: {e}")
        return False

    # Get one batch
    batch = next(iter(dataloader))

    print(f"Batch contents:")
    print(f"  images: {batch['images'].shape}")
    print(f"  seg_masks: {batch['seg_masks'].shape}")
    print(f"  num samples: {len(batch['boxes'])}")

    for i, (boxes, labels) in enumerate(zip(batch['boxes'], batch['labels'])):
        print(f"  sample {i}: {len(boxes)} boxes, labels: {labels.tolist()[:5]}...")

    # Check segmentation mask values
    seg_unique = torch.unique(batch['seg_masks'])
    print(f"  seg_mask unique values: {seg_unique.tolist()}")

    print("\n" + "="*60)
    print("Cityscapes DataLoader test passed!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    test_dataloader()
