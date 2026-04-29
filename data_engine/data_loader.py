"""PyTorch Dataset classes for BDD100K driving dataset.

Loads images and existing JSON labels for multi-task training.
Supports: detection, lane segmentation, drivable area segmentation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class BDD100KDataset(Dataset):
    """PyTorch Dataset for BDD100K with multi-task labels.

    Args:
        root_dir: Path to BDD100K data directory
        split: 'train', 'val', or 'test'
        image_size: Target image size (height, width)
        transforms: Optional albumentations transforms
    """

    # BDD100K detection classes
    CLASSES = [
        "pedestrian", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
    ]
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (640, 640),
        transforms: Any = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms

        # Setup paths
        self.images_dir = self.root_dir / "images" / "100k" / split
        self.labels_dir = self.root_dir / "labels"

        # Load detection labels
        det_labels_file = self.labels_dir / "det_20" / f"det_{split}.json"
        self.det_labels = self._load_detection_labels(det_labels_file)

        # Get list of image files
        self.image_files = sorted(self.images_dir.glob("*.jpg"))

        # Filter to only images that have labels
        labeled_names = set(self.det_labels.keys())
        self.image_files = [
            f for f in self.image_files if f.stem in labeled_names
        ]

        print(f"Loaded {len(self.image_files)} images from BDD100K {split} split")

    def _load_detection_labels(self, labels_file: Path) -> dict:
        """Load detection labels from BDD100K JSON format."""
        if not labels_file.exists():
            print(f"Warning: Labels file not found: {labels_file}")
            return {}

        with open(labels_file, "r") as f:
            data = json.load(f)

        # Index by image name (without extension)
        labels_by_name = {}
        for item in data:
            name = Path(item["name"]).stem
            labels_by_name[name] = item.get("labels", [])

        return labels_by_name

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample with image and multi-task labels.

        Returns:
            Dictionary with:
                - image: (3, H, W) tensor
                - boxes: (N, 4) tensor of [x1, y1, x2, y2] normalized coords
                - labels: (N,) tensor of class indices
                - image_id: string identifier
        """
        image_path = self.image_files[idx]
        image_name = image_path.stem

        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize image
        image = image.resize(self.image_size[::-1], Image.BILINEAR)
        image_np = np.array(image)

        # Get detection labels
        det_labels = self.det_labels.get(image_name, [])

        # Parse bounding boxes
        boxes = []
        class_labels = []

        for label in det_labels:
            if "box2d" not in label:
                continue

            category = label.get("category", "")
            if category not in self.CLASS_TO_IDX:
                continue

            box = label["box2d"]
            x1 = box["x1"] / orig_w
            y1 = box["y1"] / orig_h
            x2 = box["x2"] / orig_w
            y2 = box["y2"] / orig_h

            # Clip to valid range
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(1, x2), min(1, y2)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                class_labels.append(self.CLASS_TO_IDX[category])

        # Apply transforms if provided
        if self.transforms is not None:
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes,
                labels=class_labels,
            )
            image_np = transformed["image"]
            boxes = transformed["bboxes"]
            class_labels = transformed["labels"]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_name,
            "image_path": str(image_path),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for variable-length boxes."""
        images = torch.stack([item["image"] for item in batch])
        image_ids = [item["image_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        # Keep boxes/labels as lists (variable length per image)
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "image_ids": image_ids,
            "image_paths": image_paths,
        }


class BDD100KSubset(Dataset):
    """Wrapper to create a subset of BDD100K for faster iteration."""

    def __init__(self, dataset: BDD100KDataset, num_samples: int):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.indices = list(range(self.num_samples))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]


def get_bdd100k_dataloader(
    root_dir: str | Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    subset_size: int | None = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for BDD100K.

    Args:
        root_dir: Path to BDD100K data
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_size: If provided, use only this many samples

    Returns:
        DataLoader instance
    """
    dataset = BDD100KDataset(root_dir, split=split, **kwargs)

    if subset_size is not None:
        dataset = BDD100KSubset(dataset, subset_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=BDD100KDataset.collate_fn,
        pin_memory=True,
    )
