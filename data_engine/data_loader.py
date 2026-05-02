"""PyTorch Dataset classes for BDD100K driving dataset.

Supports both original JSON format and YOLO format labels.
Tasks: detection, lane segmentation, drivable area segmentation.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class BDD100KYOLODataset(Dataset):
    """PyTorch Dataset for BDD100K in YOLO format.

    Expected directory structure:
        root_dir/
            train/
                images/
                    0000f77c-6257be58.jpg
                    ...
                labels/
                    0000f77c-6257be58.txt
                    ...
            val/
                images/
                labels/
            test/
                images/
                labels/

    YOLO label format (per line):
        class_id x_center y_center width height
        (all coordinates normalized 0-1)

    Args:
        root_dir: Path to BDD100K YOLO data directory
        split: 'train', 'val', or 'test'
        image_size: Target image size (height, width)
        transforms: Optional albumentations transforms
    """

    # BDD100K YOLO classes (from data.yaml)
    CLASSES = [
        "person", "rider", "car", "bus", "truck",
        "bike", "motor", "traffic light", "traffic sign", "train"
    ]
    NUM_CLASSES = len(CLASSES)

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

        # Setup paths for YOLO format
        self.images_dir = self.root_dir / split / "images"
        self.labels_dir = self.root_dir / split / "labels"

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        # Get list of image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"[BDD100K-YOLO] Loaded {len(self.image_files)} images from {split} split")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_yolo_labels(self, label_path: Path) -> tuple[list, list]:
        """Load labels from YOLO format txt file.

        Returns:
            boxes: List of [x1, y1, x2, y2] normalized coordinates
            class_ids: List of class indices
        """
        boxes = []
        class_ids = []

        if not label_path.exists():
            return boxes, class_ids

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Clip to valid range
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))

                # Only add valid boxes
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(class_id)

        return boxes, class_ids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample with image and detection labels.

        Returns:
            Dictionary with:
                - image: (3, H, W) tensor, normalized to [0, 1]
                - boxes: (N, 4) tensor of [x1, y1, x2, y2] normalized coords
                - labels: (N,) tensor of class indices
                - image_id: string identifier
                - image_path: full path to image
        """
        image_path = self.image_files[idx]
        image_name = image_path.stem

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize image to target size
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        image_np = np.array(image)

        # Load YOLO labels
        label_path = self.labels_dir / f"{image_name}.txt"
        boxes, class_ids = self._load_yolo_labels(label_path)

        # Apply transforms if provided
        if self.transforms is not None:
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes,
                labels=class_ids,
            )
            image_np = transformed["image"]
            boxes = transformed["bboxes"]
            class_ids = transformed["labels"]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

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

    def __init__(self, dataset: Dataset, num_samples: int):
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
    subset_size: Optional[int] = None,
    image_size: tuple[int, int] = (640, 640),
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for BDD100K YOLO format.

    Args:
        root_dir: Path to BDD100K YOLO data (containing train/val/test folders)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_size: If provided, use only this many samples
        image_size: Target image size (height, width)

    Returns:
        DataLoader instance
    """
    dataset = BDD100KYOLODataset(
        root_dir,
        split=split,
        image_size=image_size,
        **kwargs
    )

    if subset_size is not None:
        dataset = BDD100KSubset(dataset, subset_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=BDD100KYOLODataset.collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def test_dataloader():
    """Test the YOLO dataloader."""
    print("\n" + "="*60)
    print("Testing BDD100K YOLO DataLoader")
    print("="*60 + "\n")

    root_dir = Path(__file__).parent.parent / "data" / "bdd100k_yolo"

    if not root_dir.exists():
        print(f"Dataset not found at {root_dir}")
        return

    # Test loading
    dataloader = get_bdd100k_dataloader(
        root_dir,
        split="val",
        batch_size=4,
        num_workers=0,
        subset_size=8,
    )

    # Get one batch
    batch = next(iter(dataloader))

    print(f"Batch contents:")
    print(f"  images: {batch['images'].shape}")
    print(f"  num samples: {len(batch['boxes'])}")

    for i, (boxes, labels) in enumerate(zip(batch['boxes'], batch['labels'])):
        print(f"  sample {i}: {len(boxes)} boxes, labels: {labels.tolist()[:5]}...")

    # Verify shapes
    assert batch['images'].shape == (4, 3, 640, 640)
    assert len(batch['boxes']) == 4
    assert len(batch['labels']) == 4

    print("\n" + "="*60)
    print("✅ DataLoader test passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_dataloader()
