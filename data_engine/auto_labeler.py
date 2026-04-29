"""Auto-labeling pipeline using YOLOv8 and MobileSAM.

Designed for demo use - auto-labels individual images on demand.
Uses MobileSAM for fast segmentation (~10x faster than full SAM).
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Lazy imports to avoid loading heavy models at module import
_yolo_model = None
_sam_model = None
_sam_predictor = None


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yolo_model(model_name: str = "yolov8x.pt"):
    """Load YOLOv8 model (lazy loading)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_name)
        print(f"Loaded YOLO model: {model_name}")
    return _yolo_model


def load_sam_model(checkpoint_path: str | None = None):
    """Load MobileSAM model (lazy loading).

    Falls back to downloading if checkpoint not provided.
    """
    global _sam_model, _sam_predictor

    if _sam_predictor is not None:
        return _sam_predictor

    device = get_device()

    try:
        # Try MobileSAM first (faster, smaller)
        from mobile_sam import sam_model_registry, SamPredictor

        model_type = "vit_t"  # MobileSAM uses tiny ViT

        if checkpoint_path is None:
            # Download default checkpoint
            checkpoint_path = "mobile_sam.pt"
            if not os.path.exists(checkpoint_path):
                import urllib.request
                url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                print(f"Downloading MobileSAM checkpoint...")
                urllib.request.urlretrieve(url, checkpoint_path)

        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        _sam_model.to(device)
        _sam_predictor = SamPredictor(_sam_model)
        print(f"Loaded MobileSAM on {device}")

    except ImportError:
        # Fall back to standard SAM
        print("MobileSAM not found, falling back to segment-anything")
        from segment_anything import sam_model_registry, SamPredictor

        model_type = "vit_b"  # Use base model for speed

        if checkpoint_path is None:
            checkpoint_path = "sam_vit_b.pth"
            if not os.path.exists(checkpoint_path):
                print("SAM checkpoint not found. Please download from:")
                print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                raise FileNotFoundError("SAM checkpoint required")

        _sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        _sam_model.to(device)
        _sam_predictor = SamPredictor(_sam_model)
        print(f"Loaded SAM ({model_type}) on {device}")

    return _sam_predictor


class AutoLabeler:
    """Auto-labeling pipeline combining YOLOv8 detection and SAM segmentation.

    Usage:
        labeler = AutoLabeler()
        results = labeler.label_image("path/to/image.jpg")

    Results contain:
        - boxes: List of [x1, y1, x2, y2, confidence, class_id]
        - masks: List of binary segmentation masks
        - class_names: List of detected class names
    """

    # COCO classes that are relevant for driving scenes
    DRIVING_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        11: "stop sign",
    }

    def __init__(
        self,
        yolo_model: str = "yolov8x.pt",
        sam_checkpoint: str | None = None,
        conf_threshold: float = 0.25,
        use_sam: bool = True,
    ):
        """Initialize auto-labeler.

        Args:
            yolo_model: YOLO model name or path
            sam_checkpoint: Path to SAM/MobileSAM checkpoint
            conf_threshold: Confidence threshold for detections
            use_sam: Whether to run SAM for segmentation masks
        """
        self.yolo_model_name = yolo_model
        self.sam_checkpoint = sam_checkpoint
        self.conf_threshold = conf_threshold
        self.use_sam = use_sam

        self._yolo = None
        self._sam = None

    @property
    def yolo(self):
        if self._yolo is None:
            self._yolo = load_yolo_model(self.yolo_model_name)
        return self._yolo

    @property
    def sam(self):
        if self._sam is None and self.use_sam:
            self._sam = load_sam_model(self.sam_checkpoint)
        return self._sam

    def detect_objects(self, image: np.ndarray | str | Path) -> dict:
        """Run YOLO detection on an image.

        Args:
            image: Image array (RGB) or path to image

        Returns:
            Dictionary with boxes, confidences, and class_ids
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        results = self.yolo(image, conf=self.conf_threshold, verbose=False)[0]

        boxes = []
        confidences = []
        class_ids = []
        class_names = []

        for box in results.boxes:
            cls_id = int(box.cls[0])

            # Filter to driving-relevant classes
            if cls_id not in self.DRIVING_CLASSES:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            boxes.append(xyxy.tolist())
            confidences.append(conf)
            class_ids.append(cls_id)
            class_names.append(self.DRIVING_CLASSES[cls_id])

        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "class_names": class_names,
        }

    def segment_boxes(
        self,
        image: np.ndarray,
        boxes: list[list[float]],
    ) -> list[np.ndarray]:
        """Generate segmentation masks for given boxes using SAM.

        Args:
            image: RGB image array
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary mask arrays (H, W)
        """
        if not self.use_sam or len(boxes) == 0:
            return []

        sam_predictor = self.sam
        sam_predictor.set_image(image)

        masks = []
        for box in boxes:
            box_array = np.array(box)
            mask, _, _ = sam_predictor.predict(
                box=box_array,
                multimask_output=False,
            )
            masks.append(mask[0])  # Take first mask

        return masks

    def label_image(self, image: np.ndarray | str | Path) -> dict:
        """Full auto-labeling pipeline: detection + segmentation.

        Args:
            image: Image array (RGB) or path to image

        Returns:
            Dictionary with:
                - boxes: [[x1, y1, x2, y2], ...]
                - confidences: [float, ...]
                - class_ids: [int, ...]
                - class_names: [str, ...]
                - masks: [np.ndarray, ...] (if use_sam=True)
                - image_shape: (H, W, C)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        # Run detection
        det_results = self.detect_objects(image)

        # Run segmentation if enabled
        masks = []
        if self.use_sam and len(det_results["boxes"]) > 0:
            masks = self.segment_boxes(image, det_results["boxes"])

        return {
            **det_results,
            "masks": masks,
            "image_shape": image.shape,
        }

    def to_coco_format(
        self,
        results: dict,
        image_id: int | str,
        start_annotation_id: int = 1,
    ) -> dict:
        """Convert auto-label results to COCO format.

        Args:
            results: Output from label_image()
            image_id: Unique image identifier
            start_annotation_id: Starting ID for annotations

        Returns:
            Dictionary with COCO-formatted annotations
        """
        from pycocotools import mask as mask_utils

        h, w = results["image_shape"][:2]

        annotations = []
        for i, (box, conf, cls_id, cls_name) in enumerate(zip(
            results["boxes"],
            results["confidences"],
            results["class_ids"],
            results["class_names"],
        )):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height

            annotation = {
                "id": start_annotation_id + i,
                "image_id": image_id,
                "category_id": cls_id,
                "category_name": cls_name,
                "bbox": [x1, y1, width, height],  # COCO format: [x, y, w, h]
                "area": area,
                "iscrowd": 0,
                "confidence": conf,
            }

            # Add segmentation if available
            if i < len(results["masks"]):
                mask = results["masks"][i]
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                annotation["segmentation"] = rle

            annotations.append(annotation)

        return {
            "image": {
                "id": image_id,
                "width": w,
                "height": h,
            },
            "annotations": annotations,
        }


# Convenience function for quick labeling
def auto_label(
    image_path: str | Path,
    use_sam: bool = True,
    conf_threshold: float = 0.25,
) -> dict:
    """Quick auto-label function for a single image.

    Args:
        image_path: Path to image
        use_sam: Whether to generate segmentation masks
        conf_threshold: Detection confidence threshold

    Returns:
        Auto-label results dictionary
    """
    labeler = AutoLabeler(use_sam=use_sam, conf_threshold=conf_threshold)
    return labeler.label_image(image_path)
