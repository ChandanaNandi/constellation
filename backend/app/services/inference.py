"""Inference service for HydraNet V2 model."""

import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64

from model.hydranet_v2 import HydraNetV2


# Detection classes (Cityscapes)
DET_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

DET_COLORS = [
    (255, 0, 0),      # person - red
    (255, 128, 0),    # rider - orange
    (0, 255, 0),      # car - green
    (255, 255, 0),    # truck - yellow
    (0, 255, 255),    # bus - cyan
    (128, 128, 128),  # train - gray
    (255, 0, 255),    # motorcycle - magenta
    (128, 0, 255),    # bicycle - purple
]

# Segmentation classes
SEG_CLASSES = ["background", "road", "sidewalk"]
SEG_COLORS = {
    0: (0, 0, 0),        # background - black
    1: (0, 200, 100),    # road - green
    2: (200, 100, 200),  # sidewalk - purple
}


class InferenceService:
    """Singleton service for model inference."""

    _instance: Optional["InferenceService"] = None
    _model: Optional[HydraNetV2] = None
    _device: torch.device = torch.device("cpu")
    _image_size: tuple = (512, 1024)  # H, W
    _checkpoint_candidates: tuple[str, ...] = (
        "checkpoints/best_v2.pt",
        "checkpoints/latest_v2.pt",
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Optional[Path]:
        """Resolve checkpoint path across local/dev/container environments."""
        # Highest priority: explicit arg or env var
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.exists():
                return p
        env_checkpoint = os.getenv("MODEL_CHECKPOINT")
        if env_checkpoint:
            p = Path(env_checkpoint)
            if p.exists():
                return p

        # Default candidates under project roots
        cwd = Path.cwd()
        roots = [cwd, cwd / "backend", Path("/app")]
        for root in roots:
            for rel in self._checkpoint_candidates:
                p = root / rel
                if p.exists():
                    return p
        return None

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load the HydraNet V2 model."""
        if self._model is not None:
            return  # Already loaded

        resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        print(
            "Loading HydraNet V2 model"
            + (f" from {resolved_checkpoint}..." if resolved_checkpoint else " (no checkpoint found)...")
        )

        self._model = HydraNetV2(
            num_det_classes=8,
            num_seg_classes=3,
            pretrained_backbone=False,
        )

        if resolved_checkpoint is not None:
            checkpoint = torch.load(resolved_checkpoint, map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', '?')
            print(f"Loaded checkpoint (epoch {epoch})")
        else:
            print("Warning: No checkpoint file found in expected locations")
            print("Running with random weights...")

        self._model.to(self._device)
        self._model.eval()
        print("Model ready for inference!")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        # Resize to model input size
        image = image.resize((self._image_size[1], self._image_size[0]))

        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

        # Normalize (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor.unsqueeze(0)  # Add batch dim

    def decode_detections(
        self,
        outputs: dict,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5,
    ) -> list[dict]:
        """Decode FCOS outputs to bounding boxes."""
        H, W = self._image_size
        strides = {'p3': 8, 'p4': 16, 'p5': 32}
        det = outputs['detection']

        all_boxes = []
        all_scores = []
        all_labels = []

        for scale_name in ['p3', 'p4', 'p5']:
            scale_out = det[scale_name]
            stride = strides[scale_name]

            cls_logits = scale_out.cls_logits[0]
            bbox_pred = scale_out.bbox_pred[0]
            centerness = scale_out.centerness[0]

            cls_scores = torch.sigmoid(cls_logits)
            center_scores = torch.sigmoid(centerness)
            scores = cls_scores * center_scores

            C, fh, fw = cls_scores.shape
            max_scores, max_classes = scores.max(dim=0)

            mask = max_scores > score_thresh
            if mask.sum() == 0:
                continue

            ys, xs = torch.where(mask)
            bbox_pred_pos = F.relu(bbox_pred[:, mask])

            cx = (xs.float() + 0.5) * stride
            cy = (ys.float() + 0.5) * stride

            x1 = (cx - bbox_pred_pos[0]).clamp(0, W)
            y1 = (cy - bbox_pred_pos[1]).clamp(0, H)
            x2 = (cx + bbox_pred_pos[2]).clamp(0, W)
            y2 = (cy + bbox_pred_pos[3]).clamp(0, H)

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            all_boxes.append(boxes)
            all_scores.append(max_scores[mask])
            all_labels.append(max_classes[mask])

        if len(all_boxes) == 0:
            return []

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Simple NMS
        keep = self._simple_nms(all_boxes, all_scores, nms_thresh)

        detections = []
        for idx in keep[:50]:  # Max 50 detections
            detections.append({
                'box': all_boxes[idx].tolist(),
                'class_id': all_labels[idx].item(),
                'class_name': DET_CLASSES[all_labels[idx].item()],
                'confidence': round(all_scores[idx].item(), 3),
            })

        return detections

    def _simple_nms(self, boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> list[int]:
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i.item())

            if len(order) == 1:
                break

            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            mask = iou < thresh
            order = order[1:][mask]

        return keep

    def get_segmentation(self, outputs: dict) -> dict:
        """Get segmentation results from model outputs."""
        seg_logits = outputs['segmentation'][0]  # (C, H, W)
        seg_mask = seg_logits.argmax(dim=0).numpy()  # (H, W)

        # Calculate percentages
        total_pixels = seg_mask.size
        road_pixels = (seg_mask == 1).sum()
        sidewalk_pixels = (seg_mask == 2).sum()

        return {
            'mask': seg_mask.tolist(),
            'road_percentage': round(100 * road_pixels / total_pixels, 1),
            'sidewalk_percentage': round(100 * sidewalk_pixels / total_pixels, 1),
        }

    def predict(
        self,
        image: Image.Image,
        score_threshold: float = 0.3,
    ) -> dict:
        """Run full inference on an image.

        Args:
            image: PIL Image
            score_threshold: Detection confidence threshold

        Returns:
            Dict with detections and segmentation results
        """
        if self._model is None:
            self.load_model()

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            outputs = self._model(input_tensor)

        # Decode results
        detections = self.decode_detections(outputs, score_thresh=score_threshold)
        segmentation = self.get_segmentation(outputs)

        return {
            'detections': detections,
            'num_detections': len(detections),
            'segmentation': {
                'road_percentage': segmentation['road_percentage'],
                'sidewalk_percentage': segmentation['sidewalk_percentage'],
            },
            'image_size': {
                'width': self._image_size[1],
                'height': self._image_size[0],
            },
        }

    def predict_from_path(self, image_path: str, score_threshold: float = 0.3) -> dict:
        """Run inference on an image from file path."""
        image = Image.open(image_path).convert('RGB')
        return self.predict(image, score_threshold)


# Global instance
inference_service = InferenceService()
