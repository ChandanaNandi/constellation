"""Inference and visualization for HydraNet v1.

Runs detection on images and visualizes predictions.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.hydranet_v1 import HydraNetV1
from data_engine.data_loader import BDD100KYOLODataset


# BDD100K class names and colors
CLASS_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

CLASS_COLORS = [
    (255, 0, 0),      # person - red
    (255, 128, 0),    # rider - orange
    (0, 255, 0),      # car - green
    (0, 255, 255),    # bus - cyan
    (255, 255, 0),    # truck - yellow
    (128, 0, 255),    # bike - purple
    (255, 0, 255),    # motor - magenta
    (255, 0, 128),    # traffic light - pink
    (0, 128, 255),    # traffic sign - blue
    (128, 128, 128),  # train - gray
]


def decode_fcos_outputs(
    outputs: dict,
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    image_size: tuple = (640, 640),
) -> list[dict]:
    """Decode FCOS outputs to bounding boxes.

    Args:
        outputs: Model outputs with detection dict containing p3, p4, p5
        score_thresh: Minimum score to keep
        nms_thresh: NMS IoU threshold
        image_size: (H, W) of input image

    Returns:
        List of detections: [{'box': [x1,y1,x2,y2], 'class': int, 'score': float}, ...]
    """
    H, W = image_size
    strides = {'p3': 8, 'p4': 16, 'p5': 32}

    all_boxes = []
    all_scores = []
    all_classes = []

    det = outputs['detection']

    for scale_name in ['p3', 'p4', 'p5']:
        scale_out = det[scale_name]
        stride = strides[scale_name]

        # Get predictions
        cls_logits = scale_out.cls_logits[0]  # (C, H, W)
        bbox_pred = scale_out.bbox_pred[0]     # (4, H, W)
        centerness = scale_out.centerness[0]   # (1, H, W)

        # Compute scores = sigmoid(cls) * sigmoid(centerness)
        cls_scores = torch.sigmoid(cls_logits)  # (C, H, W)
        center_scores = torch.sigmoid(centerness)  # (1, H, W)
        scores = cls_scores * center_scores  # (C, H, W)

        # Get feature map size
        _, fh, fw = cls_scores.shape

        # Create grid of locations
        shifts_x = (torch.arange(fw, device=cls_scores.device) + 0.5) * stride
        shifts_y = (torch.arange(fh, device=cls_scores.device) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        # Decode boxes (l, t, r, b) -> (x1, y1, x2, y2)
        bbox_pred = F.relu(bbox_pred)  # Ensure positive
        l, t, r, b = bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]

        x1 = shift_x - l
        y1 = shift_y - t
        x2 = shift_x + r
        y2 = shift_y + b

        # Clip to image bounds
        x1 = x1.clamp(0, W)
        y1 = y1.clamp(0, H)
        x2 = x2.clamp(0, W)
        y2 = y2.clamp(0, H)

        # Stack boxes
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (H, W, 4)

        # For each class, get detections above threshold
        for cls_idx in range(len(CLASS_NAMES)):
            cls_score = scores[cls_idx]  # (H, W)
            mask = cls_score > score_thresh

            if mask.sum() == 0:
                continue

            cls_boxes = boxes[mask]  # (N, 4)
            cls_scores_filtered = cls_score[mask]  # (N,)

            all_boxes.append(cls_boxes)
            all_scores.append(cls_scores_filtered)
            all_classes.append(torch.full((len(cls_boxes),), cls_idx, device=cls_boxes.device))

    if len(all_boxes) == 0:
        return []

    # Concatenate all detections
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_classes = torch.cat(all_classes, dim=0)

    # Apply NMS per class
    keep_indices = []
    for cls_idx in range(len(CLASS_NAMES)):
        cls_mask = all_classes == cls_idx
        if cls_mask.sum() == 0:
            continue

        cls_boxes = all_boxes[cls_mask]
        cls_scores_nms = all_scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]

        # Simple NMS
        keep = nms(cls_boxes, cls_scores_nms, nms_thresh)
        keep_indices.extend(cls_indices[keep].tolist())

    # Build result
    detections = []
    for idx in keep_indices:
        detections.append({
            'box': all_boxes[idx].tolist(),
            'class': all_classes[idx].item(),
            'score': all_scores[idx].item(),
        })

    # Sort by score
    detections.sort(key=lambda x: x['score'], reverse=True)

    return detections


def nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> list[int]:
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with rest
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # Keep boxes with IoU < threshold
        mask = iou < thresh
        order = order[1:][mask]

    return keep


def visualize_detections(
    image: Image.Image,
    detections: list[dict],
    gt_boxes: list = None,
    gt_labels: list = None,
) -> Image.Image:
    """Draw detections on image.

    Args:
        image: PIL Image
        detections: List of detection dicts
        gt_boxes: Optional ground truth boxes (normalized)
        gt_labels: Optional ground truth labels

    Returns:
        Image with drawn boxes
    """
    draw = ImageDraw.Draw(image)
    W, H = image.size

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    # Draw ground truth boxes in white/dashed if provided
    if gt_boxes is not None and gt_labels is not None:
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            # Convert from normalized to pixel coords
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H

            # Draw GT box with dashed line (white)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=1)
            draw.text((x1, y1 - 15), f"GT:{CLASS_NAMES[label]}", fill=(255, 255, 255), font=font)

    # Draw predictions
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_idx = det['class']
        score = det['score']

        color = CLASS_COLORS[cls_idx]
        label = f"{CLASS_NAMES[cls_idx]} {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill=(0, 0, 0), font=font)

    return image


def run_inference(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    num_images: int = 10,
    score_thresh: float = 0.3,
    show_gt: bool = True,
):
    """Run inference on dataset images and save visualizations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    device = torch.device('cpu')  # Use CPU for inference

    model = HydraNetV1(num_classes=10, pretrained_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f})")

    # Load dataset
    dataset = BDD100KYOLODataset(data_dir, split='val')

    print(f"\nRunning inference on {num_images} images...")
    print(f"Score threshold: {score_thresh}")
    print(f"Output directory: {output_dir}\n")

    total_detections = 0

    for i in range(min(num_images, len(dataset))):
        sample = dataset[i]
        image_tensor = sample['image'].unsqueeze(0)  # (1, 3, H, W)
        image_path = sample['image_path']

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Decode detections
        detections = decode_fcos_outputs(outputs, score_thresh=score_thresh)
        total_detections += len(detections)

        # Load original image for visualization
        image = Image.open(image_path).convert('RGB')
        image = image.resize((640, 640))

        # Get ground truth
        gt_boxes = sample['boxes'].tolist() if show_gt else None
        gt_labels = sample['labels'].tolist() if show_gt else None

        # Visualize
        vis_image = visualize_detections(image, detections, gt_boxes, gt_labels)

        # Save
        output_path = output_dir / f"detection_{i:03d}.jpg"
        vis_image.save(output_path)

        print(f"  [{i+1}/{num_images}] {Path(image_path).name}: {len(detections)} detections")

    print(f"\nDone! Total detections: {total_detections}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='HydraNet v1 Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/bdd100k_yolo',
                        help='Path to BDD100K YOLO dataset')
    parser.add_argument('--output-dir', type=str, default='output/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of images to process')
    parser.add_argument('--score-thresh', type=float, default=0.3,
                        help='Detection score threshold')
    parser.add_argument('--no-gt', action='store_true',
                        help='Do not show ground truth boxes')
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        score_thresh=args.score_thresh,
        show_gt=not args.no_gt,
    )


if __name__ == '__main__':
    main()
