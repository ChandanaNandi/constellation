"""Multi-task inference and visualization for HydraNet v2.

Visualizes both detection (bounding boxes) and segmentation (drivable area overlay).

Usage:
    python inference_multitask.py --checkpoint checkpoints/best_v2.pt --num-images 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.hydranet_v2 import HydraNetV2
from data_engine.cityscapes_loader import CityscapesDataset


# Cityscapes detection classes
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
SEG_COLORS = {
    0: (0, 0, 0),        # background - black (transparent)
    1: (0, 200, 100),    # road - green
    2: (200, 100, 200),  # sidewalk - purple
}


def decode_detections(
    outputs: dict,
    score_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    image_size: tuple = (512, 1024),
) -> list[dict]:
    """Decode FCOS outputs to bounding boxes."""
    H, W = image_size
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
    keep = simple_nms(all_boxes, all_scores, nms_thresh)

    detections = []
    for idx in keep[:50]:  # Max 50 detections
        detections.append({
            'box': all_boxes[idx].tolist(),
            'class': all_labels[idx].item(),
            'score': all_scores[idx].item(),
        })

    return detections


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> list[int]:
    """Simple NMS."""
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


def visualize_multitask(
    image: Image.Image,
    detections: list[dict],
    seg_mask: np.ndarray,
    seg_alpha: float = 0.4,
) -> Image.Image:
    """Visualize detection boxes and segmentation overlay.

    Args:
        image: PIL Image
        detections: List of detection dicts with box, class, score
        seg_mask: (H, W) numpy array with class indices
        seg_alpha: Transparency for segmentation overlay

    Returns:
        Visualized image
    """
    # Create segmentation overlay
    W, H = image.size
    seg_overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))

    # Resize seg_mask to image size if needed
    if seg_mask.shape != (H, W):
        seg_mask = np.array(
            Image.fromarray(seg_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
        )

    # Color the segmentation mask
    seg_rgb = np.zeros((H, W, 4), dtype=np.uint8)
    for class_id, color in SEG_COLORS.items():
        if class_id == 0:
            continue  # Skip background
        mask = seg_mask == class_id
        seg_rgb[mask] = (*color, int(255 * seg_alpha))

    seg_overlay = Image.fromarray(seg_rgb, mode='RGBA')

    # Composite with original image
    image_rgba = image.convert('RGBA')
    result = Image.alpha_composite(image_rgba, seg_overlay)
    result = result.convert('RGB')

    # Draw detection boxes
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_idx = det['class']
        score = det['score']

        color = DET_COLORS[cls_idx % len(DET_COLORS)]
        label = f"{DET_CLASSES[cls_idx]} {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill=(0, 0, 0), font=font)

    return result


def run_inference(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    num_images: int = 10,
    score_thresh: float = 0.3,
    image_size: tuple = (512, 1024),
):
    """Run multi-task inference and save visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = HydraNetV2(
        num_det_classes=8,
        num_seg_classes=3,
        pretrained_backbone=False,
    )

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with random weights for testing...")

    model.eval()

    # Load dataset
    dataset = CityscapesDataset(data_dir, split='val', image_size=image_size)

    print(f"\nRunning inference on {num_images} images...")
    print(f"Score threshold: {score_thresh}")
    print(f"Output directory: {output_dir}\n")

    total_detections = 0

    for i in range(min(num_images, len(dataset))):
        sample = dataset[i]
        image_tensor = sample['image'].unsqueeze(0)
        image_path = sample['image_path']

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Decode detections
        detections = decode_detections(outputs, score_thresh=score_thresh, image_size=image_size)
        total_detections += len(detections)

        # Get segmentation mask
        seg_logits = outputs['segmentation'][0]  # (C, H, W)
        seg_mask = seg_logits.argmax(dim=0).numpy()  # (H, W)

        # Load original image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((image_size[1], image_size[0]))

        # Visualize
        vis_image = visualize_multitask(image, detections, seg_mask)

        # Save
        output_path = output_dir / f"multitask_{i:03d}.jpg"
        vis_image.save(output_path, quality=95)

        # Compute segmentation stats
        road_pct = 100 * (seg_mask == 1).sum() / seg_mask.size
        sidewalk_pct = 100 * (seg_mask == 2).sum() / seg_mask.size

        print(f"  [{i+1}/{num_images}] {Path(image_path).name}: "
              f"{len(detections)} detections, "
              f"road={road_pct:.1f}%, sidewalk={sidewalk_pct:.1f}%")

    print(f"\nDone! Total detections: {total_detections}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='HydraNet v2 Multi-Task Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_v2.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/cityscapes',
                        help='Path to Cityscapes dataset')
    parser.add_argument('--output-dir', type=str, default='output/multitask_vis',
                        help='Output directory for visualizations')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of images to process')
    parser.add_argument('--score-thresh', type=float, default=0.3,
                        help='Detection score threshold')
    parser.add_argument('--image-size', type=str, default='512,1024',
                        help='Image size as H,W')
    args = parser.parse_args()

    image_size = tuple(map(int, args.image_size.split(',')))

    run_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        score_thresh=args.score_thresh,
        image_size=image_size,
    )


if __name__ == '__main__':
    main()
