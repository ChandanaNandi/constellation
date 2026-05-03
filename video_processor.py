"""Video Processor for Constellation X.

Processes sequential frames through HydraNet V2 and generates annotated video.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.hydranet_v2 import HydraNetV2
from decision_engine import DecisionEngine, Decision, Action


# Colors for visualization
ACTION_COLORS = {
    Action.MAINTAIN: (0, 200, 0),    # Green
    Action.SLOW: (0, 200, 255),      # Orange
    Action.STOP: (0, 0, 255),        # Red
    Action.CAUTION: (0, 255, 255),   # Yellow
}

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

SEG_COLORS = {
    1: (0, 200, 100),    # road - green
    2: (200, 100, 200),  # sidewalk - purple
}


class VideoProcessor:
    """Process video frames through HydraNet V2 with decision engine."""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (512, 1024),
        device: str = "cpu",
        score_thresh: float = 0.25,
    ):
        self.image_size = image_size
        self.device = torch.device(device)
        self.score_thresh = score_thresh

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = HydraNetV2(
            num_det_classes=8,
            num_seg_classes=3,
            pretrained_backbone=False,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

        # Initialize decision engine
        self.decision_engine = DecisionEngine(image_size=image_size)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To tensor and normalize
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def decode_detections(self, outputs: dict) -> List[dict]:
        """Decode FCOS outputs to bounding boxes."""
        H, W = self.image_size
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

            mask = max_scores > self.score_thresh
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
        keep = self._nms(all_boxes, all_scores, 0.5)

        detections = []
        for idx in keep[:50]:
            detections.append({
                'box': all_boxes[idx].cpu().tolist(),
                'class': all_labels[idx].cpu().item(),
                'score': all_scores[idx].cpu().item(),
            })

        return detections

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> List[int]:
        """Non-maximum suppression."""
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

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Decision]:
        """Process single frame and return annotated frame with decision."""
        # Preprocess
        input_tensor = self.preprocess(frame)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Decode outputs
        detections_raw = self.decode_detections(outputs)
        seg_logits = outputs['segmentation'][0]
        seg_mask = seg_logits.argmax(dim=0).cpu().numpy()

        # Convert to Decision Engine format
        detections = self.decision_engine.detections_from_model_output(detections_raw)

        # Get decision
        decision = self.decision_engine.analyze(detections, seg_mask)

        # Visualize
        annotated = self.visualize(frame, detections_raw, seg_mask, decision)

        return annotated, decision

    def visualize(
        self,
        frame: np.ndarray,
        detections: List[dict],
        seg_mask: np.ndarray,
        decision: Decision,
    ) -> np.ndarray:
        """Visualize detections, segmentation, and decision on frame."""
        H, W = self.image_size

        # Resize frame to model size
        frame = cv2.resize(frame, (W, H))

        # Create segmentation overlay
        overlay = frame.copy()
        for class_id, color in SEG_COLORS.items():
            mask = seg_mask == class_id
            overlay[mask] = color

        # Blend segmentation
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Draw detection boxes
        class_names = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cls_idx = det['class']
            score = det['score']

            color = DET_COLORS[cls_idx % len(DET_COLORS)]
            # Convert BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            label = f"{class_names[cls_idx]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color_bgr, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw decision overlay (top-right)
        action_color = ACTION_COLORS[decision.action]
        action_color_bgr = (action_color[2], action_color[1], action_color[0])

        # Action box
        cv2.rectangle(frame, (W - 200, 10), (W - 10, 80), action_color_bgr, -1)
        cv2.rectangle(frame, (W - 200, 10), (W - 10, 80), (255, 255, 255), 2)

        # Action text
        cv2.putText(
            frame, decision.action.value,
            (W - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        # Reason text (bottom)
        cv2.rectangle(frame, (10, H - 50), (W - 10, H - 10), (0, 0, 0), -1)
        cv2.putText(
            frame, decision.reason,
            (20, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # Stats (top-left)
        stats = f"Road: {decision.road_percentage:.0%} | Sidewalk: {decision.sidewalk_percentage:.0%} | Objects: {len(detections)}"
        cv2.rectangle(frame, (10, 10), (350, 40), (0, 0, 0), -1)
        cv2.putText(frame, stats, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_video(
        self,
        input_frames: List[Path],
        output_path: str,
        fps: int = 10,
        max_frames: Optional[int] = None,
    ) -> None:
        """Process sequence of frames and output video."""
        if max_frames:
            input_frames = input_frames[:max_frames]

        print(f"\nProcessing {len(input_frames)} frames...")

        # Get frame size
        first_frame = cv2.imread(str(input_frames[0]))
        H, W = self.image_size

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        decisions_log = []

        for i, frame_path in enumerate(input_frames):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"  Warning: Could not read {frame_path}")
                continue

            annotated, decision = self.process_frame(frame)
            writer.write(annotated)

            decisions_log.append({
                'frame': i,
                'action': decision.action.value,
                'reason': decision.reason,
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(input_frames)}] {decision.action.value}: {decision.reason[:50]}...")

        writer.release()
        print(f"\nVideo saved to: {output_path}")

        # Print decision summary
        action_counts = {}
        for d in decisions_log:
            action_counts[d['action']] = action_counts.get(d['action'], 0) + 1

        print("\nDecision Summary:")
        for action, count in sorted(action_counts.items()):
            pct = 100 * count / len(decisions_log)
            print(f"  {action}: {count} frames ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Constellation X Video Processor')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_v2.pt')
    parser.add_argument('--input-dir', type=str, default='data/cityscapes/leftImg8bit/val/frankfurt')
    parser.add_argument('--output', type=str, default='output/constellation_x_demo.mp4')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max-frames', type=int, default=60)
    parser.add_argument('--score-thresh', type=float, default=0.25)
    args = parser.parse_args()

    # Get input frames
    input_dir = Path(args.input_dir)
    frames = sorted(input_dir.glob('*.png'))

    if not frames:
        print(f"No frames found in {input_dir}")
        return

    print(f"Found {len(frames)} frames in {input_dir}")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Process
    processor = VideoProcessor(
        checkpoint_path=args.checkpoint,
        score_thresh=args.score_thresh,
    )

    processor.process_video(
        input_frames=frames,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
