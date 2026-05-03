"""Gradio app for Constellation HydraNet V2 image + video inference."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from model.hydranet_v2 import HydraNetV2
from video_processor import VideoProcessor


IMAGE_SIZE = (512, 1024)  # H, W
DET_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
DET_COLORS = [
    (255, 0, 0),
    (255, 128, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (128, 128, 128),
    (255, 0, 255),
    (128, 0, 255),
]
SEG_COLORS = {
    0: (0, 0, 0),
    1: (0, 200, 100),
    2: (200, 100, 200),
}


def resolve_checkpoint_path() -> Path:
    candidates = [
        Path(__file__).parent / "checkpoints" / "best_v2.pt",
        Path(__file__).parent / "checkpoints" / "latest_v2.pt",
        Path(__file__).parent / "checkpoints" / "best.pt",
        Path(__file__).parent / "checkpoints" / "latest.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No checkpoint found in checkpoints/ directory")


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float = 0.5) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep: list[int] = []
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
        order = order[1:][iou < thresh]

    return keep


def decode_detections(outputs: dict, score_thresh: float = 0.3) -> list[dict]:
    H, W = IMAGE_SIZE
    strides = {"p3": 8, "p4": 16, "p5": 32}
    det = outputs["detection"]

    all_boxes = []
    all_scores = []
    all_labels = []

    for scale_name in ["p3", "p4", "p5"]:
        scale_out = det[scale_name]
        stride = strides[scale_name]

        cls_logits = scale_out.cls_logits[0]
        bbox_pred = scale_out.bbox_pred[0]
        centerness = scale_out.centerness[0]

        cls_scores = torch.sigmoid(cls_logits)
        center_scores = torch.sigmoid(centerness)
        scores = cls_scores * center_scores

        _, _, _ = cls_scores.shape
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

        all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
        all_scores.append(max_scores[mask])
        all_labels.append(max_classes[mask])

    if len(all_boxes) == 0:
        return []

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    keep = simple_nms(all_boxes, all_scores, 0.5)
    detections = []
    for idx in keep[:50]:
        detections.append(
            {
                "box": all_boxes[idx].tolist(),
                "class": all_labels[idx].item(),
                "score": all_scores[idx].item(),
            }
        )
    return detections


def visualize_results(image: Image.Image, detections: list[dict], seg_mask: np.ndarray) -> Image.Image:
    W, H = image.size
    if seg_mask.shape != (H, W):
        seg_mask = np.array(Image.fromarray(seg_mask.astype(np.uint8)).resize((W, H), Image.NEAREST))

    seg_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for class_id, color in SEG_COLORS.items():
        if class_id == 0:
            continue
        mask = seg_mask == class_id
        seg_rgba[mask] = (*color, 100)

    seg_overlay = Image.fromarray(seg_rgba)
    result = Image.alpha_composite(image.convert("RGBA"), seg_overlay).convert("RGB")
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_idx = det["class"]
        score = det["score"]
        color = DET_COLORS[cls_idx % len(DET_COLORS)]
        label = f"{DET_CLASSES[cls_idx]} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill=(0, 0, 0), font=font)

    return result


print("Loading HydraNet V2 model...")
checkpoint_path = resolve_checkpoint_path()
image_model = HydraNetV2(num_det_classes=8, num_seg_classes=3, pretrained_backbone=False)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model.eval()
print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

# Reuse one video processor instance and only update threshold per request
video_processor = VideoProcessor(
    checkpoint_path=str(checkpoint_path),
    image_size=IMAGE_SIZE,
    device="cpu",
    score_thresh=0.25,
)
print("Model(s) ready!")


def predict(image: Image.Image, score_threshold: float = 0.3) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Please upload an image."

    image_resized = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    image_np = np.array(image_resized)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image_tensor)

    detections = decode_detections(outputs, score_thresh=score_threshold)
    seg_logits = outputs["segmentation"][0]
    seg_mask = seg_logits.argmax(dim=0).numpy()

    total_pixels = seg_mask.size
    road_pct = 100 * (seg_mask == 1).sum() / total_pixels
    sidewalk_pct = 100 * (seg_mask == 2).sum() / total_pixels

    class_counts: dict[str, int] = {}
    for det in detections:
        cls_name = DET_CLASSES[det["class"]]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    summary_lines = [f"**Detections:** {len(detections)} objects found", "", "**Objects:**"]
    if class_counts:
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            summary_lines.append(f"- {cls_name}: {count}")
    else:
        summary_lines.append("- None detected")

    summary_lines.extend(
        [
            "",
            "**Drivable Area:**",
            f"- Road: {road_pct:.1f}%",
            f"- Sidewalk: {sidewalk_pct:.1f}%",
        ]
    )

    result_image = visualize_results(image_resized, detections, seg_mask)
    return result_image, "\n".join(summary_lines)


def predict_video(video_path: str | None, score_threshold: float = 0.25) -> tuple[str | None, str]:
    if not video_path:
        return None, "Please upload a video."

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open uploaded video."

        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        if fps <= 0:
            fps = 10.0

        tmp_dir = Path(tempfile.mkdtemp(prefix="constellation_video_"))
        output_path = tmp_dir / "annotated_output.mp4"

        H, W = IMAGE_SIZE
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (W, H),
        )

        video_processor.score_thresh = float(score_threshold)
        action_counts: Counter[str] = Counter()
        frame_count = 0
        max_frames = 300  # keep runtime bounded on free Spaces

        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, decision = video_processor.process_frame(frame)
            writer.write(annotated)
            action_counts[decision.action.value] += 1
            frame_count += 1

        cap.release()
        writer.release()

        if frame_count == 0:
            return None, "No frames processed from uploaded video."

        summary_lines = [
            f"**Processed Frames:** {frame_count}",
            f"**FPS:** {fps:.1f}",
            "",
            "**Decision Distribution:**",
        ]
        for action, count in action_counts.most_common():
            pct = (count / frame_count) * 100
            summary_lines.append(f"- {action}: {count} ({pct:.1f}%)")

        return str(output_path), "\n".join(summary_lines)
    except Exception as exc:
        return None, f"Video inference failed: {exc}"


with gr.Blocks(title="Constellation: Multi-Task Vision") as demo:
    gr.Markdown(
        """
        # Constellation: Multi-Task Vision for Autonomous Driving
        HydraNet V2 + Constellation X decision engine demo.
        """
    )

    with gr.Tab("Image Inference"):
        gr.Interface(
            fn=predict,
            inputs=[
                gr.Image(type="pil", label="Upload Driving Image"),
                gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Detection Threshold"),
            ],
            outputs=[
                gr.Image(type="pil", label="Detection + Segmentation"),
                gr.Markdown(label="Results Summary"),
            ],
            allow_flagging="never",
        )

    with gr.Tab("Video Inference (Constellation X)"):
        vid_in = gr.Video(label="Upload Driving Video")
        vid_thr = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Detection Threshold")
        vid_out = gr.Video(label="Annotated Output Video")
        vid_txt = gr.Markdown(label="Decision Summary")
        vid_btn = gr.Button("Run Video Inference", variant="primary")
        vid_btn.click(fn=predict_video, inputs=[vid_in, vid_thr], outputs=[vid_out, vid_txt])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, share=False)
