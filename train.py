"""Training script for HydraNet v1 detection model.

Usage:
    # Local test (small subset)
    python train.py --subset 100 --epochs 2 --batch-size 4

    # Full training on GPU
    python train.py --epochs 50 --batch-size 32 --device cuda
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection import MeanAveragePrecision

from model.hydranet_v1 import HydraNetV1
from model.fcos_targets import FCOSTargetAssigner
from data_engine.data_loader import get_bdd100k_dataloader


def decode_fcos_batch(
    outputs: dict,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    max_detections: int = 100,
    image_size: tuple = (640, 640),
) -> list[dict]:
    """Decode FCOS outputs to boxes for mAP computation.

    Returns list of dicts (one per image) with keys: boxes, scores, labels
    """
    H, W = image_size
    strides = {'p3': 8, 'p4': 16, 'p5': 32}
    det = outputs['detection']

    batch_size = det['p3'].cls_logits.shape[0]
    batch_results = []

    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []

        for scale_name in ['p3', 'p4', 'p5']:
            scale_out = det[scale_name]
            stride = strides[scale_name]

            cls_logits = scale_out.cls_logits[b]  # (C, H, W)
            bbox_pred = scale_out.bbox_pred[b]     # (4, H, W)
            centerness = scale_out.centerness[b]   # (1, H, W)

            # Compute scores
            cls_scores = torch.sigmoid(cls_logits)
            center_scores = torch.sigmoid(centerness)
            scores = cls_scores * center_scores  # (C, H, W)

            C, fh, fw = cls_scores.shape

            # Get max score per location
            max_scores, max_classes = scores.max(dim=0)  # (H, W)

            # Filter by threshold
            mask = max_scores > score_thresh
            if mask.sum() == 0:
                continue

            # Get coordinates
            ys, xs = torch.where(mask)

            # Decode boxes
            bbox_pred_pos = bbox_pred[:, mask]  # (4, N)
            bbox_pred_pos = F.relu(bbox_pred_pos)

            cx = (xs.float() + 0.5) * stride
            cy = (ys.float() + 0.5) * stride

            x1 = cx - bbox_pred_pos[0]
            y1 = cy - bbox_pred_pos[1]
            x2 = cx + bbox_pred_pos[2]
            y2 = cy + bbox_pred_pos[3]

            # Clip to image
            x1 = x1.clamp(0, W)
            y1 = y1.clamp(0, H)
            x2 = x2.clamp(0, W)
            y2 = y2.clamp(0, H)

            boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)

            all_boxes.append(boxes)
            all_scores.append(max_scores[mask])
            all_labels.append(max_classes[mask])

        if len(all_boxes) == 0:
            batch_results.append({
                'boxes': torch.zeros((0, 4), device=cls_logits.device),
                'scores': torch.zeros((0,), device=cls_logits.device),
                'labels': torch.zeros((0,), dtype=torch.long, device=cls_logits.device),
            })
            continue

        # Concatenate all scales
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Simple class-agnostic NMS
        keep = simple_nms(all_boxes, all_scores, nms_thresh)
        keep = keep[:max_detections]

        batch_results.append({
            'boxes': all_boxes[keep],
            'scores': all_scores[keep],
            'labels': all_labels[keep],
        })

    return batch_results


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """Simple NMS returning indices to keep."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

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

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class FCOSLoss(nn.Module):
    """FCOS detection loss combining focal, IoU, and centerness losses."""

    def __init__(
        self,
        num_classes: int = 10,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for classification.

        IMPORTANT: Normalizes by number of POSITIVE samples, not total pixels.
        This follows the FCOS paper convention.
        """
        # pred: (B, C, H, W), target: (B, H, W) with class indices
        B, C, H, W = pred.shape

        # Count positive samples for normalization
        num_pos = (target > 0).sum().float().clamp(min=1.0)

        # Convert to (B*H*W, C)
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.reshape(-1)

        # Create one-hot targets (class 0 = background -> all zeros)
        target_one_hot = F.one_hot(target_flat, C + 1)[:, 1:]  # Remove background column
        target_one_hot = target_one_hot.float()

        # Focal loss computation
        pred_sigmoid = torch.sigmoid(pred_flat)
        pt = torch.where(target_one_hot == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_t = torch.where(target_one_hot == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_flat, target_one_hot, reduction='none'
        )

        # Sum focal loss and normalize by number of positives (FCOS convention)
        focal_loss = (focal_weight * bce_loss).sum() / num_pos

        return focal_loss

    def iou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """IoU loss for box regression, only on positive samples."""
        # pred, target: (B, 4, H, W) in l,t,r,b format
        # mask: (B, H, W) boolean for positive samples

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Extract positive samples
        pred_pos = pred.permute(0, 2, 3, 1)[mask]  # (N_pos, 4)
        target_pos = target.permute(0, 2, 3, 1)[mask]  # (N_pos, 4)

        # Ensure positive values
        pred_pos = F.relu(pred_pos) + 1e-6
        target_pos = target_pos.clamp(min=1e-6)

        # Compute areas
        pred_area = (pred_pos[:, 0] + pred_pos[:, 2]) * (pred_pos[:, 1] + pred_pos[:, 3])
        target_area = (target_pos[:, 0] + target_pos[:, 2]) * (target_pos[:, 1] + target_pos[:, 3])

        # Intersection
        w_inter = torch.min(pred_pos[:, 0], target_pos[:, 0]) + torch.min(pred_pos[:, 2], target_pos[:, 2])
        h_inter = torch.min(pred_pos[:, 1], target_pos[:, 1]) + torch.min(pred_pos[:, 3], target_pos[:, 3])
        intersection = w_inter * h_inter

        # Union
        union = pred_area + target_area - intersection + 1e-6
        iou = intersection / union

        return (1 - iou).mean()

    def centerness_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss for centerness, only on positive samples."""
        # pred: (B, 1, H, W), target: (B, H, W)
        # mask: (B, H, W) boolean

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_pos = pred.squeeze(1)[mask]
        target_pos = target[mask]

        return F.binary_cross_entropy_with_logits(pred_pos, target_pos)

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute total FCOS loss across all scales.

        Args:
            predictions: Dict with 'detection' containing p3, p4, p5 outputs
            targets: Dict with p3, p4, p5 targets from FCOSTargetAssigner

        Returns:
            Dict with 'cls', 'bbox', 'centerness', 'total' losses
        """
        total_cls = 0.0
        total_bbox = 0.0
        total_center = 0.0

        det = predictions['detection']

        for scale in ['p3', 'p4', 'p5']:
            scale_pred = det[scale]
            scale_target = targets[scale]

            # Get positive mask (class > 0)
            pos_mask = scale_target['cls'] > 0

            # Classification loss (all pixels)
            cls_loss = self.focal_loss(
                scale_pred.cls_logits,
                scale_target['cls'],
            )

            # Regression loss (positive pixels only)
            bbox_loss = self.iou_loss(
                scale_pred.bbox_pred,
                scale_target['reg'],
                pos_mask,
            )

            # Centerness loss (positive pixels only)
            center_loss = self.centerness_loss(
                scale_pred.centerness,
                scale_target['centerness'],
                pos_mask,
            )

            total_cls = total_cls + cls_loss
            total_bbox = total_bbox + bbox_loss
            total_center = total_center + center_loss

        # Average over scales
        total_cls = total_cls / 3
        total_bbox = total_bbox / 3
        total_center = total_center / 3

        total = total_cls + total_bbox + total_center

        return {
            'cls': total_cls,
            'bbox': total_bbox,
            'centerness': total_center,
            'total': total,
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: FCOSLoss,
    target_assigner: FCOSTargetAssigner,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_center = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        boxes = batch['boxes']
        labels = batch['labels']

        # Move boxes/labels to device
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Generate FCOS targets
        targets = target_assigner.assign_targets_batch(
            boxes, labels, image_size=(640, 640)
        )

        # DEBUG: Print target statistics for first batch of first epoch
        if batch_idx == 0 and epoch == 0:
            for scale in ['p3', 'p4', 'p5']:
                cls_t = targets[scale]['cls']
                n_pos = (cls_t > 0).sum().item()
                n_total = cls_t.numel()
                print(f"  [DEBUG] {scale}: {n_pos}/{n_total} positive ({100*n_pos/n_total:.4f}%)")

        # Move targets to device
        for scale in targets:
            for key in targets[scale]:
                targets[scale][key] = targets[scale][key].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        losses = loss_fn(outputs, targets)
        loss = losses['total']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_cls += losses['cls'].item()
        total_bbox += losses['bbox'].item()
        total_center += losses['centerness'].item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, "
                  f"cls={losses['cls'].item():.4f}, "
                  f"bbox={losses['bbox'].item():.4f}, "
                  f"center={losses['centerness'].item():.4f} "
                  f"[{elapsed:.1f}s]")

    return {
        'loss': total_loss / num_batches,
        'cls': total_cls / num_batches,
        'bbox': total_bbox / num_batches,
        'centerness': total_center / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: FCOSLoss,
    target_assigner: FCOSTargetAssigner,
    device: torch.device,
    compute_map: bool = True,
) -> dict[str, float]:
    """Validate the model with optional mAP computation."""
    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_center = 0.0
    num_batches = 0

    # mAP metric
    if compute_map:
        map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    for batch in dataloader:
        images = batch['images'].to(device)
        boxes = [b.to(device) for b in batch['boxes']]
        labels = [l.to(device) for l in batch['labels']]

        targets = target_assigner.assign_targets_batch(
            boxes, labels, image_size=(640, 640)
        )

        for scale in targets:
            for key in targets[scale]:
                targets[scale][key] = targets[scale][key].to(device)

        outputs = model(images)
        losses = loss_fn(outputs, targets)

        total_loss += losses['total'].item()
        total_cls += losses['cls'].item()
        total_bbox += losses['bbox'].item()
        total_center += losses['centerness'].item()
        num_batches += 1

        # Compute mAP
        if compute_map:
            # Decode predictions
            preds = decode_fcos_batch(outputs, score_thresh=0.05)

            # Format for torchmetrics (needs CPU tensors)
            pred_list = []
            target_list = []

            for i in range(len(preds)):
                pred_list.append({
                    'boxes': preds[i]['boxes'].cpu(),
                    'scores': preds[i]['scores'].cpu(),
                    'labels': preds[i]['labels'].cpu(),
                })

                # Ground truth: convert normalized to pixel coords
                gt_boxes = batch['boxes'][i].cpu() * 640  # Scale to pixels
                gt_labels = batch['labels'][i].cpu()

                target_list.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels,
                })

            map_metric.update(pred_list, target_list)

    results = {
        'loss': total_loss / num_batches,
        'cls': total_cls / num_batches,
        'bbox': total_bbox / num_batches,
        'centerness': total_center / num_batches,
    }

    # Compute final mAP
    if compute_map:
        map_results = map_metric.compute()
        results['mAP'] = map_results['map'].item()
        results['mAP_50'] = map_results['map_50'].item()
        results['mAP_75'] = map_results['map_75'].item()

    return results


def main():
    parser = argparse.ArgumentParser(description='Train HydraNet v1')
    parser.add_argument('--data-dir', type=str, default='data/bdd100k_yolo',
                        help='Path to BDD100K YOLO dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of data for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=5,
                        help='Number of epochs to freeze backbone')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='constellation',
                        help='W&B project name')
    args = parser.parse_args()

    # Initialize W&B if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"hydranet-v1-{args.epochs}ep",
                config={
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'subset': args.subset,
                    'freeze_backbone_epochs': args.freeze_backbone_epochs,
                }
            )
            print("[W&B] Logging enabled")
        except ImportError:
            print("[W&B] wandb not installed, skipping logging")
            args.use_wandb = False

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"HydraNet v1 Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Subset: {args.subset if args.subset else 'Full dataset'}")
    print(f"{'='*60}\n")

    # Create data loaders
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir

    train_loader = get_bdd100k_dataloader(
        data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset,
    )

    val_loader = get_bdd100k_dataloader(
        data_dir,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset // 5 if args.subset else None,
    )

    # Create model
    model = HydraNetV1(
        num_classes=10,
        pretrained_backbone=True,
        freeze_backbone=(args.freeze_backbone_epochs > 0),
    )
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    # Create loss and target assigner
    loss_fn = FCOSLoss(num_classes=10)
    target_assigner = FCOSTargetAssigner(num_classes=10)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Unfreeze backbone after warmup
        if epoch == args.freeze_backbone_epochs:
            print("Unfreezing backbone...")
            model.unfreeze_backbone()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            target_assigner, device, epoch
        )

        print(f"\nTrain: loss={train_metrics['loss']:.4f}, "
              f"cls={train_metrics['cls']:.4f}, "
              f"bbox={train_metrics['bbox']:.4f}, "
              f"center={train_metrics['centerness']:.4f}")

        # Validate (compute mAP only every 5 epochs or last epoch to save time)
        compute_map = (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs
        val_metrics = validate(model, val_loader, loss_fn, target_assigner, device, compute_map=compute_map)

        # Print validation results with mAP
        map_str = ""
        if 'mAP_50' in val_metrics:
            map_str = f", mAP@50={val_metrics['mAP_50']:.3f}, mAP={val_metrics['mAP']:.3f}"

        print(f"Val:   loss={val_metrics['loss']:.4f}, "
              f"cls={val_metrics['cls']:.4f}, "
              f"bbox={val_metrics['bbox']:.4f}, "
              f"center={val_metrics['centerness']:.4f}{map_str}")

        # Log to W&B
        if args.use_wandb:
            import wandb
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/cls_loss': train_metrics['cls'],
                'train/bbox_loss': train_metrics['bbox'],
                'train/center_loss': train_metrics['centerness'],
                'val/loss': val_metrics['loss'],
                'val/cls_loss': val_metrics['cls'],
                'val/bbox_loss': val_metrics['bbox'],
                'val/center_loss': val_metrics['centerness'],
                'lr': scheduler.get_last_lr()[0],
            }
            if 'mAP_50' in val_metrics:
                log_dict['val/mAP_50'] = val_metrics['mAP_50']
                log_dict['val/mAP'] = val_metrics['mAP']
            wandb.log(log_dict)

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        best_val_loss = min(val_metrics['loss'], best_val_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'mAP_50': val_metrics.get('mAP_50', 0.0),
            'mAP': val_metrics.get('mAP', 0.0),
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            print(f"  New best model saved! (val_loss={val_metrics['loss']:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch + 1}.pt')

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Finish W&B run
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
