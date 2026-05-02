"""Multi-task training script for HydraNet v2.

Trains detection + segmentation jointly on Cityscapes.

Usage:
    # Local test (small subset)
    python train_multitask.py --subset 100 --epochs 2 --batch-size 4

    # Full training on GPU
    python train_multitask.py --epochs 50 --batch-size 16 --device cuda
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.hydranet_v2 import HydraNetV2
from model.fcos_targets import FCOSTargetAssigner
from model.losses.multi_task_loss import MultiTaskLoss, SegmentationLoss
from data_engine.cityscapes_loader import get_cityscapes_dataloader


class FCOSLoss(nn.Module):
    """FCOS detection loss (simplified from train.py)."""

    def __init__(self, num_classes: int = 8, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        num_pos = (target > 0).sum().float().clamp(min=1.0)

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.reshape(-1)

        target_one_hot = F.one_hot(target_flat, C + 1)[:, 1:].float()

        pred_sigmoid = torch.sigmoid(pred_flat)
        pt = torch.where(target_one_hot == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_t = torch.where(target_one_hot == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma

        bce_loss = F.binary_cross_entropy_with_logits(pred_flat, target_one_hot, reduction='none')
        return (focal_weight * bce_loss).sum() / num_pos

    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_pos = pred.permute(0, 2, 3, 1)[mask]
        target_pos = target.permute(0, 2, 3, 1)[mask]

        pred_pos = F.relu(pred_pos) + 1e-6
        target_pos = target_pos.clamp(min=1e-6)

        pred_area = (pred_pos[:, 0] + pred_pos[:, 2]) * (pred_pos[:, 1] + pred_pos[:, 3])
        target_area = (target_pos[:, 0] + target_pos[:, 2]) * (target_pos[:, 1] + target_pos[:, 3])

        w_inter = torch.min(pred_pos[:, 0], target_pos[:, 0]) + torch.min(pred_pos[:, 2], target_pos[:, 2])
        h_inter = torch.min(pred_pos[:, 1], target_pos[:, 1]) + torch.min(pred_pos[:, 3], target_pos[:, 3])
        intersection = w_inter * h_inter

        union = pred_area + target_area - intersection + 1e-6
        iou = intersection / union

        return (1 - iou).mean()

    def centerness_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_pos = pred.squeeze(1)[mask]
        target_pos = target[mask]
        return F.binary_cross_entropy_with_logits(pred_pos, target_pos)

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        total_loss = 0.0
        det = predictions['detection']

        for scale in ['p3', 'p4', 'p5']:
            scale_pred = det[scale]
            scale_target = targets[scale]
            pos_mask = scale_target['cls'] > 0

            cls_loss = self.focal_loss(scale_pred.cls_logits, scale_target['cls'])
            bbox_loss = self.iou_loss(scale_pred.bbox_pred, scale_target['reg'], pos_mask)
            center_loss = self.centerness_loss(scale_pred.centerness, scale_target['centerness'], pos_mask)

            total_loss = total_loss + cls_loss + bbox_loss + center_loss

        return total_loss / 3  # Average over scales


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    det_loss_fn: FCOSLoss,
    seg_loss_fn: SegmentationLoss,
    mtl_loss_fn: MultiTaskLoss,
    target_assigner: FCOSTargetAssigner,
    device: torch.device,
    epoch: int,
    image_size: tuple = (512, 1024),
) -> dict[str, float]:
    """Train for one epoch with multi-task learning."""
    model.train()

    metrics = {
        'loss': 0.0,
        'det_loss': 0.0,
        'seg_loss': 0.0,
    }
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        boxes = [b.to(device) for b in batch['boxes']]
        labels = [l.to(device) for l in batch['labels']]
        seg_masks = batch['seg_masks'].to(device)

        # Generate FCOS detection targets
        det_targets = target_assigner.assign_targets_batch(
            boxes, labels, image_size=image_size
        )

        # Move targets to device
        for scale in det_targets:
            for key in det_targets[scale]:
                det_targets[scale][key] = det_targets[scale][key].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute task losses
        det_loss = det_loss_fn(outputs, det_targets)
        seg_loss = seg_loss_fn(outputs['segmentation'], seg_masks)

        # Combine with uncertainty weighting
        task_losses = {
            'detection': det_loss,
            'segmentation': seg_loss,
        }
        total_loss, weighted_losses = mtl_loss_fn(task_losses)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # Accumulate metrics
        metrics['loss'] += total_loss.item()
        metrics['det_loss'] += det_loss.item()
        metrics['seg_loss'] += seg_loss.item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={total_loss.item():.4f}, "
                  f"det={det_loss.item():.4f}, "
                  f"seg={seg_loss.item():.4f} "
                  f"[{elapsed:.1f}s]")

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    det_loss_fn: FCOSLoss,
    seg_loss_fn: SegmentationLoss,
    target_assigner: FCOSTargetAssigner,
    device: torch.device,
    image_size: tuple = (512, 1024),
) -> dict[str, float]:
    """Validate the model."""
    model.eval()

    metrics = {
        'loss': 0.0,
        'det_loss': 0.0,
        'seg_loss': 0.0,
        'seg_iou': 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        images = batch['images'].to(device)
        boxes = [b.to(device) for b in batch['boxes']]
        labels = [l.to(device) for l in batch['labels']]
        seg_masks = batch['seg_masks'].to(device)

        det_targets = target_assigner.assign_targets_batch(
            boxes, labels, image_size=image_size
        )

        for scale in det_targets:
            for key in det_targets[scale]:
                det_targets[scale][key] = det_targets[scale][key].to(device)

        outputs = model(images)

        det_loss = det_loss_fn(outputs, det_targets)
        seg_loss = seg_loss_fn(outputs['segmentation'], seg_masks)
        total_loss = det_loss + seg_loss

        # Compute segmentation IoU
        seg_pred = outputs['segmentation'].argmax(dim=1)
        intersection = ((seg_pred == seg_masks) & (seg_masks > 0)).sum().float()
        union = ((seg_pred > 0) | (seg_masks > 0)).sum().float()
        iou = (intersection / (union + 1e-6)).item()

        metrics['loss'] += total_loss.item()
        metrics['det_loss'] += det_loss.item()
        metrics['seg_loss'] += seg_loss.item()
        metrics['seg_iou'] += iou
        num_batches += 1

    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train HydraNet v2 Multi-Task')
    parser.add_argument('--data-dir', type=str, default='data/cityscapes',
                        help='Path to Cityscapes dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of data for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--image-size', type=str, default='512,1024',
                        help='Image size as H,W')
    args = parser.parse_args()

    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))

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
    print(f"HydraNet v2 Multi-Task Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Subset: {args.subset if args.subset else 'Full dataset'}")
    print(f"{'='*60}\n")

    # Create data loaders
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir

    # Create datasets first
    from data_engine.cityscapes_loader import CityscapesDataset
    train_dataset = CityscapesDataset(data_dir, split='train', image_size=image_size)
    val_dataset = CityscapesDataset(data_dir, split='val', image_size=image_size)

    # Apply subset if specified
    if args.subset:
        train_dataset = Subset(train_dataset, range(min(args.subset, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.subset // 5 + 1, len(val_dataset))))
        print(f"Using subset: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CityscapesDataset.collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CityscapesDataset.collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    # Create model
    model = HydraNetV2(
        num_det_classes=8,  # Cityscapes detection classes
        num_seg_classes=3,  # background, road, sidewalk
        pretrained_backbone=True,
        freeze_backbone=True,
    )
    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    # Create losses
    det_loss_fn = FCOSLoss(num_classes=8)
    seg_loss_fn = SegmentationLoss(dice_weight=0.5)
    mtl_loss_fn = MultiTaskLoss(['detection', 'segmentation']).to(device)

    # Target assigner for detection
    target_assigner = FCOSTargetAssigner(num_classes=8)

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Unfreeze backbone after 5 epochs
        if epoch == 5:
            print("Unfreezing backbone...")
            model.unfreeze_backbone()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            det_loss_fn, seg_loss_fn, mtl_loss_fn,
            target_assigner, device, epoch, image_size
        )

        # Get task weights
        task_weights = mtl_loss_fn.get_task_weights()

        print(f"\nTrain: loss={train_metrics['loss']:.4f}, "
              f"det={train_metrics['det_loss']:.4f}, "
              f"seg={train_metrics['seg_loss']:.4f}")
        print(f"Task weights: det={task_weights['detection']:.3f}, "
              f"seg={task_weights['segmentation']:.3f}")

        # Validate
        val_metrics = validate(
            model, val_loader,
            det_loss_fn, seg_loss_fn,
            target_assigner, device, image_size
        )

        print(f"Val:   loss={val_metrics['loss']:.4f}, "
              f"det={val_metrics['det_loss']:.4f}, "
              f"seg={val_metrics['seg_loss']:.4f}, "
              f"seg_iou={val_metrics['seg_iou']:.3f}")

        scheduler.step()

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        best_val_loss = min(val_metrics['loss'], best_val_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mtl_log_vars': {k: v.item() for k, v in mtl_loss_fn.log_vars.items()},
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
        }

        torch.save(checkpoint, checkpoint_dir / 'latest_v2.pt')

        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_v2.pt')
            print(f"  New best model saved! (val_loss={val_metrics['loss']:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch + 1}_v2.pt')

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
