"""Multi-task loss with uncertainty weighting.

Implements the loss function from:
"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
Kendall, Gal, and Cipolla (CVPR 2018)

The key insight: each task has a learnable uncertainty parameter (log variance)
that automatically weights the losses. Tasks with noisy labels get lower weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss.

    Each task has a learnable log_variance parameter that automatically
    balances the contribution of each task to the total loss.

    Loss formula per task:
        L_task = (1 / (2 * exp(log_var))) * loss + log_var / 2

    This allows:
    - Tasks with high loss (noisy labels) to be down-weighted
    - Tasks with low loss (easy tasks) to be up-weighted
    - End-to-end learning of task weights

    Args:
        task_names: List of task names

    Example:
        >>> loss_fn = MultiTaskLoss(['detection', 'lane', 'depth'])
        >>> losses = {'detection': 5.0, 'lane': 0.5, 'depth': 0.1}
        >>> total, per_task = loss_fn(losses)
    """

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = task_names

        # Learnable log variance for each task
        # Initialized to 0 (variance = 1, equal weighting)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute uncertainty-weighted total loss.

        Args:
            losses: Dictionary mapping task name to scalar loss tensor

        Returns:
            Tuple of (total_loss, per_task_weighted_losses)
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weighted_losses = {}

        for name in self.task_names:
            if name not in losses:
                continue

            task_loss = losses[name]
            log_var = self.log_vars[name]

            # Kendall et al. formula:
            # L = (1/(2*sigma^2)) * L_task + log(sigma)
            # Using log_var = log(sigma^2), we get:
            # L = (1/2) * exp(-log_var) * L_task + (1/2) * log_var

            # Use exp(-log_var) instead of 1/exp(log_var) for numerical stability
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var

            weighted_losses[name] = weighted_loss
            total_loss = total_loss + weighted_loss

        return total_loss, weighted_losses

    def get_task_weights(self) -> dict[str, float]:
        """Get current task weights (inverse variance)."""
        with torch.no_grad():
            return {
                name: torch.exp(-self.log_vars[name]).item()
                for name in self.task_names
            }

    def get_log_vars(self) -> dict[str, float]:
        """Get current log variance values."""
        return {
            name: self.log_vars[name].item()
            for name in self.task_names
        }


class DetectionLoss(nn.Module):
    """Combined loss for FCOS-style detection.

    Combines:
    - Focal loss for classification (handles class imbalance)
    - IoU loss for box regression
    - BCE loss for centerness
    """

    def __init__(
        self,
        num_classes: int = 9,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss for classification."""
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_t = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (focal_weight * bce_loss).mean()

    def iou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """IoU loss for box regression (l, t, r, b format)."""
        # Convert to areas
        pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

        # Intersection
        w_intersect = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
        h_intersect = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
        intersection = w_intersect * h_intersect

        # Union
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-6)

        return (1 - iou).mean()

    def forward(
        self,
        cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        centerness_pred: torch.Tensor,
        cls_target: torch.Tensor,
        bbox_target: torch.Tensor,
        centerness_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            cls_pred: Classification predictions (B, C, H, W)
            bbox_pred: Box predictions (B, 4, H, W)
            centerness_pred: Centerness predictions (B, 1, H, W)
            cls_target: Classification targets
            bbox_target: Box targets
            centerness_target: Centerness targets

        Returns:
            Dictionary with 'cls', 'bbox', 'centerness', 'total' losses
        """
        # Flatten spatial dimensions
        B, C, H, W = cls_pred.shape
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(-1, C)
        cls_target = cls_target.permute(0, 2, 3, 1).reshape(-1, C)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target = bbox_target.permute(0, 2, 3, 1).reshape(-1, 4)

        centerness_pred = centerness_pred.reshape(-1)
        centerness_target = centerness_target.reshape(-1)

        # Compute losses
        cls_loss = self.focal_loss(cls_pred, cls_target)
        bbox_loss = self.iou_loss(bbox_pred, bbox_target)
        centerness_loss = F.binary_cross_entropy_with_logits(
            centerness_pred, centerness_target
        )

        total_loss = cls_loss + bbox_loss + centerness_loss

        return {
            'cls': cls_loss,
            'bbox': bbox_loss,
            'centerness': centerness_loss,
            'total': total_loss,
        }


class SegmentationLoss(nn.Module):
    """Combined loss for segmentation tasks.

    Combines cross-entropy and Dice loss for better boundary prediction.
    """

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Dice loss for segmentation."""
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined CE + Dice loss.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, H, W) with class indices

        Returns:
            Combined loss scalar
        """
        ce_loss = F.cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return ce_loss + self.dice_weight * dice


def test_multi_task_loss():
    """Test multi-task loss."""
    print("\n=== Testing MultiTaskLoss ===\n")

    task_names = ['detection', 'lane', 'drivable', 'depth', 'traffic_light']
    loss_fn = MultiTaskLoss(task_names)

    # Dummy losses (different magnitudes)
    losses = {
        'detection': torch.tensor(5.0),
        'lane': torch.tensor(0.5),
        'drivable': torch.tensor(0.3),
        'depth': torch.tensor(0.1),
        'traffic_light': torch.tensor(1.2),
    }

    total, weighted = loss_fn(losses)

    print(f"Input losses: {{{', '.join(f'{k}: {v.item():.2f}' for k, v in losses.items())}}}")
    print(f"Weighted losses: {{{', '.join(f'{k}: {v.item():.2f}' for k, v in weighted.items())}}}")
    print(f"Total loss: {total.item():.2f}")
    print(f"Task weights: {loss_fn.get_task_weights()}")

    # Verify gradients flow
    total.backward()
    for name, param in loss_fn.log_vars.items():
        assert param.grad is not None, f"No gradient for {name}"
    print("\n✅ Gradients flow correctly to log_vars")

    print("\n✅ All multi-task loss tests passed!\n")


if __name__ == '__main__':
    test_multi_task_loss()
