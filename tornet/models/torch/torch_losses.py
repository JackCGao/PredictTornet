from __future__ import annotations

import torch


def _prep(class_labels: torch.Tensor, logits: torch.Tensor):
    y_true = class_labels.to(dtype=logits.dtype)
    y_pred = torch.sigmoid(logits)
    return y_true, y_pred


def mae_loss(
    class_labels: torch.Tensor,
    logits: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    class_labels: tensor of binary ground truth labels (0/1).
    logits: raw model outputs before sigmoid.
    sample_weights: optional per-sample weights (same shape as labels).
    """
    y_true, y_pred = _prep(class_labels, logits)
    diff = torch.abs(y_true - y_pred)
    if sample_weights is not None:
        weights = sample_weights.to(dtype=logits.dtype)
        denom = torch.sum(weights)
        return torch.sum(weights * diff) / (denom + torch.finfo(logits.dtype).eps)
    return torch.mean(diff)


def jaccard_loss(class_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Binary Jaccard (IoU) loss computed on sigmoid probabilities.
    """
    y_true, y_pred = _prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred - intersection
    eps = torch.finfo(logits.dtype).eps
    iou = intersection / (union + eps)
    return torch.mean(1.0 - iou)


def dice_loss(class_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Binary Dice loss computed on sigmoid probabilities.
    """
    y_true, y_pred = _prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred
    eps = torch.finfo(logits.dtype).eps
    dice = (2.0 * intersection + eps) / (union + eps)
    return torch.mean(1.0 - dice)
