"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

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
