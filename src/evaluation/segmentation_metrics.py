"""
Module: segmentation_metrics
Description: Comprehensive semantic segmentation evaluation metrics and threshold optimization
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.model_utils import get_device


def compute_binary_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float = 1.0,
) -> Dict[str, float]:
    """Compute pixel-level binary segmentation metrics.

    Args:
        pred_mask: Binary prediction tensor {0, 1} of shape (B, 1, H, W) or (B, H, W).
        target_mask: Binary ground-truth mask {0, 1} of matching shape.
        smooth: Laplace smoothing constant. Defaults to 1.0.

    Returns:
        Dict[str, float]: Dictionary containing:
            - dice: Dice coefficient (F1 score)
            - iou: Intersection over Union (Jaccard index)
            - precision: Positive predictive value
            - recall: Sensitivity (true positive rate)
            - pixel_accuracy: Overall pixel classification accuracy
            - defect_ratio_gt: Percentage of defect pixels in ground truth
            - defect_ratio_pred: Percentage of defect pixels in prediction
    """
    p = pred_mask.view(-1).float()
    t = target_mask.view(-1).float()

    tp = (p * t).sum().item()
    fp = (p * (1.0 - t)).sum().item()
    fn = ((1.0 - p) * t).sum().item()
    tn = ((1.0 - p) * (1.0 - t)).sum().item()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    pixel_acc = (tp + tn) / max(1.0, tp + tn + fp + fn)

    total_pixels = max(1.0, float(len(t)))
    gt_ratio = t.sum().item() / total_pixels
    pred_ratio = p.sum().item() / total_pixels

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_acc),
        "defect_ratio_gt": float(gt_ratio),
        "defect_ratio_pred": float(pred_ratio),
    }


def evaluate_segmentation(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate a segmentation model across a DataLoader at a specific threshold.

    Computes both dataset-level global metrics and per-image average metrics.

    Args:
        model: Trained SolarLightweightUNet model.
        data_loader: Segmentation DataLoader yielding (images, masks).
        device: Target computation device. Defaults to get_device().
        threshold: Sigmoid probability threshold for binary mask classification. Defaults to 0.5.
        smooth: Smoothing constant for metrics. Defaults to 1.0.

    Returns:
        Dict[str, Any]: Evaluation summary with global and per-image metrics.
    """
    dev = device or get_device()
    model = model.to(dev)
    model.eval()

    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0
    all_tn = 0.0

    per_image_dices: List[float] = []
    per_image_ious: List[float] = []
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for b in range(images.size(0)):
                p_b = preds[b].view(-1)
                t_b = targets[b].view(-1)

                tp = (p_b * t_b).sum().item()
                fp = (p_b * (1.0 - t_b)).sum().item()
                fn = ((1.0 - p_b) * t_b).sum().item()
                tn = ((1.0 - p_b) * (1.0 - t_b)).sum().item()

                all_tp += tp
                all_fp += fp
                all_fn += fn
                all_tn += tn

                img_dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
                img_iou = (tp + smooth) / (tp + fp + fn + smooth)
                per_image_dices.append(img_dice)
                per_image_ious.append(img_iou)
                total_images += 1

    # Global dataset-level metrics
    global_dice = (2.0 * all_tp + smooth) / (2.0 * all_tp + all_fp + all_fn + smooth)
    global_iou = (all_tp + smooth) / (all_tp + all_fp + all_fn + smooth)
    global_prec = (all_tp + smooth) / (all_tp + all_fp + smooth)
    global_rec = (all_tp + smooth) / (all_tp + all_fn + smooth)
    global_acc = (all_tp + all_tn) / max(1.0, all_tp + all_tn + all_fp + all_fn)

    return {
        "threshold": float(threshold),
        "global_dice": round(float(global_dice), 4),
        "global_iou": round(float(global_iou), 4),
        "global_precision": round(float(global_prec), 4),
        "global_recall": round(float(global_rec), 4),
        "global_pixel_accuracy": round(float(global_acc), 4),
        "mean_image_dice": (
            round(float(np.mean(per_image_dices)), 4) if per_image_dices else 0.0
        ),
        "mean_image_iou": (
            round(float(np.mean(per_image_ious)), 4) if per_image_ious else 0.0
        ),
        "total_images": total_images,
    }


def find_optimal_threshold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    candidate_thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """Search for the threshold maximizing validation Dice coefficient.

    Uses strictly validation data to prevent data leakage onto the test split.

    Args:
        model: Trained model.
        val_loader: Validation DataLoader.
        device: Computation device.
        candidate_thresholds: Iterable of candidate probability thresholds.

    Returns:
        Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
            - best_threshold
            - best_metrics_dict
            - all_threshold_results
    """
    all_results: List[Dict[str, Any]] = []
    best_thresh = 0.5
    best_dice = -1.0
    best_metrics: Dict[str, Any] = {}

    for thresh in candidate_thresholds:
        metrics = evaluate_segmentation(
            model=model,
            data_loader=val_loader,
            device=device,
            threshold=thresh,
        )
        all_results.append(metrics)
        # Select on validation global dice
        if metrics["global_dice"] > best_dice:
            best_dice = metrics["global_dice"]
            best_thresh = thresh
            best_metrics = metrics

    return best_thresh, best_metrics, all_results
