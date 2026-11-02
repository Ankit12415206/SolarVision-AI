"""
Module: metrics
Description: Evaluation metrics calculation for multiclass classification
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.utils.model_utils import get_device


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full evaluation on a DataLoader and compute multi-class metrics.

    Args:
        model: Trained PyTorch classification model.
        data_loader: Evaluation DataLoader (val or test).
        device: Computation device. Defaults to get_device().
        class_names: List of class name strings for per-class reporting.

    Returns:
        Dict[str, Any]: Dictionary containing overall metrics, per-class metrics,
            confusion matrix, raw predictions, and ground truth targets.
    """
    dev = device or get_device()
    model = model.to(dev)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    for images, labels in data_loader:
        images = images.to(dev, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    prec_weighted = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report,
        "total_samples": len(y_true),
        "predictions": all_preds,
        "targets": all_targets,
    }
