"""
Module: detection_metrics
Description: Mean Average Precision (mAP) and evaluation metrics for object detection
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader

from src.models.detection.postprocess import postprocess_detections
from src.utils.model_utils import get_device


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO/VOC style).

    Args:
        recalls: 1D array of recall values.
        precisions: 1D array of precision values.

    Returns:
        float: Computed AP scalar in [0.0, 1.0].
    """
    # Append sentinel values at ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve using points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def evaluate_detection(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    num_classes: int = 2,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Evaluate detector on a DataLoader and compute mAP@0.5, mAP@0.5:0.95, precision, and recall.

    Args:
        model: Trained SolarYOLOv8n model.
        data_loader: Detection DataLoader yielding (images, targets).
        device: Computation device. Defaults to get_device().
        num_classes: Number of detection target classes. Defaults to 2.
        conf_threshold: Confidence threshold for predictions. Defaults to 0.15.
        iou_threshold: NMS IoU threshold. Defaults to 0.45.
        class_names: Mapping from class ID to string name. Defaults to {0: "Dust", 1: "bird-droppings"}.

    Returns:
        Dict[str, Any]: Metrics dictionary containing mAP@0.5, mAP@0.5:0.95, per-class AP, precision, recall.
    """
    dev = device or get_device()
    model = model.to(dev)
    model.eval()

    if class_names is None:
        class_names = {0: "Dust", 1: "bird-droppings"}

    all_detections: List[torch.Tensor] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(dev)
            preds = model(images)
            batch_dets = postprocess_detections(
                preds,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            for det in batch_dets:
                all_detections.append(det.cpu())
            for tgt in targets:
                all_targets.append(
                    {
                        "boxes": tgt["boxes"].cpu(),
                        "labels": tgt["labels"].cpu(),
                    }
                )

    # Evaluate per class at IoU thresholds 0.5 to 0.95
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    per_class_results: Dict[str, Any] = {}
    aps_at_50: List[float] = []
    aps_all: List[float] = []

    for cls_id in range(num_classes):
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        cls_aps_across_iou: List[float] = []

        # Count total ground truth boxes for this class
        total_gt = sum(
            int((tgt["labels"] == cls_id).sum().item()) for tgt in all_targets
        )

        for iou_thresh in iou_thresholds:
            tp_list: List[int] = []
            fp_list: List[int] = []
            conf_list: List[float] = []

            for det, tgt in zip(all_detections, all_targets):
                if det.numel() == 0:
                    continue

                det_cls_mask = det[:, 5] == cls_id
                pred_cls_boxes = det[det_cls_mask, :4]
                pred_cls_scores = det[det_cls_mask, 4]

                tgt_cls_mask = tgt["labels"] == cls_id
                gt_boxes_xywh = tgt["boxes"][tgt_cls_mask]

                if gt_boxes_xywh.numel() > 0:
                    # Convert GT [cx, cy, w, h] to [x1, y1, x2, y2]
                    gt_x1 = torch.clamp(
                        gt_boxes_xywh[:, 0] - gt_boxes_xywh[:, 2] / 2.0, 0.0, 1.0
                    )
                    gt_y1 = torch.clamp(
                        gt_boxes_xywh[:, 1] - gt_boxes_xywh[:, 3] / 2.0, 0.0, 1.0
                    )
                    gt_x2 = torch.clamp(
                        gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2] / 2.0, 0.0, 1.0
                    )
                    gt_y2 = torch.clamp(
                        gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3] / 2.0, 0.0, 1.0
                    )
                    gt_boxes_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)
                else:
                    gt_boxes_xyxy = torch.empty((0, 4))

                detected_gt = set()

                for p_idx in range(len(pred_cls_boxes)):
                    p_box = pred_cls_boxes[p_idx : p_idx + 1]
                    conf = float(pred_cls_scores[p_idx].item())
                    conf_list.append(conf)

                    if len(gt_boxes_xyxy) == 0:
                        tp_list.append(0)
                        fp_list.append(1)
                        continue

                    ious = ops.box_iou(p_box, gt_boxes_xyxy)[0]
                    max_iou, max_idx = torch.max(ious, dim=0)

                    if (
                        max_iou.item() >= iou_thresh
                        and max_idx.item() not in detected_gt
                    ):
                        tp_list.append(1)
                        fp_list.append(0)
                        detected_gt.add(max_idx.item())
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

            if total_gt == 0 or len(tp_list) == 0:
                cls_aps_across_iou.append(0.0)
                continue

            # Sort by confidence
            sort_indices = np.argsort(-np.array(conf_list))
            tp = np.cumsum(np.array(tp_list)[sort_indices])
            fp = np.cumsum(np.array(fp_list)[sort_indices])

            recalls = tp / max(1, total_gt)
            precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = compute_ap(recalls, precisions)
            cls_aps_across_iou.append(ap)

        ap_50 = cls_aps_across_iou[0]
        ap_mean = float(np.mean(cls_aps_across_iou))
        aps_at_50.append(ap_50)
        aps_all.append(ap_mean)

        per_class_results[cls_name] = {
            "ap50": round(ap_50, 4),
            "ap50_95": round(ap_mean, 4),
            "ground_truth_count": total_gt,
        }

    map_50 = float(np.mean(aps_at_50)) if aps_at_50 else 0.0
    map_50_95 = float(np.mean(aps_all)) if aps_all else 0.0

    return {
        "mAP_50": round(map_50, 4),
        "mAP_50_95": round(map_50_95, 4),
        "per_class": per_class_results,
        "total_images_evaluated": len(all_targets),
    }
