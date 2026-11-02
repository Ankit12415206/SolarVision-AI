"""
Module: postprocess
Description: Prediction decoding and class-aware Non-Maximum Suppression for SolarYOLOv8n
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import List, Tuple

import torch
import torchvision.ops as ops


def decode_predictions(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    strides: Tuple[int, int, int] = (8, 16, 32),
    conf_threshold: float = 0.25,
) -> List[torch.Tensor]:
    """Decode raw multi-scale YOLO feature maps into candidate bounding boxes.

    Args:
        preds: Tuple of predictions (pred_p3, pred_p4, pred_p5).
        strides: Spatial strides for each scale. Defaults to (8, 16, 32).
        conf_threshold: Minimum confidence threshold to keep a candidate.

    Returns:
        List[torch.Tensor]: Per-image tensor of candidates of shape (M, 6):
            [x1, y1, x2, y2, score, class_id] in normalized coordinates [0.0, 1.0].
    """
    b_size = preds[0].size(0)
    device = preds[0].device
    batch_candidates: List[torch.Tensor] = [[] for _ in range(b_size)]

    for pred, stride in zip(preds, strides):
        _, _, h_s, w_s = pred.shape

        # pred channels: [0:4 -> bbox, 4 -> obj, 5: -> cls]
        p_box = pred[:, :4, :, :]
        p_obj = torch.sigmoid(pred[:, 4:5, :, :])
        p_cls = torch.sigmoid(pred[:, 5:, :, :])

        # Precompute coordinate grids (gx, gy)
        gy, gx = torch.meshgrid(
            torch.arange(h_s, device=device, dtype=torch.float32),
            torch.arange(w_s, device=device, dtype=torch.float32),
            indexing="ij",
        )
        gx = gx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gy = gy.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Center decoding: linear clamped offset consistent with loss.py targets
        dx = torch.clamp(p_box[:, 0:1, :, :], 0.0, 1.0)
        dy = torch.clamp(p_box[:, 1:2, :, :], 0.0, 1.0)
        cx = torch.clamp((gx + dx) / w_s, 0.0, 1.0)
        cy = torch.clamp((gy + dy) / h_s, 0.0, 1.0)

        # Size decoding: exp(clamp(dw)) / W_s clamped to valid normalized bounds
        w = torch.clamp(
            torch.exp(torch.clamp(p_box[:, 2:3, :, :], -5.0, 5.0)) / w_s,
            0.0,
            1.0,
        )
        h = torch.clamp(
            torch.exp(torch.clamp(p_box[:, 3:4, :, :], -5.0, 5.0)) / h_s,
            0.0,
            1.0,
        )

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        x1 = torch.clamp(cx - w / 2.0, 0.0, 1.0)
        y1 = torch.clamp(cy - h / 2.0, 0.0, 1.0)
        x2 = torch.clamp(cx + w / 2.0, 0.0, 1.0)
        y2 = torch.clamp(cy + h / 2.0, 0.0, 1.0)

        # Combined class scores: score = obj * class_prob
        # Shape: (B, num_classes, H, W)
        scores = p_obj * p_cls
        max_scores, class_ids = torch.max(scores, dim=1, keepdim=True)

        for b in range(b_size):
            mask = max_scores[b, 0] >= conf_threshold
            if mask.any():
                b_x1 = x1[b, 0][mask]
                b_y1 = y1[b, 0][mask]
                b_x2 = x2[b, 0][mask]
                b_y2 = y2[b, 0][mask]
                b_sc = max_scores[b, 0][mask]
                b_cl = class_ids[b, 0][mask].float()

                cands = torch.stack([b_x1, b_y1, b_x2, b_y2, b_sc, b_cl], dim=1)
                batch_candidates[b].append(cands)

    output: List[torch.Tensor] = []
    for b in range(b_size):
        if batch_candidates[b]:
            output.append(torch.cat(batch_candidates[b], dim=0))
        else:
            output.append(torch.empty((0, 6), device=device, dtype=torch.float32))

    return output


def class_aware_nms(
    detections: torch.Tensor,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> torch.Tensor:
    """Apply class-aware Non-Maximum Suppression to candidate detections.

    Shifts boxes by class ID so detections across different classes never suppress
    each other, preserving multi-class detections at overlapping locations.

    Args:
        detections: Tensor of shape (M, 6) [x1, y1, x2, y2, score, class_id].
        iou_threshold: IoU overlap threshold for suppression. Defaults to 0.45.
        max_detections: Maximum detections to return per image. Defaults to 100.

    Returns:
        torch.Tensor: Filtered detections tensor of shape (K, 6) where K <= max_detections.
    """
    if detections.numel() == 0:
        return detections

    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5]

    # Offset trick for class-aware NMS: offset box coordinates by class_id * large_value
    offsets = classes * 4096.0
    shifted_boxes = boxes + offsets.unsqueeze(1)

    keep_indices = ops.nms(shifted_boxes, scores, iou_threshold)
    keep_indices = keep_indices[:max_detections]

    return detections[keep_indices]


def postprocess_detections(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> List[torch.Tensor]:
    """Execute complete postprocessing pipeline: decoding + class-aware NMS.

    Args:
        preds: Raw model output tuple (pred_p3, pred_p4, pred_p5).
        conf_threshold: Minimum detection confidence threshold. Defaults to 0.25.
        iou_threshold: NMS IoU threshold. Defaults to 0.45.
        max_detections: Maximum detections per image. Defaults to 100.

    Returns:
        List[torch.Tensor]: List of post-processed detection tensors (one per image).
    """
    candidates_list = decode_predictions(preds, conf_threshold=conf_threshold)
    results: List[torch.Tensor] = []

    for cands in candidates_list:
        filtered = class_aware_nms(
            cands,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        results.append(filtered)

    return results
