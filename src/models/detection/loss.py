"""
Module: loss
Description: Anchor-free detection loss for multi-scale YOLO feature heads
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SolarDetectionLoss(nn.Module):
    """Loss module for SolarYOLOv8n multi-scale decoupled detection heads.

    Computes balanced focal objectness, anchor-free bounding box regression, and multi-class classification
    losses across P3 (stride 8), P4 (stride 16), and P5 (stride 32) feature maps.
    """

    def __init__(
        self,
        num_classes: int = 2,
        strides: Tuple[int, int, int] = (8, 16, 32),
        lambda_obj: float = 1.0,
        lambda_box: float = 2.0,
        lambda_cls: float = 1.0,
        pos_weight_obj: float = 1.0,
        neg_weight_obj: float = 1.0,
        focal_gamma: float = 1.5,
    ) -> None:
        """Initialize SolarDetectionLoss.

        Args:
            num_classes: Number of defect classes. Defaults to 2 (Dust, bird-droppings).
            strides: Downsampling factors for P3, P4, P5 heads. Defaults to (8, 16, 32).
            lambda_obj: Objectness loss scaling weight. Defaults to 1.0.
            lambda_box: Bounding box regression loss scaling weight. Defaults to 2.0.
            lambda_cls: Classification loss scaling weight. Defaults to 1.0.
            pos_weight_obj: Scaling weight for positive objectness term. Defaults to 1.0.
            neg_weight_obj: Scaling weight for negative objectness term. Defaults to 1.0.
            focal_gamma: Focusing parameter for modulating easy/hard examples. Defaults to 1.5.
        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.pos_weight_obj = pos_weight_obj
        self.neg_weight_obj = neg_weight_obj
        self.focal_gamma = focal_gamma

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1, reduction="mean")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")

    def compute_balanced_focal_obj_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute numerically stable balanced focal BCE loss for objectness.

        Separately normalizes positive and negative contributions to prevent overwhelming
        background dominance across thousands of feature cells.

        Args:
            logits: Predicted objectness logits (B, 1, H, W).
            targets: Binary target map (B, 1, H, W) in {0.0, 1.0}.

        Returns:
            torch.Tensor: Scalar balanced focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)
        focal_bce = focal_weight * bce

        pos_mask = targets == 1.0
        neg_mask = targets == 0.0

        n_pos = torch.clamp(pos_mask.sum(), min=1.0)
        n_neg = torch.clamp(neg_mask.sum(), min=1.0)

        loss_pos = focal_bce[pos_mask].sum() / n_pos
        loss_neg = focal_bce[neg_mask].sum() / n_neg

        return self.pos_weight_obj * loss_pos + self.neg_weight_obj * loss_neg

    def build_targets(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Construct multi-scale ground truth grid tensors for P3, P4, and P5.

        Args:
            preds: Model predictions (pred_p3, pred_p4, pred_p5).
            targets: List of target dicts containing 'boxes' (N, 4) in [cx, cy, w, h] and 'labels' (N,).
            device: Target torch device.

        Returns:
            List[Dict[str, torch.Tensor]]: List of 3 dicts (one per scale) containing:
                - 'obj': (B, 1, H_s, W_s)
                - 'box': (B, 4, H_s, W_s)
                - 'cls': (B, num_classes, H_s, W_s)
                - 'mask': (B, 1, H_s, W_s)
        """
        b_size = preds[0].size(0)
        scale_targets: List[Dict[str, torch.Tensor]] = []

        for p, stride in zip(preds, self.strides):
            _, _, h_s, w_s = p.shape
            t_obj = torch.zeros(
                (b_size, 1, h_s, w_s), device=device, dtype=torch.float32
            )
            t_box = torch.zeros(
                (b_size, 4, h_s, w_s), device=device, dtype=torch.float32
            )
            t_cls = torch.zeros(
                (b_size, self.num_classes, h_s, w_s), device=device, dtype=torch.float32
            )
            mask = torch.zeros((b_size, 1, h_s, w_s), device=device, dtype=torch.bool)

            scale_targets.append(
                {"obj": t_obj, "box": t_box, "cls": t_cls, "mask": mask}
            )

        for b_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)

            if len(boxes) == 0:
                continue

            # Sort boxes by area descending so smaller boxes are assigned LAST.
            # This conflict resolution ensures smaller defect boxes (e.g. tiny bird droppings)
            # take priority over larger diffuse dust clouds if mapping to the same cell.
            box_areas = boxes[:, 2] * boxes[:, 3]
            sort_indices = torch.argsort(box_areas, descending=True)
            sorted_boxes = boxes[sort_indices]
            sorted_labels = labels[sort_indices]

            for i in range(len(sorted_boxes)):
                cx, cy, w, h = sorted_boxes[i].tolist()
                cls_id = int(sorted_labels[i].item())
                if cls_id >= self.num_classes:
                    continue

                # Scale assignment based on bounding box dimension
                box_sz = max(w, h)
                if box_sz < 0.2:
                    scale_idx = 0  # P3 (stride 8)
                elif box_sz < 0.5:
                    scale_idx = 1  # P4 (stride 16)
                else:
                    scale_idx = 2  # P5 (stride 32)

                s_dict = scale_targets[scale_idx]
                _, _, h_s, w_s = preds[scale_idx].shape

                gx = int(min(w_s - 1, max(0, cx * w_s)))
                gy = int(min(h_s - 1, max(0, cy * h_s)))

                # Linear offset targets strictly within [0.0, 1.0]
                s_dict["obj"][b_idx, 0, gy, gx] = 1.0
                s_dict["mask"][b_idx, 0, gy, gx] = True
                s_dict["box"][b_idx, 0, gy, gx] = (cx * w_s) - gx
                s_dict["box"][b_idx, 1, gy, gx] = (cy * h_s) - gy
                s_dict["box"][b_idx, 2, gy, gx] = torch.log(
                    torch.tensor(max(1e-3, w * w_s), device=device)
                )
                s_dict["box"][b_idx, 3, gy, gx] = torch.log(
                    torch.tensor(max(1e-3, h * h_s), device=device)
                )
                s_dict["cls"][b_idx, :, gy, gx] = 0.0  # Reset other classes on cell
                s_dict["cls"][b_idx, cls_id, gy, gx] = 1.0

        return scale_targets

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined detection loss across all feature scales.

        Args:
            preds: Model output tuple (pred_p3, pred_p4, pred_p5).
            targets: List of target dictionaries.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: (total_loss, metric_loss_dict).
        """
        device = preds[0].device
        scale_targets = self.build_targets(preds, targets, device)

        total_loss_obj = torch.tensor(0.0, device=device)
        total_loss_box = torch.tensor(0.0, device=device)
        total_loss_cls = torch.tensor(0.0, device=device)
        total_positives = 0

        for pred, tgt in zip(preds, scale_targets):
            p_box = pred[:, :4, :, :]
            p_obj = pred[:, 4:5, :, :]
            p_cls = pred[:, 5:, :, :]

            # Balanced Focal Objectness loss
            scale_obj_loss = self.compute_balanced_focal_obj_loss(p_obj, tgt["obj"])
            total_loss_obj += scale_obj_loss

            # Box and class loss only on cells with active targets
            mask = tgt["mask"]
            num_pos = mask.sum().item()

            if num_pos > 0:
                mask_4 = mask.expand(-1, 4, -1, -1)
                total_loss_box += self.smooth_l1(p_box[mask_4], tgt["box"][mask_4])

                mask_cls = mask.expand(-1, self.num_classes, -1, -1)
                total_loss_cls += self.bce_cls(p_cls[mask_cls], tgt["cls"][mask_cls])
                total_positives += num_pos

        loss = (
            self.lambda_obj * total_loss_obj
            + self.lambda_box * total_loss_box
            + self.lambda_cls * total_loss_cls
        )

        loss_dict = {
            "loss_total": float(loss.item()),
            "loss_obj": float(total_loss_obj.item()),
            "loss_box": float(total_loss_box.item()),
            "loss_cls": float(total_loss_cls.item()),
            "num_positives": total_positives,
        }

        return loss, loss_dict
