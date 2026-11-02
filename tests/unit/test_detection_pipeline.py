"""
Module: test_detection_pipeline
Description: Unit tests for detection loss, post-processing, metrics, and trainer
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.ops as ops
from torch.utils.data import DataLoader, Dataset

from src.evaluation.detection_metrics import compute_ap, evaluate_detection
from src.models.detection import (
    SolarDetectionLoss,
    build_detection_model,
    class_aware_nms,
    decode_predictions,
)
from src.training.detection_trainer import DetectionTrainer


class DummyDetectionDataset(Dataset):
    """Synthetic dataset generating dummy images and bounding box annotations."""

    def __init__(self, size: int = 4, img_size: int = 64) -> None:
        """Initialize synthetic dataset.

        Args:
            size: Number of samples. Defaults to 4.
            img_size: Image dimension (square). Defaults to 64.
        """
        self.size = size
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img = torch.rand((3, self.img_size, self.img_size), dtype=torch.float32)
        # 2 boxes per image: one small (P3), one larger (P4 or P5)
        boxes = torch.tensor(
            [[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.4, 0.3]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1], dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}


def test_detection_loss_computation() -> None:
    """Test SolarDetectionLoss forward pass and target assignment."""
    criterion = SolarDetectionLoss(num_classes=2)
    b_size = 2

    # Preds for P3 (8x8), P4 (4x4), P5 (2x2) with 7 channels each
    preds = (
        torch.randn(b_size, 7, 8, 8),
        torch.randn(b_size, 7, 4, 4),
        torch.randn(b_size, 7, 2, 2),
    )

    targets = [
        {
            "boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[0.3, 0.4, 0.3, 0.3]]),
            "labels": torch.tensor([0]),
        },
    ]

    loss, loss_dict = criterion(preds, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert "loss_total" in loss_dict
    assert "loss_obj" in loss_dict
    assert "loss_box" in loss_dict
    assert "loss_cls" in loss_dict
    assert loss_dict["num_positives"] >= 2


def test_roundtrip_box_encoding_decoding() -> None:
    """Verify exact round-trip consistency between target encoding and prediction decoding."""
    test_boxes = [
        (0.123, 0.456, 0.025, 0.018),  # Small (bird-dropping)
        (0.505, 0.505, 0.150, 0.120),  # Medium
        (0.334, 0.086, 0.338, 0.172),  # Large (dust)
        (0.001, 0.001, 0.010, 0.010),  # Top-left corner boundary
        (0.995, 0.995, 0.010, 0.010),  # Bottom-right corner boundary
    ]
    w_s, h_s = 80, 80

    for cx, cy, w, h in test_boxes:
        # 1. Target encoding
        x_g, y_g = cx * w_s, cy * h_s
        gx, gy = int(min(w_s - 1, max(0, x_g))), int(min(h_s - 1, max(0, y_g)))
        dx = x_g - gx
        dy = y_g - gy
        dw = torch.log(torch.tensor(max(1e-3, w * w_s)))
        dh = torch.log(torch.tensor(max(1e-3, h * h_s)))

        # 2. Decoding using postprocess logic
        p_box = torch.tensor([dx, dy, dw, dh])
        dec_dx = torch.clamp(p_box[0], 0.0, 1.0)
        dec_dy = torch.clamp(p_box[1], 0.0, 1.0)
        dec_cx = torch.clamp((gx + dec_dx) / w_s, 0.0, 1.0).item()
        dec_cy = torch.clamp((gy + dec_dy) / h_s, 0.0, 1.0).item()
        dec_w = torch.clamp(
            torch.exp(torch.clamp(p_box[2], -5.0, 5.0)) / w_s, 0.0, 1.0
        ).item()
        dec_h = torch.clamp(
            torch.exp(torch.clamp(p_box[3], -5.0, 5.0)) / h_s, 0.0, 1.0
        ).item()

        b_orig = torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])
        b_dec = torch.tensor(
            [
                [
                    dec_cx - dec_w / 2,
                    dec_cy - dec_h / 2,
                    dec_cx + dec_w / 2,
                    dec_cy + dec_h / 2,
                ]
            ]
        )
        iou = ops.box_iou(b_orig, b_dec)[0, 0].item()
        assert (
            iou > 0.9999
        ), f"Round-trip failed for box {(cx, cy, w, h)} with IoU {iou}"


def test_balanced_focal_loss_gradient_ratio() -> None:
    """Verify balanced focal loss yields balanced positive and negative gradient magnitudes."""
    criterion = SolarDetectionLoss(
        num_classes=2, pos_weight_obj=1.0, neg_weight_obj=1.0
    )

    logits = torch.zeros(1, 1, 80, 80, requires_grad=True)
    target = torch.zeros(1, 1, 80, 80)
    # 5 positives in an 80x80 grid (6400 cells)
    target[0, 0, 10, 10] = 1.0
    target[0, 0, 20, 20] = 1.0
    target[0, 0, 30, 30] = 1.0
    target[0, 0, 40, 40] = 1.0
    target[0, 0, 50, 50] = 1.0

    loss = criterion.compute_balanced_focal_obj_loss(logits, target)
    loss.backward()

    pos_grad = logits.grad[target == 1.0].abs().sum().item()
    neg_grad = logits.grad[target == 0.0].abs().sum().item()

    ratio = neg_grad / max(1e-6, pos_grad)
    # The ratio should be close to 1.0, NOT 480x or hundreds
    assert 0.8 <= ratio <= 1.2, f"Gradient ratio {ratio:.2f} is not balanced!"


def test_target_assignment_conflict_resolution() -> None:
    """Verify that smaller boxes take priority over larger boxes in target assignment."""
    criterion = SolarDetectionLoss(num_classes=2)
    b_size = 1

    # Preds for P3 (80x80), P4 (40x40), P5 (20x20)
    preds = (
        torch.zeros(b_size, 7, 80, 80),
        torch.zeros(b_size, 7, 40, 40),
        torch.zeros(b_size, 7, 20, 20),
    )

    # Two boxes mapping to scale P3 at the same integer cell (40, 40)
    # Box 1: Large box (w=0.18, h=0.18, class=0)
    # Box 2: Small box (w=0.03, h=0.03, class=1)
    targets = [
        {
            "boxes": torch.tensor(
                [[0.502, 0.502, 0.18, 0.18], [0.504, 0.504, 0.03, 0.03]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([0, 1], dtype=torch.int64),
        }
    ]

    scale_targets = criterion.build_targets(preds, targets, device=torch.device("cpu"))
    p3_tgt = scale_targets[0]

    # Verify that the smaller box (class 1) won the conflict resolution
    gx, gy = int(0.504 * 80), int(0.504 * 80)
    assigned_cls = p3_tgt["cls"][0, :, gy, gx]
    assert assigned_cls[1].item() == 1.0
    assert assigned_cls[0].item() == 0.0


def test_decode_predictions_and_nms() -> None:
    """Test decoding and class-aware NMS behavior."""
    b_size = 1
    p3 = torch.zeros((b_size, 7, 4, 4))
    p4 = torch.zeros((b_size, 7, 2, 2))
    p5 = torch.zeros((b_size, 7, 1, 1))

    # Activate an object in P3 at cell (1, 1)
    p3[0, 0, 1, 1] = 0.5  # dx in [0, 1]
    p3[0, 1, 1, 1] = 0.5  # dy in [0, 1]
    p3[0, 2, 1, 1] = 0.0  # dw
    p3[0, 3, 1, 1] = 0.0  # dh
    p3[0, 4, 1, 1] = 5.0  # high objectness
    p3[0, 5, 1, 1] = 5.0  # high class 0 logit
    p3[0, 6, 1, 1] = -5.0  # low class 1 logit

    preds = (p3, p4, p5)
    candidates = decode_predictions(preds, conf_threshold=0.2)

    assert len(candidates) == 1
    assert candidates[0].shape[0] >= 1
    assert candidates[0].shape[1] == 6  # [x1, y1, x2, y2, score, class_id]
    assert candidates[0][0, 5].item() == 0.0  # class 0


def test_class_aware_nms_separation() -> None:
    """Verify class-aware NMS does not suppress overlapping boxes of different classes."""
    detections = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.5, 0.9, 0.0],
            [0.1, 0.1, 0.5, 0.5, 0.85, 1.0],
        ],
        dtype=torch.float32,
    )
    keep = class_aware_nms(detections, iou_threshold=0.5)
    assert len(keep) == 2  # Both retained

    same_class = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.5, 0.9, 0.0],
            [0.12, 0.12, 0.5, 0.5, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    keep_same = class_aware_nms(same_class, iou_threshold=0.5)
    assert len(keep_same) == 1  # Suppressed lower confidence box


def test_compute_ap() -> None:
    """Test Average Precision computation with known curves."""
    recalls = np.array([0.0, 0.5, 1.0])
    precisions = np.array([1.0, 1.0, 1.0])
    ap_perfect = compute_ap(recalls, precisions)
    assert np.isclose(ap_perfect, 1.0)

    recalls_zero = np.array([0.0, 0.0, 0.0])
    precisions_zero = np.array([0.0, 0.0, 0.0])
    ap_zero = compute_ap(recalls_zero, precisions_zero)
    assert np.isclose(ap_zero, 0.0)


def test_evaluate_detection_metrics() -> None:
    """Test evaluate_detection on mock DataLoader."""
    dataset = DummyDetectionDataset(size=2, img_size=64)

    def collate_fn(batch: Any) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        imgs = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return imgs, targets

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = build_detection_model("yolov8n", num_classes=2, base_channels=8)

    metrics = evaluate_detection(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        num_classes=2,
        class_names={0: "Dust", 1: "bird-droppings"},
    )

    assert "mAP_50" in metrics
    assert "mAP_50_95" in metrics
    assert "per_class" in metrics
    assert "Dust" in metrics["per_class"]
    assert "bird-droppings" in metrics["per_class"]
    assert metrics["total_images_evaluated"] == 2


def test_detection_trainer_step(tmp_path: Path) -> None:
    """Test DetectionTrainer train_epoch, validate_epoch, and fit for 1 epoch."""
    dataset = DummyDetectionDataset(size=4, img_size=64)

    def collate_fn(batch: Any) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        imgs = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return imgs, targets

    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    model = build_detection_model("yolov8n", num_classes=2, base_channels=8)
    criterion = SolarDetectionLoss(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="test_yolo",
    )

    history = trainer.fit(epochs=1, verbose=False)

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert (tmp_path / "checkpoints" / "test_yolo_best.pt").exists()
