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


def test_decode_predictions_and_nms() -> None:
    """Test decoding and class-aware NMS behavior."""
    b_size = 1
    # Create simple prediction with one confident detection
    p3 = torch.zeros((b_size, 7, 4, 4))
    p4 = torch.zeros((b_size, 7, 2, 2))
    p5 = torch.zeros((b_size, 7, 1, 1))

    # Activate an object in P3 at cell (1, 1)
    p3[0, 0, 1, 1] = 0.0  # dx
    p3[0, 1, 1, 1] = 0.0  # dy
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
    # Two identical boxes with different class IDs (0 and 1)
    detections = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.5, 0.9, 0.0],
            [0.1, 0.1, 0.5, 0.5, 0.85, 1.0],
        ],
        dtype=torch.float32,
    )
    keep = class_aware_nms(detections, iou_threshold=0.5)
    assert len(keep) == 2  # Both retained

    # Two overlapping boxes of the same class (both class 0)
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
    # Perfect precision-recall curve
    recalls = np.array([0.0, 0.5, 1.0])
    precisions = np.array([1.0, 1.0, 1.0])
    ap_perfect = compute_ap(recalls, precisions)
    assert np.isclose(ap_perfect, 1.0)

    # Empty/zero precision
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
