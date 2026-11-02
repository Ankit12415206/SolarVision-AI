"""
Module: test_segmentation_pipeline
Description: Unit tests for segmentation metrics, threshold search, and trainer
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.segmentation_metrics import (
    compute_binary_metrics,
    evaluate_segmentation,
    find_optimal_threshold,
)
from src.models.segmentation import (
    BCEDiceLoss,
    build_segmentation_model,
)
from src.training.segmentation_trainer import SegmentationTrainer


class DummySegmentationDataset(Dataset):
    """Synthetic dataset generating single-channel EL images and binary masks."""

    def __init__(self, size: int = 4, img_size: int = 32) -> None:
        """Initialize synthetic dataset.

        Args:
            size: Number of samples. Defaults to 4.
            img_size: Image dimension. Defaults to 32.
        """
        self.size = size
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.rand((1, self.img_size, self.img_size), dtype=torch.float32)
        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        # Sparse defect in center
        mask[0, 10:15, 10:15] = 1.0
        return img, mask


def test_compute_binary_metrics_perfect_overlap() -> None:
    """Verify metrics with identical prediction and ground truth masks."""
    mask = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    mask[0, 0, 5:10, 5:10] = 1.0

    metrics = compute_binary_metrics(mask, mask, smooth=1e-6)
    assert np.isclose(metrics["dice"], 1.0, atol=1e-4)
    assert np.isclose(metrics["iou"], 1.0, atol=1e-4)
    assert np.isclose(metrics["precision"], 1.0, atol=1e-4)
    assert np.isclose(metrics["recall"], 1.0, atol=1e-4)
    assert np.isclose(metrics["pixel_accuracy"], 1.0, atol=1e-4)


def test_compute_binary_metrics_zero_overlap() -> None:
    """Verify metrics with disjoint prediction and ground truth masks."""
    gt = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    gt[0, 0, 0:5, 0:5] = 1.0

    pred = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    pred[0, 0, 20:25, 20:25] = 1.0

    metrics = compute_binary_metrics(pred, gt, smooth=1e-6)
    assert np.isclose(metrics["dice"], 0.0, atol=1e-4)
    assert np.isclose(metrics["iou"], 0.0, atol=1e-4)
    assert np.isclose(metrics["precision"], 0.0, atol=1e-4)
    assert np.isclose(metrics["recall"], 0.0, atol=1e-4)


def test_evaluate_segmentation_and_threshold_search() -> None:
    """Verify segmentation evaluation and validation threshold search."""
    dataset = DummySegmentationDataset(size=4, img_size=32)
    loader = DataLoader(dataset, batch_size=2)

    model = build_segmentation_model(
        "unet", in_channels=1, out_channels=1, base_channels=8
    )

    metrics = evaluate_segmentation(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        threshold=0.5,
    )

    assert "global_dice" in metrics
    assert "global_iou" in metrics
    assert "global_pixel_accuracy" in metrics
    assert metrics["total_images"] == 4

    best_thresh, best_metrics, all_results = find_optimal_threshold(
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        candidate_thresholds=(0.3, 0.5, 0.7),
    )

    assert best_thresh in (0.3, 0.5, 0.7)
    assert len(all_results) == 3
    assert "global_dice" in best_metrics


def test_segmentation_trainer_synthetic_fit(tmp_path: Path) -> None:
    """Verify SegmentationTrainer train_epoch, validate_epoch, and checkpoint saving."""
    dataset = DummySegmentationDataset(size=4, img_size=32)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    model = build_segmentation_model(
        "unet", in_channels=1, out_channels=1, base_channels=8
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="test_unet",
    )

    history = trainer.fit(epochs=1, verbose=False)

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert len(history["val_dice"]) == 1

    best_ckpt = tmp_path / "checkpoints" / "test_unet_best.pt"
    assert best_ckpt.exists()

    ckpt = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt
    assert ckpt["epoch"] == 1
