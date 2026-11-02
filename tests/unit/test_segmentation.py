"""
Module: test_segmentation
Description: Unit tests for segmentation dataset, mask integrity, and dataloaders
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pandas as pd
import torch

from src.data.segmentation import (
    SolarSegmentationDataset,
    create_segmentation_splits,
    get_segmentation_dataloaders,
    scan_segmentation_data,
)


def test_scan_segmentation_data() -> None:
    """Test scanning segmentation images and masks."""
    splits = scan_segmentation_data("data/raw/microcracks/msps")
    assert "train" in splits
    assert "test" in splits
    assert len(splits["train"]) == 1004
    assert len(splits["test"]) == 503


def test_create_segmentation_splits(tmp_path: Path) -> None:
    """Test creating segmentation splits."""
    stats = create_segmentation_splits(
        root_dir="data/raw/microcracks/msps",
        output_dir=tmp_path,
        val_ratio=0.15,
        seed=42,
    )
    assert stats["total_samples"] == 1507
    assert stats["test_samples"] == 503
    assert (tmp_path / "segmentation_train.csv").exists()
    assert (tmp_path / "segmentation_val.csv").exists()
    assert (tmp_path / "segmentation_test.csv").exists()


def test_segmentation_manifests() -> None:
    """Verify segmentation manifests preserve test split and sum to 1507."""
    manifest_dir = Path("data/processed/splits")
    df_train = pd.read_csv(manifest_dir / "segmentation_train.csv")
    df_val = pd.read_csv(manifest_dir / "segmentation_val.csv")
    df_test = pd.read_csv(manifest_dir / "segmentation_test.csv")

    assert len(df_test) == 503, "Official test split must be strictly 503 samples!"
    assert len(df_train) == 853
    assert len(df_val) == 151
    assert len(df_train) + len(df_val) + len(df_test) == 1507


def test_segmentation_dataloader_and_masks() -> None:
    """Test segmentation DataLoader shapes and mask discrete binary values."""
    manifest_dir = Path("data/processed/splits")
    train_loader, val_loader, test_loader = get_segmentation_dataloaders(
        manifest_dir=manifest_dir,
        image_size=128,
        batch_size=4,
        num_workers=0,
    )

    batch_images, batch_masks = next(iter(train_loader))
    assert batch_images.shape == (4, 1, 128, 128)
    assert batch_masks.shape == (4, 1, 128, 128)
    assert batch_images.dtype == torch.float32
    assert batch_masks.dtype == torch.float32

    # Check discrete mask values
    unique_mask_values = torch.unique(batch_masks).tolist()
    assert set(unique_mask_values).issubset({0.0, 1.0})


def test_segmentation_dataset_from_list() -> None:
    """Test initializing segmentation dataset with list of dicts."""
    train_csv = Path("data/processed/splits/segmentation_train.csv")
    sample_row = pd.read_csv(train_csv).iloc[0].to_dict()
    ds = SolarSegmentationDataset([sample_row])
    assert len(ds) == 1
    img, mask = ds[0]
    assert img.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)
