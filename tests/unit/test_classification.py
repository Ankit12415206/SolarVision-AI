"""
Module: test_classification
Description: Unit tests for classification dataset, manifests, zero-leakage splits, and dataloaders
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pandas as pd
import torch

from src.data.classification import (
    CLASSIFICATION_CLASSES,
    SolarClassificationDataset,
    create_classification_splits,
    get_classification_dataloaders,
    scan_classification_images,
)


def test_scan_classification_images() -> None:
    """Test scanning raw classification images."""
    records = scan_classification_images("data/raw/classification")
    assert len(records) == 885
    assert all("hash_id" in r for r in records)


def test_create_classification_splits(tmp_path: Path) -> None:
    """Test creating classification splits in a temporary directory."""
    stats = create_classification_splits(
        root_dir="data/raw/classification",
        output_dir=tmp_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert stats["total_images"] == 885
    assert not stats["leakage_detected"]
    assert (tmp_path / "classification_train.csv").exists()
    assert (tmp_path / "classification_val.csv").exists()
    assert (tmp_path / "classification_test.csv").exists()


def test_classification_manifests_exist() -> None:
    """Verify classification train/val/test split manifests exist."""
    manifest_dir = Path("data/processed/splits")
    train_csv = manifest_dir / "classification_train.csv"
    val_csv = manifest_dir / "classification_val.csv"
    test_csv = manifest_dir / "classification_test.csv"

    assert train_csv.exists(), "classification_train.csv missing"
    assert val_csv.exists(), "classification_val.csv missing"
    assert test_csv.exists(), "classification_test.csv missing"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Check that all 885 images are accounted for
    total_imgs = len(df_train) + len(df_val) + len(df_test)
    assert total_imgs == 885, f"Expected 885 images, got {total_imgs}"


def test_classification_zero_duplicate_leakage() -> None:
    """Verify zero duplicate content leakage across train, val, and test splits."""
    manifest_dir = Path("data/processed/splits")
    df_train = pd.read_csv(manifest_dir / "classification_train.csv")
    df_val = pd.read_csv(manifest_dir / "classification_val.csv")
    df_test = pd.read_csv(manifest_dir / "classification_test.csv")

    hashes_train = set(df_train["hash_id"])
    hashes_val = set(df_val["hash_id"])
    hashes_test = set(df_test["hash_id"])

    assert (
        len(hashes_train & hashes_val) == 0
    ), "Leakage detected between train and val!"
    assert (
        len(hashes_train & hashes_test) == 0
    ), "Leakage detected between train and test!"
    assert len(hashes_val & hashes_test) == 0, "Leakage detected between val and test!"


def test_classification_all_classes_in_splits() -> None:
    """Verify all 6 classes are present in train, val, and test splits."""
    manifest_dir = Path("data/processed/splits")
    for split in ["train", "val", "test"]:
        df = pd.read_csv(manifest_dir / f"classification_{split}.csv")
        classes_present = set(df["class_name"])
        for expected_cls in CLASSIFICATION_CLASSES:
            assert (
                expected_cls in classes_present
            ), f"Class {expected_cls} missing in {split}!"


def test_classification_dataset_and_loader() -> None:
    """Test Dataset loading and DataLoader mini-batch generation."""
    manifest_dir = Path("data/processed/splits")
    train_loader, val_loader, test_loader = get_classification_dataloaders(
        manifest_dir=manifest_dir,
        image_size=128,
        batch_size=4,
        num_workers=0,
    )

    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape == (4, 3, 128, 128)
    assert batch_labels.shape == (4,)
    assert batch_images.dtype == torch.float32
    assert batch_labels.dtype == torch.int64


def test_classification_dataset_from_dataframe() -> None:
    """Test initializing Dataset from dataframe and list."""
    train_csv = Path("data/processed/splits/classification_train.csv")
    sample_row = pd.read_csv(train_csv).iloc[0].to_dict()
    df = pd.DataFrame([sample_row])
    ds = SolarClassificationDataset(df)
    assert len(ds) == 1
    img, lbl = ds[0]
    assert isinstance(lbl, int)

    ds_list = SolarClassificationDataset([sample_row])
    assert len(ds_list) == 1
