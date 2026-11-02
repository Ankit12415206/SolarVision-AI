"""
Module: test_detection
Description: Unit tests for detection dataset, YOLO parsing, and dataloaders
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pandas as pd

from src.data.detection import (
    SolarDetectionDataset,
    get_detection_dataloaders,
    parse_yolo_label,
    verify_and_manifest_detection_data,
)


def test_parse_yolo_label(tmp_path: Path) -> None:
    """Test parsing YOLO format coordinates."""
    lbl_file = tmp_path / "label.txt"
    lbl_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.05 0.08\n", encoding="utf-8")

    boxes = parse_yolo_label(lbl_file)
    assert len(boxes) == 2
    assert boxes[0]["class_id"] == 0
    assert boxes[0]["x_center"] == 0.5
    assert boxes[1]["class_id"] == 1


def test_parse_yolo_label_nonexistent() -> None:
    """Test parsing a nonexistent label file."""
    boxes = parse_yolo_label("non_existent_label.txt")
    assert boxes == []


def test_verify_and_manifest_detection_data(tmp_path: Path) -> None:
    """Test verifying and creating manifests for detection dataset."""
    stats = verify_and_manifest_detection_data(
        root_dir="data/raw/dust/aerial_soiling",
        output_dir=tmp_path,
    )
    assert stats["total_images"] == 302
    assert not stats["leakage_detected"]
    assert (tmp_path / "detection_train.csv").exists()
    assert (tmp_path / "detection_val.csv").exists()
    assert (tmp_path / "detection_test.csv").exists()


def test_detection_manifests_and_leakage() -> None:
    """Verify detection manifests and check zero cross-split duplicate leakage."""
    manifest_dir = Path("data/processed/splits")
    df_train = pd.read_csv(manifest_dir / "detection_train.csv")
    df_val = pd.read_csv(manifest_dir / "detection_val.csv")
    df_test = pd.read_csv(manifest_dir / "detection_test.csv")

    assert len(df_train) == 257
    assert len(df_val) == 22
    assert len(df_test) == 23

    hashes_train = set(df_train["hash_id"])
    hashes_val = set(df_val["hash_id"])
    hashes_test = set(df_test["hash_id"])

    assert (
        len(hashes_train & hashes_val) == 0
    ), "Detection leakage between train and val!"
    assert (
        len(hashes_train & hashes_test) == 0
    ), "Detection leakage between train and test!"
    assert len(hashes_val & hashes_test) == 0, "Detection leakage between val and test!"


def test_detection_dataloader() -> None:
    """Test detection DataLoader with custom collate function."""
    manifest_dir = Path("data/processed/splits")
    train_loader, val_loader, test_loader = get_detection_dataloaders(
        manifest_dir=manifest_dir,
        image_size=320,
        batch_size=4,
        num_workers=0,
    )

    batch_images, batch_targets = next(iter(train_loader))
    assert batch_images.shape == (4, 3, 320, 320)
    assert isinstance(batch_targets, list)
    assert len(batch_targets) == 4
    for target in batch_targets:
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].ndim == 2


def test_detection_dataset_from_list() -> None:
    """Test SolarDetectionDataset initialization with list."""
    train_csv = Path("data/processed/splits/detection_train.csv")
    sample_row = pd.read_csv(train_csv).iloc[0].to_dict()
    ds = SolarDetectionDataset([sample_row])
    assert len(ds) == 1
    img, target = ds[0]
    assert img.ndim == 3
