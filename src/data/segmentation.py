"""
Module: segmentation
Description: Dataset, DataLoaders, and splits for MSPS EL Defect Segmentation
Author: SolarVision AI Team
Created: 2026-11-02
"""

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import DualSegmentationTransform
from src.utils.image import load_image


def scan_segmentation_data(
    root_dir: Union[str, Path],
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan Dataset C images and masks, verifying 1-to-1 correspondence.

    Args:
        root_dir: Path to `data/raw/microcracks/msps/`.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping 'train' and 'test' to sample record lists.

    Raises:
        ValueError: If image and mask pairs do not match.
    """
    root_path = Path(root_dir)
    splits_data: Dict[str, List[Dict[str, Any]]] = {}

    for split in ["train", "test"]:
        split_dir = root_path / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        lbl01_dir = split_dir / "labels01"

        if not img_dir.exists():
            continue

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".png"])
        recs: List[Dict[str, Any]] = []

        for img_path in images:
            mask_path = lbl01_dir / img_path.name
            if not mask_path.exists():
                mask_path = lbl_dir / img_path.name

            if not mask_path.exists():
                raise ValueError(f"Missing mask for image: {img_path}")

            recs.append(
                {
                    "image_path": str(img_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "relative_path": str(img_path.relative_to(root_path.parent.parent)),
                    "filename": img_path.name,
                    "split": split,
                }
            )

        splits_data[split] = recs

    return splits_data


def create_segmentation_splits(
    root_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create validation split from official train data while strictly preserving test split.

    Args:
        root_dir: Path to `data/raw/microcracks/msps/`.
        output_dir: Output directory for manifest CSVs.
        val_ratio: Fraction of train set to allocate to validation. Defaults to 0.15.
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Preprocessing summary with sample counts and pixel imbalance stats.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits_data = scan_segmentation_data(root_dir)
    train_pool = list(splits_data.get("train", []))
    test_recs = list(splits_data.get("test", []))

    # Deterministically split train_pool into train and val
    rng = random.Random(seed)
    rng.shuffle(train_pool)

    n_val = int(round(len(train_pool) * val_ratio))
    val_recs = train_pool[:n_val]
    train_recs = train_pool[n_val:]

    # Mark split names
    for r in train_recs:
        r["split"] = "train"
    for r in val_recs:
        r["split"] = "val"
    for r in test_recs:
        r["split"] = "test"

    fieldnames = ["image_path", "mask_path", "relative_path", "filename", "split"]

    for split_name, recs in [
        ("train", train_recs),
        ("val", val_recs),
        ("test", test_recs),
    ]:
        csv_file = out_path / f"segmentation_{split_name}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)

    # Compute pixel imbalance statistics on training samples
    total_pixels = 0
    defect_pixels = 0
    sample_for_stats = train_recs[:100]

    for rec in sample_for_stats:
        with Image.open(rec["mask_path"]) as m:
            arr = np.array(m)
            defect_pixels += int(np.sum(arr > 0))
            total_pixels += arr.size

    bg_pixels = total_pixels - defect_pixels
    defect_ratio = defect_pixels / max(1, total_pixels)
    pos_weight = float(bg_pixels / max(1, defect_pixels))

    return {
        "train_samples": len(train_recs),
        "val_samples": len(val_recs),
        "test_samples": len(test_recs),
        "total_samples": len(train_recs) + len(val_recs) + len(test_recs),
        "pixel_imbalance_stats": {
            "sampled_images": len(sample_for_stats),
            "total_pixels": total_pixels,
            "defect_pixels": defect_pixels,
            "defect_percentage": round(defect_ratio * 100, 4),
            "pos_weight_for_bce": round(pos_weight, 2),
        },
    }


class SolarSegmentationDataset(Dataset):
    """PyTorch Dataset for MSPS EL Defect Segmentation."""

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]],
        image_size: int = 256,
        is_training: bool = False,
        dual_transform: Optional[DualSegmentationTransform] = None,
    ) -> None:
        """Initialize segmentation dataset.

        Args:
            manifest: Path to manifest CSV, DataFrame, or list of records.
            image_size: Target square spatial resolution. Defaults to 256.
            is_training: Whether to enable training augmentations.
            dual_transform: Optional custom DualSegmentationTransform instance.
        """
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        elif isinstance(manifest, list):
            self.df = pd.DataFrame(manifest)
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        if dual_transform is not None:
            self.transform = dual_transform
        else:
            self.transform = DualSegmentationTransform(
                image_size=image_size,
                is_training=is_training,
            )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image and binary mask lazily.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image_tensor: FloatTensor of shape (1, H, W) normalized.
                - mask_tensor: FloatTensor of shape (1, H, W) binary {0.0, 1.0}.
        """
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = load_image(img_path, mode="L")
        mask = load_image(mask_path, mode="L")

        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor


def get_segmentation_dataloaders(
    manifest_dir: Union[str, Path],
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders for segmentation.

    Args:
        manifest_dir: Directory containing segmentation train/val/test CSV manifests.
        image_size: Target square spatial dimension. Defaults to 256.
        batch_size: Batch size. Defaults to 8.
        num_workers: Background workers. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    manifest_path = Path(manifest_dir)

    train_ds = SolarSegmentationDataset(
        manifest_path / "segmentation_train.csv",
        image_size=image_size,
        is_training=True,
    )
    val_ds = SolarSegmentationDataset(
        manifest_path / "segmentation_val.csv",
        image_size=image_size,
        is_training=False,
    )
    test_ds = SolarSegmentationDataset(
        manifest_path / "segmentation_test.csv",
        image_size=image_size,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
