"""
Module: classification
Description: Dataset, DataLoaders, and splits for Solar Panel Classification
Author: SolarVision AI Team
Created: 2026-11-02
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import get_classification_transforms
from src.utils.image import compute_file_hash, load_image

# Canonical 6 classes and integer mapping for Dataset A
CLASSIFICATION_CLASSES: List[str] = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]
CLASS_TO_IDX: Dict[str, int] = {
    cls_name: i for i, cls_name in enumerate(CLASSIFICATION_CLASSES)
}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for cls_name, i in CLASS_TO_IDX.items()}


def scan_classification_images(root_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scan Dataset A folder and extract metadata with SHA-256 content hashes.

    Args:
        root_dir: Path to data/raw/classification/.

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries containing
            image_path, class_name, class_id, and hash_id.
    """
    root_path = Path(root_dir)
    records: List[Dict[str, Any]] = []

    for cls_name in CLASSIFICATION_CLASSES:
        cls_dir = root_path / cls_name
        if not cls_dir.exists():
            continue

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".mpo"]:
                file_hash = compute_file_hash(img_path)
                records.append(
                    {
                        "image_path": str(img_path.resolve()),
                        "relative_path": str(
                            img_path.relative_to(root_path.parent.parent)
                        ),
                        "filename": img_path.name,
                        "class_name": cls_name,
                        "class_id": CLASS_TO_IDX[cls_name],
                        "hash_id": file_hash,
                    }
                )

    return records


def create_classification_splits(
    root_dir: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create reproducible, group-aware stratified train/val/test splits.

    Detects duplicate images by content hash and ensures that all duplicates of
    the same image content belong strictly to the SAME split. This completely prevents
    data leakage across splits while preserving all 885 images.

    Args:
        root_dir: Path to raw classification directory.
        output_dir: Path to directory where split manifest CSVs will be stored.
        train_ratio: Proportion of groups for training split.
        val_ratio: Proportion of groups for validation split.
        test_ratio: Proportion of groups for test split.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict[str, Any]: Split statistics including image counts and class distributions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = scan_classification_images(root_dir)
    if not records:
        raise ValueError(f"No valid classification images found in {root_dir}")

    # Group records by hash
    hash_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        hash_to_records[rec["hash_id"]].append(rec)

    # Detect duplicates
    duplicate_groups = {h: recs for h, recs in hash_to_records.items() if len(recs) > 1}
    total_duplicate_files = sum(len(recs) - 1 for recs in duplicate_groups.values())

    # Map each group (hash) to a primary class for stratified splitting
    groups_by_class: Dict[str, List[str]] = defaultdict(list)
    for h, recs in hash_to_records.items():
        # Select the majority or first class
        class_counts = Counter(r["class_name"] for r in recs)
        primary_class = class_counts.most_common(1)[0][0]
        groups_by_class[primary_class].append(h)

    rng = random.Random(seed)
    train_hashes: Set[str] = set()
    val_hashes: Set[str] = set()
    test_hashes: Set[str] = set()

    for cls_name, hash_list in sorted(groups_by_class.items()):
        # Deterministically shuffle the unique groups of this class
        shuffled = list(hash_list)
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 group in val and test if n is large enough
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train : n_train + n_val]
        test_part = shuffled[n_train + n_val :]

        train_hashes.update(train_part)
        val_hashes.update(val_part)
        test_hashes.update(test_part)

    # Assign each record to split based on its hash
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for rec in records:
        h = rec["hash_id"]
        is_dup = len(hash_to_records[h]) > 1
        rec["is_duplicate"] = is_dup

        if h in train_hashes:
            rec["split"] = "train"
            train_records.append(rec)
        elif h in val_hashes:
            rec["split"] = "val"
            val_records.append(rec)
        elif h in test_hashes:
            rec["split"] = "test"
            test_records.append(rec)
        else:
            rec["split"] = "train"
            train_records.append(rec)

    # Save manifest CSVs
    fieldnames = [
        "image_path",
        "relative_path",
        "filename",
        "class_name",
        "class_id",
        "hash_id",
        "is_duplicate",
        "split",
    ]

    for split_name, split_recs in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        csv_file = out_path / f"classification_{split_name}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_recs)

    # Verify zero leakage across splits
    leakage_train_val = set(r["hash_id"] for r in train_records) & set(
        r["hash_id"] for r in val_records
    )
    leakage_train_test = set(r["hash_id"] for r in train_records) & set(
        r["hash_id"] for r in test_records
    )
    leakage_val_test = set(r["hash_id"] for r in val_records) & set(
        r["hash_id"] for r in test_records
    )

    assert (
        len(leakage_train_val) == 0
    ), f"Duplicate leakage between train and val: {len(leakage_train_val)}"
    assert (
        len(leakage_train_test) == 0
    ), f"Duplicate leakage between train and test: {len(leakage_train_test)}"
    assert (
        len(leakage_val_test) == 0
    ), f"Duplicate leakage between val and test: {len(leakage_val_test)}"

    stats = {
        "total_images": len(records),
        "unique_content_hashes": len(hash_to_records),
        "duplicate_groups_count": len(duplicate_groups),
        "total_duplicate_files": total_duplicate_files,
        "splits": {
            "train": {
                "total": len(train_records),
                "classes": dict(Counter(r["class_name"] for r in train_records)),
            },
            "val": {
                "total": len(val_records),
                "classes": dict(Counter(r["class_name"] for r in val_records)),
            },
            "test": {
                "total": len(test_records),
                "classes": dict(Counter(r["class_name"] for r in test_records)),
            },
        },
        "leakage_detected": False,
    }

    return stats


class SolarClassificationDataset(Dataset):
    """PyTorch Dataset for Solar Panel Image Classification."""

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize classification dataset.

        Args:
            manifest: Path to manifest CSV, a pandas DataFrame, or list of record dicts.
            transform: Optional torchvision transform to apply to images.
        """
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        elif isinstance(manifest, list):
            self.df = pd.DataFrame(manifest)
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        self.transform = transform

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load image lazily and return tensor with class index.

        Args:
            idx: Integer sample index.

        Returns:
            Tuple[torch.Tensor, int]: (transformed_image_tensor, class_id).
        """
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        class_id = int(row["class_id"])

        image = load_image(img_path, mode="RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, class_id


def get_classification_dataloaders(
    manifest_dir: Union[str, Path],
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders for classification.

    Args:
        manifest_dir: Directory with classification_train.csv, val.csv, test.csv.
        image_size: Input spatial dimension. Defaults to 224.
        batch_size: Mini-batch size. Defaults to 8 (CPU-friendly).
        num_workers: DataLoader background workers. Defaults to 0.
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    manifest_path = Path(manifest_dir)

    train_tf = get_classification_transforms(
        image_size=image_size,
        is_training=True,
        mean=mean,
        std=std,
    )
    eval_tf = get_classification_transforms(
        image_size=image_size,
        is_training=False,
        mean=mean,
        std=std,
    )

    train_ds = SolarClassificationDataset(
        manifest_path / "classification_train.csv", transform=train_tf
    )
    val_ds = SolarClassificationDataset(
        manifest_path / "classification_val.csv", transform=eval_tf
    )
    test_ds = SolarClassificationDataset(
        manifest_path / "classification_test.csv", transform=eval_tf
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
