"""
Module: detection
Description: Dataset, YOLO parser, and DataLoaders for Aerial Soiling Detection
Author: SolarVision AI Team
Created: 2026-11-02
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.image import compute_file_hash, load_image

DETECTION_CLASSES: Dict[int, str] = {
    0: "Dust",
    1: "bird-droppings",
}


def parse_yolo_label(
    label_path: Union[str, Path],
) -> List[Dict[str, Union[int, float]]]:
    """Parse and validate a YOLO format text label file.

    Format per line: `<class_id> <x_center> <y_center> <width> <height>`
    Coordinates are normalized floats in [0.0, 1.0].

    Args:
        label_path: Path to the .txt YOLO annotation file.

    Returns:
        List[Dict[str, Union[int, float]]]: List of box dicts.
    """
    path = Path(label_path)
    if not path.exists():
        return []

    boxes: List[Dict[str, Union[int, float]]] = []
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Clamp normalized coordinates to [0.0, 1.0]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            boxes.append(
                {
                    "class_id": cls_id,
                    "x_center": cx,
                    "y_center": cy,
                    "width": w,
                    "height": h,
                }
            )

    return boxes


def verify_and_manifest_detection_data(
    root_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Verify Dataset B splits, check for duplicate leakage, and write manifest CSVs.

    Preserves the official splits (train: 257, valid: 22, test: 23).
    Verifies that the 40 known duplicate images do not cross split boundaries.

    Args:
        root_dir: Path to `data/raw/dust/aerial_soiling/`.
        output_dir: Directory where split manifest CSVs will be saved.

    Returns:
        Dict[str, Any]: Summary dictionary with counts, duplicate verification, and instance distributions.
    """
    root_path = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    hashes_by_split: Dict[str, Set[str]] = defaultdict(set)
    hash_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    split_stats: Dict[str, Any] = {}

    for split in ["train", "valid", "test"]:
        split_dir = root_path / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        split_images = sorted(
            [
                p
                for p in img_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        split_recs: List[Dict[str, Any]] = []
        class_instance_counts: Counter = Counter()

        for img_path in split_images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = parse_yolo_label(lbl_path)
            for b in boxes:
                class_instance_counts[int(b["class_id"])] += 1

            f_hash = compute_file_hash(img_path)
            hashes_by_split[split].add(f_hash)

            rec = {
                "image_path": str(img_path.resolve()),
                "relative_path": str(img_path.relative_to(root_path.parent.parent)),
                "label_path": str(lbl_path.resolve()) if lbl_path.exists() else "",
                "filename": img_path.name,
                "split": split,
                "hash_id": f_hash,
                "num_boxes": len(boxes),
                "dust_boxes": sum(1 for b in boxes if b["class_id"] == 0),
                "bird_boxes": sum(1 for b in boxes if b["class_id"] == 1),
            }
            split_recs.append(rec)
            hash_to_records[f_hash].append(rec)

        split_stats[split] = {
            "images": len(split_recs),
            "total_boxes": sum(r["num_boxes"] for r in split_recs),
            "dust_boxes": class_instance_counts[0],
            "bird_boxes": class_instance_counts[1],
        }
        records.extend(split_recs)

    # Cross-split duplicate leakage verification
    leakage_train_val = hashes_by_split["train"] & hashes_by_split["valid"]
    leakage_train_test = hashes_by_split["train"] & hashes_by_split["test"]
    leakage_val_test = hashes_by_split["valid"] & hashes_by_split["test"]

    has_leakage = bool(leakage_train_val or leakage_train_test or leakage_val_test)

    # Mark duplicate flags
    for rec in records:
        rec["is_duplicate"] = len(hash_to_records[rec["hash_id"]]) > 1

    # Save manifests
    fieldnames = [
        "image_path",
        "relative_path",
        "label_path",
        "filename",
        "split",
        "hash_id",
        "is_duplicate",
        "num_boxes",
        "dust_boxes",
        "bird_boxes",
    ]

    for split in ["train", "valid", "test"]:
        manifest_name = "val" if split == "valid" else split
        csv_file = out_path / f"detection_{manifest_name}.csv"
        split_data = [r for r in records if r["split"] == split]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)

    dup_groups = {h: recs for h, recs in hash_to_records.items() if len(recs) > 1}

    return {
        "total_images": len(records),
        "unique_content_hashes": len(hash_to_records),
        "duplicate_groups_count": len(dup_groups),
        "total_duplicate_files": sum(len(recs) - 1 for recs in dup_groups.values()),
        "splits": split_stats,
        "leakage_train_valid": len(leakage_train_val),
        "leakage_train_test": len(leakage_train_test),
        "leakage_valid_test": len(leakage_val_test),
        "leakage_detected": has_leakage,
    }


class SolarDetectionDataset(Dataset):
    """PyTorch Dataset for Aerial Soiling Object Detection with YOLO annotations."""

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]],
        image_size: int = 640,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize detection dataset.

        Args:
            manifest: Path to detection manifest CSV or DataFrame.
            image_size: Target square image dimension. Defaults to 640.
            transform: Optional image transform.
        """
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        elif isinstance(manifest, list):
            self.df = pd.DataFrame(manifest)
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        self.image_size = (image_size, image_size)
        self.transform = transform

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load image and annotations lazily.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - image_tensor: FloatTensor of shape (3, H, W).
                - target: Dict with 'boxes' (N, 4) in normalized [cx, cy, w, h] format,
                          'labels' (N,) LongTensor, and 'image_id'.
        """
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        lbl_path = (
            Path(row["label_path"])
            if pd.notna(row["label_path"]) and row["label_path"]
            else None
        )

        image = load_image(img_path, mode="RGB")
        orig_w, orig_h = image.size

        # Image transform / resize
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Default resize & tensor conversion
            resized = image.resize(self.image_size, Image.BILINEAR)
            image_tensor = transforms.ToTensor()(resized)

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []

        if lbl_path and lbl_path.exists():
            parsed_boxes = parse_yolo_label(lbl_path)
            for b in parsed_boxes:
                boxes_list.append(
                    [b["x_center"], b["y_center"], b["width"], b["height"]]
                )
                labels_list.append(int(b["class_id"]))

        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        return image_tensor, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Collate function for object detection batches with variable box counts.

    Args:
        batch: List of tuples (image_tensor, target_dict).

    Returns:
        Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
            - Batched images tensor of shape (B, 3, H, W).
            - List of target dicts for each image in the batch.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def get_detection_dataloaders(
    manifest_dir: Union[str, Path],
    image_size: int = 640,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, valid, and test DataLoaders for object detection.

    Args:
        manifest_dir: Directory with detection train/val/test CSV manifests.
        image_size: Input spatial dimension. Defaults to 640.
        batch_size: Mini-batch size. Defaults to 4 (CPU and memory safe).
        num_workers: DataLoader background workers. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    manifest_path = Path(manifest_dir)

    train_ds = SolarDetectionDataset(
        manifest_path / "detection_train.csv", image_size=image_size
    )
    val_ds = SolarDetectionDataset(
        manifest_path / "detection_val.csv", image_size=image_size
    )
    test_ds = SolarDetectionDataset(
        manifest_path / "detection_test.csv", image_size=image_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
