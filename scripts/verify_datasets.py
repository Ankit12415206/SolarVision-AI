"""
Module: verify_datasets
Description: Comprehensive dataset verification and integrity checking script
Author: SolarVision AI Team
Created: 2026-11-02
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import click
from PIL import Image


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_image_file(path: Path) -> Tuple[bool, str, Tuple[int, int], str]:
    """Check image file integrity, format, and dimensions."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return True, "", img.size, img.format or "UNKNOWN"
    except Exception as e:
        return False, str(e), (0, 0), "CORRUPT"


def verify_dataset_b(root_dir: Path) -> Dict[str, Any]:
    """Verify Dataset B: Aerial Soiling Detection (YOLO format)."""
    results: Dict[str, Any] = {
        "name": "Dataset B — Aerial Soiling Detection",
        "path": str(root_dir),
        "present": root_dir.exists(),
        "total_images": 0,
        "splits": {},
        "class_names": {0: "Dust", 1: "bird-droppings"},
        "instance_counts": Counter(),
        "dimension_counts": Counter(),
        "format_counts": Counter(),
        "corrupt_files": [],
        "duplicates": [],
        "missing_labels": [],
        "empty_labels": 0,
    }

    if not root_dir.exists():
        return results

    hashes: Dict[str, Path] = {}

    for split in ["train", "valid", "test"]:
        split_dir = root_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        split_info = {
            "images": 0,
            "labels": 0,
            "missing_labels": 0,
        }

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            results["total_images"] += 1
            split_info["images"] += 1

            # Integrity check
            ok, err, dim, fmt = check_image_file(img_path)
            if not ok:
                results["corrupt_files"].append(f"{img_path}: {err}")
            else:
                results["dimension_counts"][f"{dim[0]}x{dim[1]}"] += 1
                results["format_counts"][fmt] += 1

            # Duplicate check
            file_hash = compute_file_hash(img_path)
            if file_hash in hashes:
                results["duplicates"].append(
                    f"{img_path.name} duplicate of {hashes[file_hash].name}"
                )
            else:
                hashes[file_hash] = img_path

            # Corresponding label check
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                results["missing_labels"].append(str(img_path.relative_to(root_dir)))
                split_info["missing_labels"] += 1
            else:
                split_info["labels"] += 1
                try:
                    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
                    if not lines:
                        results["empty_labels"] += 1
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            results["instance_counts"][cls_id] += 1
                except Exception as e:
                    results["corrupt_files"].append(f"{lbl_path}: {e}")

        results["splits"][split] = split_info

    return results


def verify_dataset_c(root_dir: Path) -> Dict[str, Any]:
    """Verify Dataset C: MSPS EL Defect Segmentation."""
    results: Dict[str, Any] = {
        "name": "Dataset C — MSPS EL Defect Segmentation",
        "path": str(root_dir),
        "present": root_dir.exists(),
        "total_images": 0,
        "splits": {},
        "dimension_counts": Counter(),
        "format_counts": Counter(),
        "corrupt_files": [],
        "duplicates": [],
        "missing_masks": [],
        "missing_masks01": [],
        "mask_values": set(),
        "mask01_values": set(),
    }

    if not root_dir.exists():
        return results

    hashes: Dict[str, Path] = {}

    for split in ["train", "test"]:
        split_dir = root_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        lbl01_dir = split_dir / "labels01"

        split_info = {
            "images": 0,
            "labels": 0,
            "labels01": 0,
        }

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            results["total_images"] += 1
            split_info["images"] += 1

            # Integrity check
            ok, err, dim, fmt = check_image_file(img_path)
            if not ok:
                results["corrupt_files"].append(f"{img_path}: {err}")
            else:
                results["dimension_counts"][f"{dim[0]}x{dim[1]}"] += 1
                results["format_counts"][fmt] += 1

            # Duplicate check
            file_hash = compute_file_hash(img_path)
            if file_hash in hashes:
                results["duplicates"].append(
                    f"{img_path.name} duplicate of {hashes[file_hash].name}"
                )
            else:
                hashes[file_hash] = img_path

            # Label check (0-255 visual mask)
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                results["missing_masks"].append(str(img_path.relative_to(root_dir)))
            else:
                split_info["labels"] += 1
                # Sample mask values
                if split_info["labels"] <= 50:
                    try:
                        with Image.open(lbl_path) as m_img:
                            results["mask_values"].update(m_img.getextrema() if m_img.mode == 'L' else [])
                    except Exception:
                        pass

            # Label01 check (0-1 binary mask)
            lbl01_path = lbl01_dir / img_path.name
            if not lbl01_path.exists():
                results["missing_masks01"].append(str(img_path.relative_to(root_dir)))
            else:
                split_info["labels01"] += 1
                if split_info["labels01"] <= 50:
                    try:
                        with Image.open(lbl01_path) as m_img:
                            results["mask01_values"].update(m_img.getextrema() if m_img.mode == 'L' else [])
                    except Exception:
                        pass

        results["splits"][split] = split_info

    # Convert sets to sorted lists for JSON serialization
    results["mask_values"] = sorted(list(results["mask_values"]))
    results["mask01_values"] = sorted(list(results["mask01_values"]))

    return results


def verify_dataset_a_status(external_dir: Path, raw_dir: Path) -> Dict[str, Any]:
    """Check Dataset A download and placement status."""
    archive_path = external_dir / "dataset_a_solar_panel_images.zip"
    has_archive = archive_path.exists() and archive_path.stat().st_size > 1000000

    # Check if raw directories have images
    clean_dir = raw_dir / "clean"
    clean_images = (
        len(list(clean_dir.glob("*.jpg"))) + len(list(clean_dir.glob("*.png")))
        if clean_dir.exists()
        else 0
    )

    return {
        "name": "Dataset A — Solar Panel Classification (Kaggle)",
        "source": "https://www.kaggle.com/datasets/pythonafroz/solar-panel-images",
        "archive_present": has_archive,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.exists() else 0,
        "clean_extracted_images": clean_images,
        "expected_classes": [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-Damage",
            "Snow-Covered",
        ],
        "expected_total_images": 875,
        "expected_class_breakdown": {
            "Clean": 194,
            "Bird-drop": 192,
            "Dusty": 191,
            "Snow-Covered": 124,
            "Electrical-damage": 104,
            "Physical-Damage": 70,
        },
        "license": "Unspecified / Web collection",
        "manual_action_required": not has_archive and clean_images == 0,
        "manual_instructions": (
            "1. Visit https://www.kaggle.com/datasets/pythonafroz/solar-panel-images\n"
            "2. Click 'Download' (archive.zip ~305 MB)\n"
            "3. Place archive.zip at 'data/external/dataset_a_solar_panel_images.zip'\n"
            "   OR extract its folders directly into data/raw/."
        ),
    }


@click.command()
@click.option(
    "--output-dir",
    default="reports/dataset_verification",
    help="Directory to save the verification reports.",
)
def main(output_dir: str) -> None:
    """Run comprehensive dataset verification and save report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=== Running SolarVision AI Dataset Verification ===")

    # Verify B
    b_path = Path("data/raw/dust/aerial_soiling")
    print(f"Verifying Dataset B at {b_path}...")
    res_b = verify_dataset_b(b_path)

    # Verify C
    c_path = Path("data/raw/microcracks/msps")
    print(f"Verifying Dataset C at {c_path}...")
    res_c = verify_dataset_c(c_path)

    # Check A
    print("Checking Dataset A status...")
    res_a = verify_dataset_a_status(Path("data/external"), Path("data/raw"))

    # Convert Counters to Dicts for JSON
    res_b["instance_counts"] = dict(res_b["instance_counts"])
    res_b["dimension_counts"] = dict(res_b["dimension_counts"])
    res_b["format_counts"] = dict(res_b["format_counts"])

    res_c["dimension_counts"] = dict(res_c["dimension_counts"])
    res_c["format_counts"] = dict(res_c["format_counts"])

    full_results = {
        "timestamp": "2026-11-02",
        "dataset_a": res_a,
        "dataset_b": res_b,
        "dataset_c": res_c,
    }

    # Save JSON report
    json_path = out_path / "verification_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved JSON report: {json_path}")

    # Save Markdown report
    md_path = out_path / "verification_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SolarVision AI — Dataset Verification Report\n\n")
        f.write(f"> Generated: 2026-11-02 | Step 2: Dataset Research & Acquisition\n\n")

        # Dataset A
        f.write("## 1. Dataset A — Solar Panel Classification (Kaggle)\n\n")
        f.write(f"- **Source:** {res_a['source']}\n")
        f.write(f"- **Download Status:** {'Available' if res_a['archive_present'] else 'Manual action required'}\n")
        f.write(f"- **Expected Total Images:** {res_a['expected_total_images']}\n")
        f.write(f"- **License:** {res_a['license']}\n")
        f.write("- **Expected Class Distribution:**\n")
        for cls_name, count in res_a['expected_class_breakdown'].items():
            f.write(f"  - `{cls_name}`: {count} images\n")
        if res_a['manual_action_required']:
            f.write("\n> ⚠️ **Manual Action Required:**\n")
            f.write("```text\n" + res_a['manual_instructions'] + "\n```\n")

        # Dataset B
        f.write("\n## 2. Dataset B — Aerial Soiling Detection (Roboflow / IA-Cobotics)\n\n")
        f.write(f"- **Present:** {res_b['present']}\n")
        f.write(f"- **Total Images:** {res_b['total_images']}\n")
        f.write(f"- **Corrupt Files:** {len(res_b['corrupt_files'])}\n")
        f.write(f"- **Duplicate Files:** {len(res_b['duplicates'])}\n")
        f.write(f"- **Missing Labels:** {len(res_b['missing_labels'])}\n")
        f.write("- **Splits:**\n")
        for split, info in res_b['splits'].items():
            f.write(f"  - `{split}`: {info['images']} images, {info['labels']} label files\n")
        f.write("- **Dimensions:** " + ", ".join(f"{k} ({v})" for k, v in res_b['dimension_counts'].items()) + "\n")
        f.write("- **Formats:** " + ", ".join(f"{k} ({v})" for k, v in res_b['format_counts'].items()) + "\n")
        f.write("- **Instance Counts (Bounding Boxes):**\n")
        for cls_id, count in res_b['instance_counts'].items():
            cls_name = res_b['class_names'].get(cls_id, f"Class {cls_id}")
            f.write(f"  - `{cls_name}` (ID {cls_id}): {count} bounding boxes\n")
        f.write("- **Annotation Format:** YOLO PyTorch (normalized `class x_center y_center width height`)\n")
        f.write("- **Supports:** Object Detection pipeline\n")
        f.write("- **License:** CC BY 4.0\n")

        # Dataset C
        f.write("\n## 3. Dataset C — MSPS EL Defect Segmentation (Zenodo / GitHub)\n\n")
        f.write(f"- **Present:** {res_c['present']}\n")
        f.write(f"- **Total Images:** {res_c['total_images']}\n")
        f.write(f"- **Corrupt Files:** {len(res_c['corrupt_files'])}\n")
        f.write(f"- **Duplicate Files:** {len(res_c['duplicates'])}\n")
        f.write(f"- **Missing Masks:** {len(res_c['missing_masks'])}\n")
        f.write(f"- **Missing 0/1 Masks:** {len(res_c['missing_masks01'])}\n")
        f.write("- **Splits:**\n")
        for split, info in res_c['splits'].items():
            f.write(f"  - `{split}`: {info['images']} images, {info['labels']} masks, {info['labels01']} binary masks\n")
        f.write("- **Dimensions:** " + ", ".join(f"{k} ({v})" for k, v in res_c['dimension_counts'].items()) + "\n")
        f.write("- **Formats:** " + ", ".join(f"{k} ({v})" for k, v in res_c['format_counts'].items()) + "\n")
        f.write(f"- **Mask Value Ranges:** `labels`={res_c['mask_values']}, `labels01`={res_c['mask01_values']}\n")
        f.write("- **Annotation Format:** Pixel-level binary segmentation masks (0=background, 1/255=defect)\n")
        f.write("- **Supports:** Semantic Segmentation pipeline (U-Net)\n")
        f.write("- **License:** CC BY 4.0\n")

    print(f"Saved Markdown report: {md_path}")
    print("Verification completed successfully!")


if __name__ == "__main__":
    main()
