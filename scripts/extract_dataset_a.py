"""
Module: extract_dataset_a.py
Description: Extract and organize Dataset A (Solar Panel Classification) into data/raw/
Author: SolarVision AI Team
Created: 2026-11-02
"""

import shutil
import zipfile
from pathlib import Path

EXTERNAL_ZIP = Path("data/external/dataset_a_solar_panel_images.zip")
RAW_DIR = Path("data/raw")
CLASSIF_DIR = RAW_DIR / "classification"
CLEAN_DIR = RAW_DIR / "clean"

print(f"Extracting {EXTERNAL_ZIP}...")

# Target classes
classes = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

extracted_counts = {c: 0 for c in classes}

with zipfile.ZipFile(EXTERNAL_ZIP, 'r') as z:
    for member in z.infolist():
        if member.filename.endswith('/') or member.filename.endswith('.ini'):
            continue
        parts = member.filename.split('/')
        if len(parts) >= 3 and parts[0] == 'Faulty_solar_panel':
            cls_name = parts[1]
            if cls_name in classes:
                dest = CLASSIF_DIR / cls_name / parts[-1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted_counts[cls_name] += 1
                
                # Also place clean images in data/raw/clean/
                if cls_name == "Clean":
                    clean_dest = CLEAN_DIR / parts[-1]
                    clean_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, clean_dest)

print("Extraction summary into data/raw/classification/:")
for c, cnt in sorted(extracted_counts.items()):
    print(f"  {c}: {cnt} images")
print(f"Populated data/raw/clean/ with {extracted_counts['Clean']} images.")
print("Total extracted images for Dataset A:", sum(extracted_counts.values()))
