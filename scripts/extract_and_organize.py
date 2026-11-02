"""
Module: extract_and_organize.py
Description: Extract and organize Dataset B and Dataset C into data/raw/
Author: SolarVision AI Team
Created: 2026-11-02
"""

import zipfile
import shutil
from pathlib import Path

DATA_EXTERNAL = Path("data/external")
DATA_RAW = Path("data/raw")

def extract_dataset_b():
    zip_path = DATA_EXTERNAL / "dataset_b_aerial_soiling.zip"
    target_dir = DATA_RAW / "dust" / "aerial_soiling"
    print(f"Extracting {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Check root folder name in zip
        prefix = "Soiling_DS_v2_14/"
        for member in z.infolist():
            if member.filename.startswith(prefix) and not member.filename.endswith('/'):
                rel_path = member.filename[len(prefix):]
                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest_file, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    print(f"Dataset B extracted to {target_dir}")

def extract_dataset_c():
    zip_path = DATA_EXTERNAL / "dataset_c_msps.zip"
    target_dir = DATA_RAW / "microcracks" / "msps"
    print(f"Extracting {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        prefix = "MSPS/"
        for member in z.infolist():
            # Skip macOS hidden files like __MACOSX or .DS_Store if any
            if "__MACOSX" in member.filename or ".DS_Store" in member.filename:
                continue
            if member.filename.startswith(prefix) and not member.filename.endswith('/'):
                rel_path = member.filename[len(prefix):]
                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest_file, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    print(f"Dataset C extracted to {target_dir}")

def main():
    extract_dataset_b()
    extract_dataset_c()
    print("Extraction and organization complete!")

if __name__ == "__main__":
    main()
