"""
Module: download_datasets.py
Description: Download the three raw dataset archives into data/external/
Author: SolarVision AI Team
Created: 2026-11-02
"""

import os
import sys
import time
import requests
import re
from pathlib import Path

DATA_EXTERNAL = Path("data/external")
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def download_file(url, target_path, session=None):
    if session is None:
        session = requests.Session()
    
    print(f"Downloading from {url[:80]}... to {target_path}")
    start = time.time()
    
    # Handle GDrive confirmation if needed
    if 'drive.google.com' in url or 'drive.usercontent.google.com' in url:
        r = session.get(url, headers=HEADERS)
        if 'drive.usercontent.google.com' not in r.url and '<form' in r.text:
            inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', r.text))
            action = re.search(r'<form[^>]+action="([^"]+)"', r.text).group(1)
            r = session.get(action, params=inputs, headers=HEADERS, stream=True)
        else:
            r = session.get(url, headers=HEADERS, stream=True)
    else:
        r = session.get(url, headers=HEADERS, stream=True, allow_redirects=True)
    
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB
    
    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    print(f"\r  Downloaded: {mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)", end='', flush=True)
    print()
    elapsed = time.time() - start
    print(f"Finished {target_path.name} in {elapsed:.1f}s ({target_path.stat().st_size / (1024*1024):.2f} MB)")

def main():
    # 1. Dataset B (Aerial Soiling) - ~18 MB
    target_b = DATA_EXTERNAL / "dataset_b_aerial_soiling.zip"
    if not target_b.exists():
        url_b = "https://files.cdn-files-a.com/uploads/8291626/normal_6aa6510870fd9.zip"
        print("=== Downloading Dataset B (Aerial Soiling) ===")
        download_file(url_b, target_b)
    else:
        print(f"Dataset B already exists at {target_b} ({target_b.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Dataset C (MSPS) - ~343 MB
    target_c = DATA_EXTERNAL / "dataset_c_msps.zip"
    if not target_c.exists():
        url_c = "https://drive.google.com/uc?id=1X3F0fwZMI4bk8dXYewu9eBVJDeEz6eoY&export=download"
        print("=== Downloading Dataset C (MSPS Defect Segmentation) ===")
        download_file(url_c, target_c)
    else:
        print(f"Dataset C already exists at {target_c} ({target_c.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Dataset A (Kaggle Solar Panel Images) - ~305 MB
    target_a = DATA_EXTERNAL / "dataset_a_solar_panel_images.zip"
    if not target_a.exists():
        url_a = "https://www.kaggle.com/api/v1/datasets/download/pythonafroz/solar-panel-images"
        print("=== Downloading Dataset A (Kaggle Solar Panel Images) ===")
        download_file(url_a, target_a)
    else:
        print(f"Dataset A already exists at {target_a} ({target_a.stat().st_size / (1024*1024):.2f} MB)")

    print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
