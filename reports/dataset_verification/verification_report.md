# SolarVision AI — Dataset Verification Report

> Generated: 2026-11-02 | Step 2: Dataset Research & Acquisition

## 1. Dataset A — Solar Panel Classification (Kaggle)

- **Source:** https://www.kaggle.com/datasets/pythonafroz/solar-panel-images
- **Download Status:** Manual action required
- **Expected Total Images:** 875
- **License:** Unspecified / Web collection
- **Expected Class Distribution:**
  - `Clean`: 194 images
  - `Bird-drop`: 192 images
  - `Dusty`: 191 images
  - `Snow-Covered`: 124 images
  - `Electrical-damage`: 104 images
  - `Physical-Damage`: 70 images

> ⚠️ **Manual Action Required:**
```text
1. Visit https://www.kaggle.com/datasets/pythonafroz/solar-panel-images
2. Click 'Download' (archive.zip ~305 MB)
3. Place archive.zip at 'data/external/dataset_a_solar_panel_images.zip'
   OR extract its folders directly into data/raw/.
```

## 2. Dataset B — Aerial Soiling Detection (Roboflow / IA-Cobotics)

- **Present:** True
- **Total Images:** 302
- **Corrupt Files:** 0
- **Duplicate Files:** 40
- **Missing Labels:** 0
- **Splits:**
  - `train`: 257 images, 257 label files
  - `valid`: 22 images, 22 label files
  - `test`: 23 images, 23 label files
- **Dimensions:** 640x640 (302)
- **Formats:** JPEG (302)
- **Instance Counts (Bounding Boxes):**
  - `Dust` (ID 0): 473 bounding boxes
  - `bird-droppings` (ID 1): 637 bounding boxes
- **Annotation Format:** YOLO PyTorch (normalized `class x_center y_center width height`)
- **Supports:** Object Detection pipeline
- **License:** CC BY 4.0

## 3. Dataset C — MSPS EL Defect Segmentation (Zenodo / GitHub)

- **Present:** True
- **Total Images:** 1507
- **Corrupt Files:** 0
- **Duplicate Files:** 0
- **Missing Masks:** 0
- **Missing 0/1 Masks:** 0
- **Splits:**
  - `train`: 1004 images, 1004 masks, 1004 binary masks
  - `test`: 503 images, 503 masks, 503 binary masks
- **Dimensions:** 640x590 (1507)
- **Formats:** PNG (1507)
- **Mask Value Ranges:** `labels`=[0, 255], `labels01`=[0, 1]
- **Annotation Format:** Pixel-level binary segmentation masks (0=background, 1/255=defect)
- **Supports:** Semantic Segmentation pipeline (U-Net)
- **License:** CC BY 4.0
