# Dataset Research & Acquisition — SolarVision AI

> Status: Completed (Step 2) | Last updated: 2026-11-02

---

## 1. Overview

This document catalogs the three selected datasets acquired and verified for **SolarVision AI**. The project integrates three distinct computer vision pipelines:
1. **Classification:** Clean vs. defective surface identification.
2. **Object Detection:** Bounding-box localization of surface soiling and bird droppings.
3. **Semantic Segmentation:** Pixel-level segmentation of microscopic and structural cracks on solar cells.

---

## 2. Dataset A — Solar Panel Classification

| Field | Specification |
|---|---|
| **Dataset Title** | Solar Panel Images Clean and Faulty Images |
| **Source URL** | https://www.kaggle.com/datasets/pythonafroz/solar-panel-images |
| **Publisher** | User `pythonafroz` on Kaggle (2023) |
| **Download Status** | ✅ **Downloaded & Verified** (Manually acquired archive from Kaggle) |
| **Archive Location** | `data/external/dataset_a_solar_panel_images.zip` (304.93 MB, 319,746,451 bytes) |
| **Extracted Path** | `data/raw/classification/` (6 class folders) and `data/raw/clean/` |
| **Total Images** | 885 images |
| **Image Resolution** | Varied (top resolutions: 720×960, 2048×1152, 1536×2048, 474×355) |
| **Image Format** | JPEG (862), PNG (21), MPO (2) (100% verified, 0 corrupt files) |
| **Class Distribution** | 6 classes (885 images total): <br>• `Bird-drop`: 207 images <br>• `Clean`: 193 images <br>• `Dusty`: 190 images <br>• `Electrical-damage`: 103 images <br>• `Physical-Damage`: 69 images <br>• `Snow-Covered`: 123 images |
| **Annotation Format** | Directory-based classification labels |
| **Supported Pipeline** | **Image Classification Module** (ResNet-50 / EfficientNet-B0) |
| **License** | **Unspecified / Web collection** (Use with standard academic citation) |
| **Limitations** | • Web-scraped imagery with varied resolutions and lighting.<br>• Class imbalance (69 to 207 per class).<br>• 91 duplicate files across classes due to multi-source web collection.<br>• `Physical-Damage` reflects macroscopic fractures/broken glass, **not** electroluminescence micro-cracks. |

---

## 3. Dataset B — Aerial Soiling Detection

| Field | Specification |
|---|---|
| **Dataset Title** | Aerial Soiling & Droppings PV Inspection Dataset (`soiling_dropping-2` v14) |
| **Source URL** | https://www.ia-cobotics.com/research-projects/ai-driven-sola-panel-inspection |
| **Research Reference**| PMC11821171 (https://pmc.ncbi.nlm.nih.gov/articles/PMC11821171/) |
| **Download Status** | ✅ **Downloaded & Verified** |
| **Archive Location** | `data/external/dataset_b_aerial_soiling.zip` (18.28 MB) |
| **Extracted Path** | `data/raw/dust/aerial_soiling/` |
| **Total Images** | 302 RGB aerial images |
| **Image Resolution** | 640 × 640 pixels |
| **Image Format** | JPEG (100% verified, 0 corrupt files) |
| **Official Splits** | • `train`: 257 images, 257 label files <br>• `valid`: 22 images, 22 label files <br>• `test`: 23 images, 23 label files |
| **Annotation Format** | **YOLO v5/v8 PyTorch format** (`<class_id> <x_center> <y_center> <width> <height>` normalized) |
| **Classes & Instances**| 2 classes, **1,110 total bounding box annotations**: <br>• `Dust` (ID 0): 473 bounding boxes <br>• `bird-droppings` (ID 1): 637 bounding boxes |
| **Label Integrity** | 100% image-to-annotation match (0 missing label files) |
| **Supported Pipeline** | **Object Detection Module** (YOLOv8-nano / Faster R-CNN) |
| **License** | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| **Limitations** | • 40 duplicate image contents resulting from Roboflow offline rotations/flips.<br>• Limited to soiling and bird-droppings; does not contain cell cracks. |

---

## 4. Dataset C — MSPS EL Defect Segmentation

| Field | Specification |
|---|---|
| **Dataset Title** | MSPS (Monocrystalline Silicon Photovoltaic Cell Defect Segmentation Dataset) |
| **Source URL** | https://doi.org/10.5281/zenodo.21549573 |
| **GitHub Repo** | https://github.com/wch313/MSPS-Dataset |
| **Download Status** | ✅ **Downloaded & Verified** |
| **Archive Location** | `data/external/dataset_c_msps.zip` (343.72 MB) |
| **Extracted Path** | `data/raw/microcracks/msps/` |
| **Total Images** | 1,507 Electroluminescence (EL) crops |
| **Image Resolution** | 640 × 590 pixels |
| **Image Format** | PNG (100% verified, 0 corrupt files) |
| **Official Splits** | • `train`: 1,004 images, 1,004 visual masks, 1,004 binary masks <br>• `test`: 503 images, 503 visual masks, 503 binary masks |
| **Annotation Format** | **Pixel-level semantic segmentation masks**: <br>• `labels/`: Visual grayscale PNG mask (`[0, 255]`) <br>• `labels01/`: Binary PNG mask (`[0, 1]`) |
| **Defect Types** | Busbar breaks, black spots, and micro-cracks |
| **Label Integrity** | 100% image-to-mask match (0 missing masks, 0 missing binary masks) |
| **Supported Pipeline** | **Semantic Segmentation Module** (U-Net with MobileNetV2 backbone) |
| **License** | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| **Citation** | Zhao et al. (2026), Zenodo DOI: `10.5281/zenodo.21549573` |
| **Limitations** | • EL imaging (near-infrared electroluminescence) rather than RGB drone cameras.<br>• Extreme class imbalance (defect pixels constitute ~0.05% of image area). |

---

## 5. Summary Matrix & Model Pipeline Mapping

| Dataset | Modality | Images | Task | Output Target | License |
|---|---|---|---|---|---|
| **Dataset A** | RGB (Handheld/Drone) | 885 | Classification | 6 Classes (Clean, Dust, Damage, etc.) | Unspecified |
| **Dataset B** | RGB (Aerial Drone) | 302 | Object Detection | Bounding boxes (Dust, Bird-drop) | CC BY 4.0 |
| **Dataset C** | EL (Electroluminescence) | 1,507 | Segmentation | Pixel-wise defect masks (Cracks, Breaks) | CC BY 4.0 |

---

## 6. Storage & Organization Rules

- **Raw Data Immutability:** All files in `data/raw/` are read-only.
- **Separation of Modalities:** RGB drone imagery (`data/raw/dust/aerial_soiling/`) and EL cell crops (`data/raw/microcracks/msps/`) are strictly kept separate.
- **Preserved Archives:** Original zip files are preserved in `data/external/`.
