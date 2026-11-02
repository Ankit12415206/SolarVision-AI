# SolarVision AI — Electroluminescence Segmentation Evaluation Report

> Generated: 2026-11-02 | Step 7: Segmentation Module

## 1. Model Architecture & Scope

- **Architecture:** `SolarLightweightUNet` (Lightweight encoder-decoder with skip connections)
- **Parameters:** 1,524,481 trainable parameters
- **Input Resolution:** $256\times256$ single-channel grayscale EL images
- **Hardware Execution:** CPU Execution (Intel Core Ultra)
- **Training Time:** 1210.7 seconds (20.2 min)
- **Loss Formulation:** Combined `BCEDiceLoss` (50% BCE with pos_weight=15.0 + 50% Soft Dice)
- **Selected Probability Threshold:** `0.70`

## 2. Validation Threshold Search Results (151 images)

| Threshold | Global Dice (%) | Global IoU (%) | Precision (%) | Recall (%) | Pixel Accuracy (%) |
|---|---|---|---|---|---|
| `0.20` | **63.59%** | 46.62% | 49.50% | 88.90% | 99.95% |
| `0.30` | **66.40%** | 49.70% | 53.37% | 87.85% | 99.95% |
| `0.40` | **68.22%** | 51.78% | 56.27% | 86.64% | 99.96% |
| `0.50` | **69.47%** | 53.22% | 58.52% | 85.45% | 99.96% |
| `0.60` | **70.42%** | 54.34% | 60.46% | 84.30% | 99.96% |
| `0.70` **(Selected)** | **71.41%** | 55.53% | 62.50% | 83.27% | 99.97% |

## 3. Official Held-out Test Set Evaluation (503 images)

| Metric | Validation Set (151 images) | Test Set (503 images) |
|---|---|---|
| **Global Dice (F1)** | **71.41%** | **73.54%** |
| **Global IoU (Jaccard)** | 55.53% | **58.16%** |
| **Global Precision** | 62.50% | 64.02% |
| **Global Recall** | 83.27% | 86.39% |
| **Pixel Accuracy** | 99.97% | 99.97% |
| **Mean Per-Image Dice** | 73.55% | 74.43% |
| **Mean Per-Image IoU** | 60.72% | 61.36% |

## 4. Defect Characterization & Limitations

- **Defect Scope:** The MSPS dataset annotations encompass micro-cracks, finger/busbar interruptions, and inactive black spot regions on monocrystalline and polycrystalline PV cells. In accordance with dataset annotations, segmentation classifies all anomalous cell regions.
- **Severe Pixel Imbalance:** Defect pixels represent a fraction of total solar cell surface area ($<1\%$). The combined BCE + Dice loss with positive weighting stabilized gradient propagation without gradient collapse.
- **Visual Inspection:** Multi-panel qualitative comparison overlays saved in `reports/segmentation/visualizations/` depicting: (1) Original Grayscale EL, (2) Ground Truth Mask, (3) Predicted Binary Mask, and (4) Color-coded Hit/False-Alarm/Miss Error Overlay.
