# SolarVision AI — Data Preprocessing & Split Report

> Generated: 2026-11-02 | Step 3: Data Preprocessing Pipeline

## 1. Dataset A — Classification Preprocessing & Split Integrity

- **Total Images Processed:** 885
- **Unique Content Hashes:** 794
- **Identified Duplicate Files:** 91
- **Duplicate Strategy:** Hash-level grouped stratification (100% data retained, **0% leakage** across splits)
- **Split Counts:** Train=617, Val=130, Test=138
- **Class Distribution per Split:**
  - **TRAIN**: `Bird-drop`: 142, `Clean`: 136, `Dusty`: 133, `Electrical-damage`: 70, `Physical-Damage`: 49, `Snow-Covered`: 87
  - **VAL**: `Bird-drop`: 30, `Clean`: 29, `Dusty`: 27, `Electrical-damage`: 16, `Physical-Damage`: 9, `Snow-Covered`: 19
  - **TEST**: `Bird-drop`: 35, `Clean`: 28, `Dusty`: 30, `Electrical-damage`: 17, `Physical-Damage`: 11, `Snow-Covered`: 17

## 2. Dataset B — Object Detection Splits & Leakage Audit

- **Total Aerial Images:** 302
- **Known Duplicate Images:** 40 groups (80 files) resulting from Roboflow augmentations
- **Cross-Split Duplicate Leakage:** **None** (Train/Val/Test sets are completely isolated)
- **Official Splits Preserved:** Train=257, Valid=22, Test=23
- **Bounding Box Distributions:**
  - **TRAIN**: 257 images, 942 total boxes (Dust=401, Bird-drop=541)
  - **VALID**: 22 images, 88 total boxes (Dust=40, Bird-drop=48)
  - **TEST**: 23 images, 80 total boxes (Dust=32, Bird-drop=48)

## 3. Dataset C — MSPS EL Defect Segmentation Splits & Imbalance

- **Total EL Images & Masks:** 1507
- **Split Allocation:** Train=853, Val=151, Test=503 (Official test split strictly preserved)
- **Severe Defect Imbalance:** Defect pixels constitute **0.0515%** of surface area
- **Recommended BCE Loss Positive Weight (`pos_weight`):** 1939.69
- **Mask Interpolation Protocol:** Strict Nearest-Neighbor interpolation preserving discrete `(0, 1)` defect labels
