# Dataset Research — SolarVision AI

> Status: Draft (Step 1) | To be completed in Step 2

---

## 1. Overview

This document catalogs candidate datasets for training and evaluating the
SolarVision AI defect detection system. Two primary defect classes are targeted:

1. **Dust / soiling** — Particulate matter reducing panel efficiency
2. **Micro-cracks** — Structural cell fractures (often invisible to the naked eye)

---

## 2. Candidate Public Datasets

### 2.1 ELPV Dataset
| Field         | Details                                                    |
|---------------|------------------------------------------------------------|
| **Name**      | ELPV (Electroluminescence PV) Dataset                      |
| **Source**    | GitHub: [Buerhop-Lutz et al.](https://github.com/zae-bayern/elpv-dataset) |
| **Images**    | 2,624 EL images of solar cells                            |
| **Defects**   | Functional / defective classification per cell            |
| **Format**    | PNG, CSV labels                                           |
| **License**   | CC BY-NC-SA 4.0                                           |
| **Notes**     | EL imaging (infrared), not drone RGB — may need adaptation|

### 2.2 Solar Panel Dust Dataset (Roboflow)
| Field         | Details                                                    |
|---------------|------------------------------------------------------------|
| **Name**      | Solar Panel Dust Detection                                |
| **Source**    | Roboflow Universe                                          |
| **Images**    | ~500–1000 (varies by version)                             |
| **Defects**   | Dust accumulation bounding boxes                          |
| **Format**    | YOLO TXT, COCO JSON                                       |
| **License**   | CC BY 4.0 (check per dataset)                             |
| **Notes**     | RGB drone imagery; directly applicable                    |

### 2.3 Photovoltaic Cell Defect Dataset (Kaggle)
| Field         | Details                                                    |
|---------------|------------------------------------------------------------|
| **Name**      | Solar Panel Image Dataset                                 |
| **Source**    | Kaggle                                                    |
| **Images**    | ~1,000–5,000                                              |
| **Defects**   | Dirty / clean classification                              |
| **Format**    | JPEG, folder-based class labels                           |
| **License**   | Various (check before use)                                |
| **Notes**     | Classification-ready, no segmentation masks               |

### 2.4 PVDN Dataset (PV Defect Detection)
| Field         | Details                                                    |
|---------------|------------------------------------------------------------|
| **Name**      | PVDN — Photovoltaic Defect Dataset                        |
| **Source**    | [TU Delft / Zenodo](https://zenodo.org/)                  |
| **Images**    | EL images with pixel-level annotations                    |
| **Defects**   | Cracks, breaks, finger interruptions                      |
| **Format**    | PNG + JSON masks                                          |
| **License**   | CC BY 4.0                                                 |
| **Notes**     | High-quality segmentation masks available                 |

### 2.5 Synthetic / Augmented Data Strategy
If real datasets are too small:
- Use **Stable Diffusion / GAN-based augmentation** to generate synthetic defective panel images.
- Apply **Albumentations** for geometric and photometric augmentation.
- Minimum recommended dataset size: **1,000 images per class** for fine-tuning pretrained CNNs.

---

## 3. Dataset Requirements

| Requirement                  | Target                             |
|------------------------------|------------------------------------|
| Minimum images (train)       | 800 per class                      |
| Minimum images (val)         | 200 per class                      |
| Minimum images (test)        | 100 per class (held-out)           |
| Image resolution             | ≥ 224×224 px (resize if needed)   |
| Annotation format            | YOLO TXT for detection; PNG masks for segmentation |
| Class balance                | Aim for ≤ 2:1 imbalance ratio      |

---

## 4. Data Acquisition Plan (Step 2 Actions)

- [ ] Download ELPV dataset from GitHub
- [ ] Download solar dust datasets from Roboflow
- [ ] Search Kaggle for additional solar panel imagery
- [ ] Evaluate dataset sizes and class distributions
- [ ] Decide on a unified annotation format
- [ ] Design data augmentation strategy
- [ ] Document data provenance and licenses in `data/README.md`

---

## 5. Licensing Summary

> ⚠️ All datasets must be reviewed for license compatibility before use.
> Commercial use may be restricted. Store license files in `data/external/`.

---

## 6. References

- Buerhop-Lutz, C. et al. (2018). ELPV Dataset. GitHub.
- Roboflow Universe: https://universe.roboflow.com
- Kaggle Datasets: https://www.kaggle.com/datasets
- Zenodo (PVDN): https://zenodo.org
- Akram, M. W. et al. (2019). A review of artificial intelligence based image
  processing for solar PV defect detection. *Solar Energy*, 207, 170–182.
