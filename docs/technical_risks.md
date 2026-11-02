# Technical Risks — SolarVision AI

> Status: Active | Last updated: 2026-11-02

---

## 1. Overview

This document tracks identified technical risks, their likelihood, impact, and
current mitigation strategies. Update this document whenever a new risk is
identified or a mitigation is applied.

**Risk matrix:**
- 🔴 High likelihood + High impact
- 🟡 Medium likelihood or Medium impact
- 🟢 Low likelihood or Low impact

---

## 2. Risk Register

### RISK-001 — No CUDA/GPU Available
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🔴 Confirmed (hardware inspection)                           |
| **Description**| No NVIDIA GPU detected. All training runs on CPU only.       |
| **Impact**     | Training times will be 10–50× slower than GPU training. Large models (YOLOv8-m, ResNet-101) may be impractical. |
| **Mitigation** | Use smallest model variants (EfficientNet-B0, YOLOv8-nano, U-Net with MobileNetV2 encoder). Use pretrained weights. Limit epochs during development. Consider Google Colab or Kaggle Notebooks for heavy training. |
| **Owner**      | Step 4–5                                                     |

---

### RISK-002 — Python 3.11 Not Installed (3.12 Used Instead)
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Confirmed (hardware inspection)                           |
| **Description**| Only Python 3.12.4 found. Project specifies Python 3.11.    |
| **Impact**     | Minor — Python 3.12 is fully backward compatible with all dependencies specified. Some very new PyTorch features may behave differently, but core functionality is unaffected. |
| **Mitigation** | Virtual environment uses Python 3.12. `pyproject.toml` lists `requires-python = ">=3.11"` to remain compatible. If strict 3.11 is needed, user can install it from python.org and recreate `.venv`. |
| **Owner**      | Step 1 (documented)                                          |

---

### RISK-003 — Small Dataset Size
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🔴 High Risk (to be confirmed in Step 2)                    |
| **Description**| Public solar panel defect datasets are small (<5,000 images). Deep learning typically requires thousands of labeled samples. |
| **Impact**     | Overfitting, poor generalization, unreliable model predictions. |
| **Mitigation** | Transfer learning from ImageNet pretrained weights. Heavy data augmentation (rotation, flips, color jitter, noise). Synthetic data generation if needed. Combine multiple datasets. |
| **Owner**      | Step 2–3                                                     |

---

### RISK-004 — Class Imbalance
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Likely (to be confirmed after dataset inspection)        |
| **Description**| Clean panels may vastly outnumber defective ones in real datasets. |
| **Impact**     | Model biased toward predicting "clean" — high accuracy but low recall for defects. |
| **Mitigation** | Weighted sampling in DataLoader. Focal Loss instead of CrossEntropy. Oversampling minority classes (SMOTE-style or duplication). Report per-class metrics (F1, recall) not just accuracy. |
| **Owner**      | Step 3–5                                                     |

---

### RISK-005 — Annotation Quality & Consistency
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Medium Risk                                               |
| **Description**| Public datasets may have inconsistent annotation styles, label noise, or missing annotations. |
| **Impact**     | Training on noisy labels reduces model performance.          |
| **Mitigation** | Manual inspection of a sample of annotations. Label smoothing during training. Outlier detection in dataset exploration notebooks. |
| **Owner**      | Step 2–3                                                     |

---

### RISK-006 — Domain Gap (EL vs RGB Drone Images)
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Medium Risk                                               |
| **Description**| Best-labeled datasets (ELPV, PVDN) use Electroluminescence (EL) imaging; actual deployment uses RGB drone cameras. |
| **Impact**     | Models trained on EL may not generalize to RGB imagery.      |
| **Mitigation** | Prioritize RGB datasets. If EL data is used, apply domain adaptation techniques or keep EL and RGB models separate. Document which datasets are EL vs RGB. |
| **Owner**      | Step 2–4                                                     |

---

### RISK-007 — OneDrive Sync Interference
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Observed during setup                                    |
| **Description**| Project is located in OneDrive (`Desktop\SolarVision-AI`). OneDrive sync may lock files during writes, interfering with training checkpoints and large data files. |
| **Impact**     | File permission errors during training. Accidental cloud upload of large model weights/datasets (slow, expensive). |
| **Mitigation** | Add `data/`, `checkpoints/`, `logs/` to OneDrive exclusions. Consider moving project to a local non-OneDrive path (e.g., `C:\Projects\SolarVision-AI`) before beginning heavy training. |
| **Owner**      | Step 1 (flagged) — resolve before Step 3                    |

---

### RISK-008 — Disk Space for Datasets and Models
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟡 Unknown (PSDrive showed 0 GB — likely a sandbox limitation) |
| **Description**| Dataset images + pretrained model weights + checkpoints can collectively require 5–20 GB of disk space. |
| **Impact**     | Insufficient disk space halts training and data download.    |
| **Mitigation** | Verify actual free disk space before Step 2 (run `df -h` or check in File Explorer). Use compressed datasets where possible. Clean up intermediate files regularly. |
| **Owner**      | Step 2                                                       |

---

### RISK-009 — Streamlit Deployment Limitations
| Field          | Value                                                        |
|----------------|--------------------------------------------------------------|
| **Status**     | 🟢 Low Risk (future concern)                                 |
| **Description**| Streamlit Community Cloud has file size limits and no GPU access. Large model inference may be slow. |
| **Impact**     | Dashboard may be slow or unusable in a deployed cloud context.|
| **Mitigation** | Optimize model size (ONNX export). Cache inference results. Offer local deployment as the primary mode. |
| **Owner**      | Step 9                                                       |

---

## 3. Resolved Risks

_(None yet — will be moved here as risks are mitigated.)_

---

## 4. Risk Update Log

| Date       | Risk ID  | Update                                      |
|------------|----------|---------------------------------------------|
| 2026-11-02 | RISK-001 | Confirmed: no GPU, CPU-only mode selected   |
| 2026-11-02 | RISK-002 | Confirmed: Python 3.12 used, documented     |
| 2026-11-02 | RISK-007 | Observed: OneDrive path requires bypass     |
