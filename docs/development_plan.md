# Development Plan — SolarVision AI

> Status: Active | Last updated: 2026-11-02

---

## 1. Project Goals

| Goal                       | Description                                                     |
|----------------------------|-----------------------------------------------------------------|
| **Primary**                | Build a working defect inspection pipeline for solar panels     |
| **Secondary**              | Create an interactive Streamlit dashboard for non-technical users |
| **Tertiary**               | Produce automated PDF inspection reports                        |

---

## 2. Development Steps

### ✅ Step 1 — Project Setup (COMPLETE)
- [x] System hardware & software inspection
- [x] Project directory structure
- [x] Virtual environment (Python 3.12, CPU-only)
- [x] `requirements.txt`, `pyproject.toml`, `.gitignore`
- [x] Git repository initialized
- [x] `AGENTS.md` — development guidelines
- [x] Documentation skeleton (`docs/`)

---

### ✅ Step 2 — Dataset Research & Acquisition
**Goal:** Identify, download, and organize datasets.

- [x] Research and evaluate public solar panel defect datasets
- [x] Download selected datasets into `data/raw/` and `data/external/`
- [x] Explore dataset statistics (class distribution, image sizes, quality)
- [x] Verify annotation formats (Classification, YOLO Detection, MSPS Binary Segmentation)
- [x] Document dataset provenance and licenses
- [x] Create dataset verification scripts and reports

**Deliverables:** Populated `data/raw/`, `docs/dataset_research.md`, verification reports.

---

### ✅ Step 3 — Data Preprocessing Pipeline
**Goal:** Build a reproducible preprocessing and augmentation pipeline.

- [x] Install PyTorch and Torchvision CPU (`torch-2.15.0+cpu`, `torchvision-0.29.2+cpu`)
- [x] Image resizing, RGB/Grayscale validation, and normalization (`src/utils/image.py`, `src/data/transforms.py`)
- [x] Train/val/test splitting with group-aware stratification to eliminate duplicate leakage (`src/data/classification.py`, `src/data/detection.py`, `src/data/segmentation.py`)
- [x] Training augmentations and nearest-neighbor mask transforms (`src/data/transforms.py`)
- [x] PyTorch `Dataset` and `DataLoader` classes for all 3 pipelines (`src/data/`)
- [x] CLI preprocessing runner and split manifest generation (`scripts/preprocess_data.py`)
- [x] 95% unit test coverage with 28 automated tests (`tests/unit/`)

**Deliverables:** `src/data/`, `src/utils/`, `data/processed/splits/`, `reports/dataset_verification/preprocessing_report.md`
- [ ] Config-driven pipeline (YAML)
- [ ] Unit tests for all data modules

**Deliverables:** `src/data/` fully implemented and tested

---

### ✅ Step 4 — Model Architecture Design
**Goal:** Define and implement model architectures for all three tasks.

- [x] Classification models: Small Custom CNN baseline (~422k params) & MobileNetV2 (~2.23M params, frozen/unfrozen)
- [x] Detection model: Lightweight anchor-free SolarYOLOv8n detector (~896k params) with decoupled heads
- [x] Segmentation model: Lightweight U-Net (~1.52M params) for grayscale EL input and binary defect masks
- [x] Numerically stable BCEDiceLoss with positive weight capping for extreme class imbalance
- [x] Shared model utilities: checkpoint save/load, CPU fallback, parameter counters, summaries (`src/utils/model_utils.py`)
- [x] 45 unit tests passing with 95% total code coverage (`tests/unit/`)

**Deliverables:** `src/models/`, `src/utils/model_utils.py`, `configs/default.yaml`, unit test suite.

---

### 🔒 Step 5 — Training Pipeline
**Goal:** Implement training loops for all model types.

- [ ] Generic trainer class (`src/training/trainer.py`)
- [ ] Loss functions: CrossEntropy, Focal Loss, Dice Loss
- [ ] LR schedulers: CosineAnnealing, ReduceLROnPlateau
- [ ] Early stopping and best-model checkpointing
- [ ] Training logs (Loguru + CSV export)
- [ ] Config-driven training (YAML)

**Deliverables:** Working training scripts for all three model types

---

### 🔒 Step 6 — Evaluation & Metrics
**Goal:** Comprehensive model evaluation.

- [ ] Metrics: Accuracy, Precision, Recall, F1, mAP, IoU, Dice
- [ ] Confusion matrix visualization (Plotly)
- [ ] GradCAM / Saliency map visualization
- [ ] PR curve and ROC curve plots
- [ ] Evaluation report generation

**Deliverables:** `src/evaluation/` implemented, evaluation reports

---

### 🔒 Step 7 — Inference Pipeline
**Goal:** Reliable, fast inference on new images.

- [ ] Single image inference (`src/inference/predictor.py`)
- [ ] Batch inference pipeline
- [ ] Result postprocessing (NMS, threshold tuning)
- [ ] ONNX export for deployment optimization (optional)

**Deliverables:** `src/inference/` implemented, inference benchmarks

---

### 🔒 Step 8 — Database & Reporting
**Goal:** Persist inspection results and generate PDF reports.

- [ ] SQLite schema design (`src/database/`)
- [ ] SQLAlchemy ORM models
- [ ] CRUD operations for inspections
- [ ] ReportLab PDF report template
- [ ] Report includes: image, predictions, confidence, timestamp, panel ID

**Deliverables:** `src/database/` implemented, sample PDF report

---

### 🔒 Step 9 — Streamlit Dashboard
**Goal:** Interactive web interface for end users.

- [ ] Upload page: image upload → inference → result display
- [ ] History page: browse past inspections
- [ ] Analytics page: aggregate stats, charts, trends
- [ ] Report page: generate and download PDF
- [ ] Responsive layout, dark-mode compatible

**Deliverables:** `app/` fully implemented, demo screenshots

---

### 🔒 Step 10 — Testing, Documentation & Packaging
**Goal:** Production-quality codebase.

- [ ] Unit tests for all `src/` modules (≥ 80% coverage)
- [ ] Integration tests for full pipeline
- [ ] API documentation (docstrings + README)
- [ ] Final `docs/` update
- [ ] Package build (`pyproject.toml` / wheel)
- [ ] Deployment instructions

**Deliverables:** All tests passing, final documentation, deployable package

---

## 3. PyTorch Installation Instructions

> ⚠️ **Do not install PyTorch until Step 3 begins.**

### For This Machine (CPU-only, Intel, Windows 11)

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Install PyTorch CPU-only (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### For Future CUDA Upgrade (if GPU is added)

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Estimated Timeline

| Step | Effort Estimate  | Cumulative |
|------|-----------------|------------|
| 1    | Setup only      | Done       |
| 2    | 1–2 sessions    | ~2 sessions|
| 3    | 2–3 sessions    | ~5 sessions|
| 4    | 2–3 sessions    | ~8 sessions|
| 5    | 2–3 sessions    | ~11 sessions|
| 6    | 1–2 sessions    | ~13 sessions|
| 7    | 1 session       | ~14 sessions|
| 8    | 1–2 sessions    | ~16 sessions|
| 9    | 2–3 sessions    | ~19 sessions|
| 10   | 2–3 sessions    | ~22 sessions|

> "Session" = one focused work period (1–3 hours). Estimates are approximate.
