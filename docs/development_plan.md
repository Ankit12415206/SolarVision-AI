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

### ✅ Step 5 — Classification Training & Evaluation
**Goal:** Implement training loops, callbacks, train Custom CNN and MobileNetV2, and evaluate on test set.

- [x] Reusable ClassificationTrainer with epoch training/validation loops (`src/training/trainer.py`)
- [x] Early stopping callback and best checkpoint auto-saving (`src/training/callbacks.py`)
- [x] Cosine Annealing learning rate scheduling
- [x] Sequential training on CPU: SolarCustomCNN (12 epochs) & SolarMobileNetV2 (probe + fine-tuning)
- [x] Multi-class metrics: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, per-class report, confusion matrix (`src/evaluation/metrics.py`)
- [x] Plotly interactive training curves and confusion matrix heatmaps (`src/evaluation/visualizer.py`)
- [x] 54 unit tests passing with 96% total code coverage (`tests/unit/`)
- [x] MobileNetV2 achieved 83.33% test accuracy (outperforming Custom CNN by +13.76%)

**Deliverables:** `src/training/`, `src/evaluation/`, `scripts/train_classification.py`, `checkpoints/classification/`, `reports/classification/classification_report.md`

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

### ✅ Step 7 — Segmentation Module
**Goal:** Train and evaluate SolarLightweightUNet on MSPS electroluminescence defect segmentation dataset.

- [x] Input: Single-channel grayscale EL crops ($256\times256$); Target: Binary defect masks
- [x] Combined `BCEDiceLoss` with positive weighting (pos_weight=15.0) and soft Dice
- [x] Dedicated CPU segmentation trainer (`SegmentationTrainer` in `src/training/segmentation_trainer.py`)
- [x] Validation threshold optimization search across candidate thresholds [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
- [x] Optimal threshold selected at 0.70 (Validation Global Dice: 71.41%)
- [x] Official 503-image test split evaluated: 73.54% Global Dice, 58.16% Global IoU, 99.97% Pixel Accuracy
- [x] Qualitative multi-panel error overlays (EL image, GT mask, prediction, hit/miss/false-alarm overlay)
- [x] 67 unit tests passing with 93% coverage across `src/`

**Deliverables:** `src/training/segmentation_trainer.py`, `src/evaluation/segmentation_metrics.py`, `scripts/train_segmentation.py`, `checkpoints/segmentation/unet_best.pt`, `reports/segmentation/`

---

### ✅ Step 8 — Unified Inference Integration
**Goal:** Integrate the three trained models into one reliable, production-ready inference pipeline without retraining.

- [x] Unified inference engine (SolarVisionInferencePipeline in src/inference/pipeline.py)
- [x] Lazy loading & warm model caching for memory conservation (16 GB RAM laptop)
- [x] Strict input validation supporting file paths, PIL Images, and NumPy arrays (2D/3D)
- [x] Standardized InferenceResult schema across Classification, Detection, and Segmentation
- [x] Modality-specific visualizers: classification banner, detection bounding boxes, 3-panel EL composite
- [x] CLI inference script (scripts/run_inference.py) with Click argument parsing and JSON saving
- [x] 87 unit and integration tests passing with 94% test coverage across src/

**Deliverables:** src/inference/, scripts/run_inference.py, tests/unit/test_inference.py, tests/integration/test_inference_pipeline.py

---

### ⏳ Step 9 — Streamlit Dashboard & Analytics
**Goal:** Build a polished, functional, multi-page dashboard around the unified inference pipeline.

- [x] Multi-page interface with styled sidebar (app/main.py)
- [x] Live single-image inspection with preset samples and threshold tuning (app/pages/inspection.py)
- [x] Sequential batch processing with progress tracking, CSV export, and memory conservation (app/pages/batch.py)
- [x] Searchable and filterable SQLite inspection history table (app/pages/history.py)
- [x] Interactive Plotly analytics (defect ratios, modality counts, and timeline trends)
- [x] SQLite database manager and index optimization (src/database/manager.py)
- [x] 100 unit and integration tests passing with 94% test coverage across src/

**Deliverables:** app/, src/database/, tests/unit/test_database.py, tests/integration/test_app_integration.py

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