# Architecture — SolarVision AI

> Status: Draft (Step 1) | Last updated: 2026-11-02

---

## 1. System Overview

SolarVision AI is a **multi-task deep learning inspection system** that processes drone-captured images of solar panel arrays to detect and classify surface defects — primarily dust accumulation and micro-crack patterns.

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Layer                            │
│              Drone Images (JPG/PNG/TIFF)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Preprocessing Pipeline                       │
│   Resize → Normalize → Augment → Split (train/val/test)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Classification │ │  Detection   │ │  Segmentation    │
│  (ResNet/EfficientNet)│(YOLOv8/Faster-RCNN)│(U-Net / DeepLab)│
│  Dust vs Clean  │ │ Bounding Box │ │  Pixel Masks     │
│  vs Micro-crack │ │ per defect   │ │  per defect type │
└────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
          │                │                  │
          └────────────────┼──────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Inference & Reporting                       │
│         Confidence scores + Defect localization             │
│             → SQLite database logging                       │
│             → PDF report generation (ReportLab)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Streamlit Dashboard (app/)                    │
│   Upload → Inspect → Visualize → Export → History          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Module Breakdown

### 2.1 `src/data/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `dataset.py`          | `SolarDataset` class (PyTorch Dataset)              |
| `transforms.py`       | Image augmentation pipelines (Albumentations/torchvision) |
| `splitter.py`         | Train/val/test split logic (stratified)             |
| `loader.py`           | DataLoader factory with config-driven batch sizing  |
| `preprocessing.py`    | Resize, normalize, color space conversion           |

### 2.2 `src/models/`

#### `classification/`
- **Task:** 6-class solar panel surface defect classification (`Bird-drop`, `Clean`, `Dusty`, `Electrical-damage`, `Physical-Damage`, `Snow-Covered`)
- **Architectures Implemented:**
  - `SolarCustomCNN`: 4-stage convolutional baseline with BatchNorm, ReLU, and Dropout (~422k parameters). Fast CPU inference baseline.
  - `SolarMobileNetV2`: Transfer learning backbone with pre-trained ImageNet weights (~2.23M parameters). Supports frozen linear probe (7.6k trainable), partially unfrozen (fine-tuning N inverted residual blocks), and fully unfrozen modes.
- **Output:** Raw logits for 6 categories (evaluated with CrossEntropyLoss).

#### `detection/`
- **Task:** Locate soiling and bird droppings with normalized bounding boxes (`[cx, cy, w, h]`)
- **Architecture Implemented:**
  - `SolarYOLOv8n`: Anchor-free lightweight detector with C2f cross-stage partial blocks and decoupled multi-scale detection heads at strides P3 (8x), P4 (16x), and P5 (32x) (~896k parameters).
- **Output:** Multi-scale bounding box coordinates + objectness + class scores (2 classes: `Dust`, `bird-droppings`).

#### `segmentation/`
- **Task:** Pixel-level binary segmentation of structural and micro-cracks in electroluminescence (EL) cells
- **Architecture Implemented:**
  - `SolarLightweightUNet`: Double-convolution encoder-decoder with skip connections and bilinear upsampling (~1.52M parameters). Input: 1-channel grayscale EL; Output: 1-channel raw binary logits.
- **Loss Function:** `BCEDiceLoss` combining weighted `BCEWithLogitsLoss` and soft `DiceLoss`. Features positive weight moderation (capping extreme raw imbalance to prevent gradient explosions) and Laplace smoothing.

### 2.3 `src/training/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `trainer.py`          | Generic training loop (supports all three tasks)    |
| `losses.py`           | Custom loss functions (Focal Loss, Dice Loss)       |
| `schedulers.py`       | LR schedulers (CosineAnnealing, ReduceLROnPlateau)  |
| `callbacks.py`        | Early stopping, checkpoint saving, logging          |

### 2.4 `src/evaluation/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `metrics.py`          | Accuracy, Precision, Recall, F1, mAP, IoU          |
| `visualizer.py`       | Confusion matrices, GradCAM, PR curves (Plotly)    |
| `evaluator.py`        | Full evaluation pipeline runner                    |

### 2.5 `src/inference/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `predictor.py`        | Load model + run inference on single image/batch   |
| `pipeline.py`         | End-to-end image → result pipeline                 |
| `postprocessing.py`   | NMS, threshold filtering, mask refinement          |

### 2.6 `src/utils/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `config.py`           | YAML config loader / validator                     |
| `logger.py`           | Loguru-based logging setup                         |
| `seed.py`             | Global reproducibility seed setter                 |
| `image_utils.py`      | Image I/O helpers (OpenCV + Pillow wrappers)       |
| `paths.py`            | Centralized pathlib.Path constants                 |

### 2.7 `src/database/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `models.py`           | SQLAlchemy ORM models (Inspection, Panel, Result)  |
| `db.py`               | Database engine creation, session management       |
| `crud.py`             | Create/read/update/delete helpers                  |
| `schema.sql`          | Raw schema reference (documentation only)          |

### 2.8 `app/`
| File (planned)        | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `main.py`             | Streamlit entrypoint, routing                      |
| `pages/upload.py`     | Image upload & single inspection page              |
| `pages/history.py`    | Inspection history browser                         |
| `pages/analytics.py`  | Aggregate statistics & charts                      |
| `pages/report.py`     | PDF report generation trigger                      |
| `components/`         | Reusable Streamlit UI components                   |

---

## 3. Data Flow

```
Raw Drone Image
    │
    ▼ (preprocessing)
Normalized Tensor [B, C, H, W]
    │
    ├──► Classification Model → Class Label + Confidence
    │
    ├──► Detection Model → [(x1,y1,x2,y2, class, conf), ...]
    │
    └──► Segmentation Model → Mask [H, W] per class
             │
             ▼
    Aggregate Results Dict
             │
    ┌────────┴────────┐
    ▼                 ▼
 SQLite DB        PDF Report
 (history)        (export)
```

---

## 4. Technology Decisions

| Decision                  | Choice              | Rationale                                         |
|---------------------------|---------------------|---------------------------------------------------|
| Deep Learning Framework   | PyTorch             | Industry standard, flexible, excellent ecosystem  |
| CV Library                | OpenCV + Pillow     | Complementary: OpenCV for transforms, Pillow for I/O |
| Classification Backbone   | EfficientNet-B0     | Highest accuracy/parameter ratio, CPU-friendly    |
| Detection Architecture    | YOLOv8-nano         | Real-time capable, smallest YOLO variant          |
| Segmentation Architecture | U-Net               | Proven for small datasets, pixel-accurate         |
| Dashboard                 | Streamlit           | Fast prototyping, Python-native                   |
| Database                  | SQLite              | Zero-config, file-based, sufficient for this scale|
| Report Generation         | ReportLab           | Pure Python, no external dependencies             |

---

## 5. Hardware Target

| Component | Spec (detected)           |
|-----------|---------------------------|
| OS        | Windows 11 (10.0.26300)   |
| CPU       | Intel Core (22 logical CPUs, Family 6 Model 170) |
| GPU       | None detected (CPU-only mode) |
| Python    | 3.12.4                    |

> **All models must run on CPU.** GPU acceleration is a future enhancement.

---

## 6. Open Decisions (to be resolved in later steps)

- [ ] Exact pretrained model source (timm library vs torchvision.models)
- [ ] Whether to use Albumentations or torchvision.transforms v2 for augmentation
- [ ] Multi-task learning (shared encoder) vs separate models per task
- [ ] Dataset format: COCO JSON vs YOLO TXT vs custom CSV
- [ ] Inference batch size for CPU (memory vs throughput tradeoff)
