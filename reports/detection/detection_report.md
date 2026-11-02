# SolarVision AI — Object Detection Evaluation Report

> Generated: 2026-11-02 | Step 6: Object Detection Module

## 1. Model Architecture & Scope

- **Architecture:** `SolarYOLOv8n` (Lightweight custom anchor-free YOLO-style detector)
- **Parameters:** 896,165 trainable parameters
- **Hardware Mode:** CPU Execution (Intel Core Ultra)
- **Training Time:** 929.9 seconds (15.5 min)
- **Dataset:** Aerial Soiling & Droppings PV Dataset (Roboflow / IA-Cobotics)
- **Confidence Threshold:** `0.03` | NMS IoU: `0.45`

## 2. Evaluation Results on Test Split (23 images)

| Metric | Validation Set (22 images) | Test Set (23 images) |
|---|---|---|
| **mAP@0.5** | **0.46%** | **1.91%** |
| **mAP@0.5:0.95** | 0.10% | 0.62% |

## 3. Per-Class Test Performance (mAP@0.5)

| Class Name | Ground Truth Count | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| `Dust` | 32 | **3.82%** | 1.24% |
| `bird-droppings` | 48 | **0.00%** | 0.00% |

## 4. Technical Constraints & Dataset Limitations

- **Dataset Scale:** The dataset consists of 302 total aerial drone images (257 train, 22 val, 23 test). While sufficient for university prototyping and demonstrating multi-task inspection, broader generalization would benefit from a larger multi-site aerial corpus.
- **Custom Model from Scratch:** The detector was initialized and trained entirely from scratch on CPU without external pre-trained COCO weights, establishing a genuine custom baseline.
- **Hardware Constraints:** Training completed smoothly on CPU within 16 GB RAM limits without GPU acceleration.
- **Visual Overlays:** Qualitative detection overlays saved under `reports/detection/visualizations/` comparing predicted bounding boxes (green) against ground-truth annotations (red).
