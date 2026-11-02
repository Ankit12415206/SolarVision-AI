# SolarVision AI — Object Detection Evaluation Report

> Generated: 2026-11-02 | Step 6: Object Detection Module (Post-Audit Corrected)

## 1. Model Architecture & Scope

- **Architecture:** `SolarYOLOv8n` (Lightweight custom anchor-free YOLO-style detector)
- **Parameters:** 896,165 trainable parameters
- **Hardware Mode:** CPU Execution (Intel Core Ultra)
- **Training Time:** 702.7 seconds (11.7 min)
- **Dataset:** Aerial Soiling & Droppings PV Dataset (Roboflow / IA-Cobotics)
- **Confidence Threshold:** `0.1` | NMS IoU: `0.45`
- **Baseline Checkpoint Preserved:** `checkpoints/detection/yolov8n_baseline.pt`

## 2. Before vs After Corrections Comparison

| Metric | Initial Baseline | Corrected SolarYOLOv8n | Change |
|---|---|---|---|
| **Test mAP@0.5** | 1.91% | **8.57%** | +6.66% |
| **Test mAP@0.5:0.95** | 0.62% | **2.70%** | +2.08% |
| **Dust AP@0.5** | 3.82% | **15.60%** | +11.78% |
| **bird-droppings AP@0.5** | 0.00% | **1.55%** | +1.55% |

## 3. Evaluation Results on Current Splits

| Metric | Validation Set (22 images) | Test Set (23 images) |
|---|---|---|
| **mAP@0.5** | **10.92%** | **8.57%** |
| **mAP@0.5:0.95** | 2.31% | 2.70% |

## 4. Per-Class Test Performance (mAP@0.5)

| Class Name | Ground Truth Count | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| `Dust` | 32 | **15.60%** | 5.06% |
| `bird-droppings` | 48 | **1.55%** | 0.35% |

## 5. Summary of Diagnostic Audit & Corrections Applied

1. **Coordinate Decoding Alignment:** Resolved the center offset mismatch where `postprocess.py` used `sigmoid(dx)` on linear offset targets. Replaced with mathematically consistent linear clamped offsets `clamp(p_box, 0.0, 1.0)`, verified with 1.000000 round-trip IoU.
2. **Balanced Focal Objectness Loss:** Replaced unnormalized BCE (which suffered 480x background gradient overpowering) with a balanced focal BCE loss that normalizes positive and negative contributions separately with safeguards, restoring 1:1 gradient equilibrium.
3. **Conflict Resolution:** Ground-truth targets sorted by area ascending during grid assignment, ensuring tiny defects (droppings) are never overwritten by large diffuse dust clouds.
4. **Dataset Constraints:** 257 training images trained from scratch on CPU without external COCO pre-training. While academic-grade for CPU execution, performance reflects honest baseline capability.
