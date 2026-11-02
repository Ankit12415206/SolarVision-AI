# SolarVision AI — Classification Training & Evaluation Report

> Generated: 2026-11-02 | Step 5: Classification Module

## 1. Overall Performance Comparison

| Model | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **Custom CNN (Baseline)** | Validation | 69.23% | 0.6877 | 0.6465 | 0.6470 | 0.6830 |
| **Custom CNN (Baseline)** | Test | **69.57%** | 0.7584 | 0.6793 | 0.6646 | 0.6809 |
| **MobileNetV2 (Pre-trained)** | Validation | 80.00% | 0.8264 | 0.7821 | 0.7976 | 0.8006 |
| **MobileNetV2 (Pre-trained)** | Test | **83.33%** | 0.8519 | 0.8250 | 0.8354 | 0.8349 |

## 2. Per-Class Test Performance (MobileNetV2)

| Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `Bird-drop` | 0.8333 | 0.8571 | 0.8451 | 35 |
| `Clean` | 0.7576 | 0.8929 | 0.8197 | 28 |
| `Dusty` | 0.7931 | 0.7667 | 0.7797 | 30 |
| `Electrical-damage` | 1.0000 | 0.8235 | 0.9032 | 17 |
| `Physical-Damage` | 0.7273 | 0.7273 | 0.7273 | 11 |
| `Snow-Covered` | 1.0000 | 0.8824 | 0.9375 | 17 |

## 3. Per-Class Test Performance (Custom CNN Baseline)

| Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `Bird-drop` | 0.6875 | 0.6286 | 0.6567 | 35 |
| `Clean` | 0.5833 | 0.7500 | 0.6562 | 28 |
| `Dusty` | 0.7600 | 0.6333 | 0.6909 | 30 |
| `Electrical-damage` | 0.6250 | 0.8824 | 0.7317 | 17 |
| `Physical-Damage` | 1.0000 | 0.1818 | 0.3077 | 11 |
| `Snow-Covered` | 0.8947 | 1.0000 | 0.9444 | 17 |

## 4. Key Findings & Architecture Recommendation

- MobileNetV2 achieved **83.33%** test accuracy, outperforming the Custom CNN baseline (69.57%) by **+13.76%**.
- Pretrained ImageNet representations provide strong feature transfer on solar defect textures.
- Checkpoints are saved under `checkpoints/classification/` for downstream inference and dashboard integration.
