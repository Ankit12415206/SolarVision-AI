"""
Module: visualizer
Description: Plotting and report generation utilities using Plotly
Author: SolarVision AI Team
Created: 2026-11-02
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "Training & Validation Metrics",
) -> Path:
    """Generate dual-axis training curves for loss and accuracy using Plotly.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Path where HTML plot will be saved.
        title: Plot title string.

    Returns:
        Path: Path to saved HTML plot.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cross-Entropy Loss", "Classification Accuracy (%)"),
    )

    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#ff7f0e"),
        ),
        row=1,
        col=1,
    )

    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_acc"],
            mode="lines+markers",
            name="Train Acc",
            line=dict(color="#2ca02c"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_acc"],
            mode="lines+markers",
            name="Val Acc",
            line=dict(color="#d62728"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text=title,
        height=450,
        width=900,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

    fig.write_html(str(out_file))
    return out_file


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> Path:
    """Plot interactive confusion matrix heatmap with count annotations.

    Args:
        cm: Square confusion matrix array.
        class_names: List of class labels for rows and columns.
        save_path: Output HTML file path.
        title: Chart title.

    Returns:
        Path: Path to saved HTML chart.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=cm_arr,
            texttemplate="%{text}",
            textfont={"size": 14},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=550,
        width=650,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )

    fig.write_html(str(out_file))
    return out_file


def generate_classification_report_markdown(
    custom_cnn_val: Dict[str, Any],
    custom_cnn_test: Dict[str, Any],
    mobilenet_val: Dict[str, Any],
    mobilenet_test: Dict[str, Any],
    class_names: List[str],
    output_path: Union[str, Path],
) -> Path:
    """Generate Markdown report comparing Custom CNN and MobileNetV2.

    Args:
        custom_cnn_val: Validation metrics for Custom CNN.
        custom_cnn_test: Test metrics for Custom CNN.
        mobilenet_val: Validation metrics for MobileNetV2.
        mobilenet_test: Test metrics for MobileNetV2.
        class_names: List of class label strings.
        output_path: Path to output Markdown file.

    Returns:
        Path: Path to written report.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# SolarVision AI — Classification Training & Evaluation Report\n\n")
        f.write("> Generated: 2026-11-02 | Step 5: Classification Module\n\n")

        f.write("## 1. Overall Performance Comparison\n\n")
        f.write(
            "| Model | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")

        for m_name, val_m, test_m in [
            ("Custom CNN (Baseline)", custom_cnn_val, custom_cnn_test),
            ("MobileNetV2 (Pre-trained)", mobilenet_val, mobilenet_test),
        ]:
            f.write(
                f"| **{m_name}** | Validation | {val_m['accuracy']*100:.2f}% | "
                f"{val_m['precision_macro']:.4f} | {val_m['recall_macro']:.4f} | "
                f"{val_m['f1_macro']:.4f} | {val_m['weighted_f1'] if 'weighted_f1' in val_m else val_m['f1_weighted']:.4f} |\n"
            )
            f.write(
                f"| **{m_name}** | Test | **{test_m['accuracy']*100:.2f}%** | "
                f"{test_m['precision_macro']:.4f} | {test_m['recall_macro']:.4f} | "
                f"{test_m['f1_macro']:.4f} | {test_m['f1_weighted']:.4f} |\n"
            )

        f.write("\n## 2. Per-Class Test Performance (MobileNetV2)\n\n")
        f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        mob_per_class = mobilenet_test.get("per_class", {})
        for cls_name in class_names:
            cls_stat = mob_per_class.get(cls_name, {})
            p = cls_stat.get("precision", 0.0)
            r = cls_stat.get("recall", 0.0)
            f1 = cls_stat.get("f1-score", 0.0)
            sup = int(cls_stat.get("support", 0))
            f.write(f"| `{cls_name}` | {p:.4f} | {r:.4f} | {f1:.4f} | {sup} |\n")

        f.write("\n## 3. Per-Class Test Performance (Custom CNN Baseline)\n\n")
        f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        cnn_per_class = custom_cnn_test.get("per_class", {})
        for cls_name in class_names:
            cls_stat = cnn_per_class.get(cls_name, {})
            p = cls_stat.get("precision", 0.0)
            r = cls_stat.get("recall", 0.0)
            f1 = cls_stat.get("f1-score", 0.0)
            sup = int(cls_stat.get("support", 0))
            f.write(f"| `{cls_name}` | {p:.4f} | {r:.4f} | {f1:.4f} | {sup} |\n")

        f.write("\n## 4. Key Findings & Architecture Recommendation\n\n")
        diff_acc = (mobilenet_test["accuracy"] - custom_cnn_test["accuracy"]) * 100
        f.write(
            f"- MobileNetV2 achieved **{mobilenet_test['accuracy']*100:.2f}%** test accuracy, outperforming the Custom CNN baseline ({custom_cnn_test['accuracy']*100:.2f}%) by **{diff_acc:+.2f}%**.\n"
        )
        f.write(
            f"- Pretrained ImageNet representations provide strong feature transfer on solar defect textures.\n"
        )
        f.write(
            f"- Checkpoints are saved under `checkpoints/classification/` for downstream inference and dashboard integration.\n"
        )

    return out_file
