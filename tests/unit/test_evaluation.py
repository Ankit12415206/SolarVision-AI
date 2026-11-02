"""
Module: test_evaluation
Description: Unit tests for classification evaluation metrics and Plotly visualizations
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import evaluate_classification
from src.evaluation.visualizer import (
    generate_classification_report_markdown,
    plot_confusion_matrix,
    plot_training_history,
)


def test_evaluate_classification_synthetic() -> None:
    """Test evaluate_classification on a synthetic model and dataset."""
    x = torch.randn(20, 8)
    y = torch.randint(0, 3, (20,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=5)

    model = nn.Linear(8, 3)
    class_names = ["ClassA", "ClassB", "ClassC"]

    metrics = evaluate_classification(
        model, loader, device=torch.device("cpu"), class_names=class_names
    )

    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 3
    assert len(metrics["predictions"]) == 20


def test_plot_training_history(tmp_path: Path) -> None:
    """Test Plotly training history curve generation."""
    history = {
        "train_loss": [0.8, 0.5, 0.3],
        "val_loss": [0.9, 0.6, 0.4],
        "train_acc": [70.0, 82.0, 90.0],
        "val_acc": [68.0, 80.0, 88.0],
    }
    out_file = tmp_path / "history.html"
    res = plot_training_history(history, out_file)
    assert res.exists()
    assert res.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Test Plotly confusion matrix heatmap generation."""
    cm = [[10, 2], [1, 15]]
    out_file = tmp_path / "cm.html"
    res = plot_confusion_matrix(cm, ["Clean", "Dust"], out_file)
    assert res.exists()
    assert res.stat().st_size > 0


def test_generate_classification_report_markdown(tmp_path: Path) -> None:
    """Test markdown report generator comparing two models."""
    dummy_metrics = {
        "accuracy": 0.85,
        "precision_macro": 0.84,
        "recall_macro": 0.83,
        "f1_macro": 0.835,
        "f1_weighted": 0.845,
        "per_class": {
            "Clean": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10},
            "Dust": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 10},
        },
    }
    out_md = tmp_path / "report.md"
    generate_classification_report_markdown(
        custom_cnn_val=dummy_metrics,
        custom_cnn_test=dummy_metrics,
        mobilenet_val=dummy_metrics,
        mobilenet_test=dummy_metrics,
        class_names=["Clean", "Dust"],
        output_path=out_md,
    )
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Custom CNN" in content
    assert "MobileNetV2" in content
