"""
Module: evaluation
Description: Metrics computation, visualization, and reporting utilities
Author: SolarVision AI Team
Created: 2026-11-02
"""

from src.evaluation.metrics import evaluate_classification
from src.evaluation.visualizer import (
    generate_classification_report_markdown,
    plot_confusion_matrix,
    plot_training_history,
)

__all__ = [
    "evaluate_classification",
    "plot_training_history",
    "plot_confusion_matrix",
    "generate_classification_report_markdown",
]
