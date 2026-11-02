"""
Module: callbacks
Description: Training callbacks including EarlyStopping and metric monitoring
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Optional


class EarlyStopping:
    """Early stopping to terminate training when monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs with no improvement before stopping. Defaults to 5.
            min_delta: Minimum change in monitored metric to qualify as improvement. Defaults to 0.001.
            mode: 'min' for metrics that should decrease (e.g., loss),
                  'max' for metrics that should increase (e.g., accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        if self.mode not in ["min", "max"]:
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'min' or 'max'.")

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def is_improvement(self, current: float) -> bool:
        """Check if current score is an improvement over the best score.

        Args:
            current: Current epoch's metric value.

        Returns:
            bool: True if improved by at least min_delta.
        """
        if self.best_score is None:
            return True

        if self.mode == "min":
            return (self.best_score - current) > self.min_delta
        else:
            return (current - self.best_score) > self.min_delta

    def step(self, current: float, epoch: int = 0) -> bool:
        """Update early stopping state with current epoch's metric.

        Args:
            current: Current epoch's metric value.
            epoch: Current epoch number.

        Returns:
            bool: True if training should terminate early.
        """
        if self.is_improvement(current):
            self.best_score = current
            self.best_epoch = epoch
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
