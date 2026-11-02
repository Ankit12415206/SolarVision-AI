"""
Module: trainer
Description: PyTorch training loop with early stopping, checkpointing, and scheduler support
Author: SolarVision AI Team
Created: 2026-11-02
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.utils.model_utils import get_device, save_checkpoint


class ClassificationTrainer:
    """Trainer for image classification models with metric tracking and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints/classification",
        model_name: str = "classification_model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ClassificationTrainer.

        Args:
            model: PyTorch classification model.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            criterion: Loss function (e.g., CrossEntropyLoss).
            optimizer: Optimizer (e.g., AdamW, SGD).
            scheduler: Learning rate scheduler.
            device: Computation device (CPU/CUDA). Defaults to get_device().
            early_stopping: Optional EarlyStopping callback instance.
            checkpoint_dir: Directory where checkpoints are saved.
            model_name: Base filename prefix for checkpoints.
            metadata: Additional metadata dictionary saved with checkpoints.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.metadata = metadata or {}

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self) -> Tuple[float, float]:
        """Run a single training epoch across all training batches.

        Returns:
            Tuple[float, float]: (average_epoch_loss, accuracy_percentage).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, float]:
        """Run validation across validation DataLoader.

        Returns:
            Tuple[float, float]: (average_val_loss, val_accuracy_percentage).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return epoch_loss, epoch_acc

    def fit(
        self,
        epochs: int = 15,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model for a specified number of epochs with early stopping.

        Args:
            epochs: Maximum number of training epochs. Defaults to 15.
            verbose: If True, prints per-epoch progress logs.

        Returns:
            Dict[str, List[float]]: Training and validation history.
        """
        if verbose:
            print(
                f"Starting training '{self.model_name}' on {self.device} for {epochs} epochs..."
            )

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Check for best model based on validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch

                save_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                metrics = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "epoch": epoch,
                }
                save_checkpoint(
                    model=self.model,
                    path=save_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=self.metadata,
                )
                self.best_checkpoint_path = save_path

            if verbose:
                mark = "*" if is_best else " "
                duration = time.time() - epoch_start
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{duration:.1f}s] {mark} "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {current_lr:.6f}"
                )

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss, epoch=epoch):
                    if verbose:
                        print(
                            f"Early stopping triggered at epoch {epoch} (patience reached)."
                        )
                    break

        total_duration = time.time() - start_time
        if verbose:
            print(
                f"Training completed in {total_duration:.1f}s. "
                f"Best Val Loss: {self.best_val_loss:.4f} (Acc: {self.best_val_acc:.1f}%) at epoch {self.best_epoch}"
            )

        return self.history
