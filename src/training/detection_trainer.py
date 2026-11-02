"""
Module: detection_trainer
Description: Dedicated PyTorch training and validation loop for SolarYOLOv8n object detector
Author: SolarVision AI Team
Created: 2026-11-02
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.detection.loss import SolarDetectionLoss
from src.training.callbacks import EarlyStopping
from src.utils.model_utils import get_device, save_checkpoint


class DetectionTrainer:
    """Trainer for SolarYOLOv8n multi-scale object detection."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: SolarDetectionLoss,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints/detection",
        model_name: str = "yolov8n",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DetectionTrainer.

        Args:
            model: SolarYOLOv8n model instance.
            train_loader: Detection DataLoader yielding (images, targets).
            val_loader: Validation DataLoader.
            criterion: SolarDetectionLoss instance.
            optimizer: Optimizer.
            scheduler: Optional learning rate scheduler.
            device: Computation device.
            early_stopping: Optional EarlyStopping callback.
            checkpoint_dir: Target folder for checkpoints.
            model_name: Checkpoint prefix name.
            metadata: Config metadata saved in checkpoint.
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
            "val_loss": [],
            "train_loss_obj": [],
            "train_loss_box": [],
            "train_loss_cls": [],
            "val_loss_obj": [],
            "val_loss_box": [],
            "val_loss_cls": [],
            "lr": [],
        }
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch across all training batches.

        Returns:
            Dict[str, float]: Average loss metrics for epoch.
        """
        self.model.train()
        running_loss = 0.0
        running_obj = 0.0
        running_box = 0.0
        running_cls = 0.0
        batch_count = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss, loss_dict = self.criterion(preds, targets)
            loss.backward()

            # Gradient clipping for training stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            running_loss += loss_dict["loss_total"]
            running_obj += loss_dict["loss_obj"]
            running_box += loss_dict["loss_box"]
            running_cls += loss_dict["loss_cls"]
            batch_count += 1

        n = max(1, batch_count)
        return {
            "loss_total": running_loss / n,
            "loss_obj": running_obj / n,
            "loss_box": running_box / n,
            "loss_cls": running_cls / n,
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Execute validation across validation DataLoader.

        Returns:
            Dict[str, float]: Validation loss metrics.
        """
        self.model.eval()
        running_loss = 0.0
        running_obj = 0.0
        running_box = 0.0
        running_cls = 0.0
        batch_count = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            preds = self.model(images)
            _, loss_dict = self.criterion(preds, targets)

            running_loss += loss_dict["loss_total"]
            running_obj += loss_dict["loss_obj"]
            running_box += loss_dict["loss_box"]
            running_cls += loss_dict["loss_cls"]
            batch_count += 1

        n = max(1, batch_count)
        return {
            "loss_total": running_loss / n,
            "loss_obj": running_obj / n,
            "loss_box": running_box / n,
            "loss_cls": running_cls / n,
        }

    def fit(self, epochs: int = 15, verbose: bool = True) -> Dict[str, List[float]]:
        """Train detector for specified epochs.

        Args:
            epochs: Maximum training epochs. Defaults to 15.
            verbose: If True, prints epoch progress.

        Returns:
            Dict[str, List[float]]: Training history metrics.
        """
        if verbose:
            print(
                f"Starting detection training '{self.model_name}' on {self.device} for {epochs} epochs..."
            )

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss_total"])
            self.history["val_loss"].append(val_metrics["loss_total"])
            self.history["train_loss_obj"].append(train_metrics["loss_obj"])
            self.history["train_loss_box"].append(train_metrics["loss_box"])
            self.history["train_loss_cls"].append(train_metrics["loss_cls"])
            self.history["val_loss_obj"].append(val_metrics["loss_obj"])
            self.history["val_loss_box"].append(val_metrics["loss_box"])
            self.history["val_loss_cls"].append(val_metrics["loss_cls"])
            self.history["lr"].append(cur_lr)

            val_total = val_metrics["loss_total"]
            is_best = val_total < self.best_val_loss
            if is_best:
                self.best_val_loss = val_total
                self.best_epoch = epoch
                save_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                metrics_dict = {
                    "val_loss": val_total,
                    "train_loss": train_metrics["loss_total"],
                    "epoch": epoch,
                }
                save_checkpoint(
                    model=self.model,
                    path=save_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics_dict,
                    config=self.metadata,
                )
                self.best_checkpoint_path = save_path

            if verbose:
                mark = "*" if is_best else " "
                dur = time.time() - t_epoch
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s] {mark} "
                    f"Train Loss: {train_metrics['loss_total']:.4f} (Obj: {train_metrics['loss_obj']:.3f}, Box: {train_metrics['loss_box']:.3f}) | "
                    f"Val Loss: {val_total:.4f} | LR: {cur_lr:.6f}"
                )

            if self.early_stopping is not None:
                if self.early_stopping.step(val_total, epoch=epoch):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

        total_dur = time.time() - start_time
        if verbose:
            print(
                f"Detection training completed in {total_dur:.1f}s. "
                f"Best Val Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
            )

        return self.history
