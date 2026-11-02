"""
Module: segmentation_trainer
Description: Dedicated PyTorch training and validation loop for SolarLightweightUNet
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

from src.training.callbacks import EarlyStopping
from src.utils.model_utils import get_device, save_checkpoint


class SegmentationTrainer:
    """Trainer for SolarLightweightUNet binary electroluminescence defect segmentation."""

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
        checkpoint_dir: Union[str, Path] = "checkpoints/segmentation",
        model_name: str = "unet",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize SegmentationTrainer.

        Args:
            model: SolarLightweightUNet model instance.
            train_loader: DataLoader yielding (images, masks).
            val_loader: Validation DataLoader.
            criterion: Loss module (e.g. BCEDiceLoss).
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
            "val_dice": [],
            "lr": [],
        }
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self) -> float:
        """Execute one training epoch across all training batches.

        Returns:
            float: Average training loss for epoch.
        """
        self.model.train()
        running_loss = 0.0
        batch_count = 0

        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()

            # Gradient clipping for training stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        return running_loss / max(1, batch_count)

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Execute validation across validation DataLoader.

        Returns:
            Dict[str, float]: Validation loss and Dice score.
        """
        self.model.eval()
        running_loss = 0.0
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        batch_count = 0

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            running_loss += float(loss.item())
            batch_count += 1

            # Quick validation Dice at threshold 0.5
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            p_flat = preds.view(-1)
            t_flat = masks.view(-1)

            total_tp += (p_flat * t_flat).sum().item()
            total_fp += (p_flat * (1.0 - t_flat)).sum().item()
            total_fn += ((1.0 - p_flat) * t_flat).sum().item()

        val_loss = running_loss / max(1, batch_count)
        val_dice = (2.0 * total_tp + 1.0) / (2.0 * total_tp + total_fp + total_fn + 1.0)

        return {"val_loss": val_loss, "val_dice": float(val_dice)}

    def fit(self, epochs: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """Train model for specified epochs.

        Args:
            epochs: Maximum training epochs. Defaults to 10.
            verbose: If True, prints epoch progress.

        Returns:
            Dict[str, List[float]]: Training history metrics.
        """
        if verbose:
            print(
                f"Starting segmentation training '{self.model_name}' on {self.device} for {epochs} epochs..."
            )

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            val_loss = val_metrics["val_loss"]
            val_dice = val_metrics["val_dice"]

            cur_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["lr"].append(cur_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                save_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                metrics_dict = {
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "train_loss": train_loss,
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
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice*100:.2f}% | LR: {cur_lr:.6f}"
                )

            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss, epoch=epoch):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

        total_dur = time.time() - start_time
        if verbose:
            print(
                f"Segmentation training completed in {total_dur:.1f}s. "
                f"Best Val Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
            )

        return self.history
