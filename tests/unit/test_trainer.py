"""
Module: test_trainer
Description: Unit tests for early stopping callback and classification trainer
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.callbacks import EarlyStopping
from src.training.trainer import ClassificationTrainer


def test_early_stopping_min_mode() -> None:
    """Test EarlyStopping in 'min' mode (loss)."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    assert not es.step(1.0, epoch=1)
    assert es.counter == 0
    assert es.best_score == 1.0

    # Improvement
    assert not es.step(0.95, epoch=2)
    assert es.counter == 0
    assert es.best_score == 0.95

    # No improvements
    assert not es.step(0.955, epoch=3)
    assert es.counter == 1
    assert not es.step(0.96, epoch=4)
    assert es.counter == 2
    # Patience reached (3)
    should_stop = es.step(0.96, epoch=5)
    assert should_stop
    assert es.counter == 3
    assert es.best_epoch == 2


def test_early_stopping_max_mode() -> None:
    """Test EarlyStopping in 'max' mode (accuracy)."""
    es = EarlyStopping(patience=2, min_delta=0.5, mode="max")

    assert not es.step(80.0, epoch=1)
    assert not es.step(82.0, epoch=2)  # Improvement
    assert not es.step(82.1, epoch=3)  # Less than min_delta 0.5
    assert es.counter == 1
    assert es.step(81.5, epoch=4)  # Stagnant -> trigger stop
    assert es.counter == 2


def test_early_stopping_invalid_mode() -> None:
    """Test exception on invalid mode."""
    try:
        EarlyStopping(mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_classification_trainer_synthetic(tmp_path: Path) -> None:
    """Test ClassificationTrainer on synthetic mini-dataset."""
    # Synthetic dataset: 16 samples, 4 features, 2 classes
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=2, mode="min")

    trainer = ClassificationTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping=early_stopping,
        checkpoint_dir=tmp_path,
        model_name="synthetic_test",
    )

    history = trainer.fit(epochs=2, verbose=False)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["train_acc"]) == 2
    assert (tmp_path / "synthetic_test_best.pt").exists()
