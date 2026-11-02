"""
Module: test_model_utils
Description: Unit tests for model persistence, CPU device selection, and parameter counting
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.model_utils import (
    count_parameters,
    get_device,
    get_model_summary,
    load_checkpoint,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    """Tiny model for persistence testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_get_device() -> None:
    """Test device selection with CPU fallback."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ["cpu", "cuda"]

    dev_cpu = get_device("cpu")
    assert dev_cpu.type == "cpu"


def test_count_parameters() -> None:
    """Test parameter counting on fully trainable and partially frozen models."""
    model = SimpleModel()
    total, trainable = count_parameters(model)
    # 10*2 weights + 2 biases = 22 params
    assert total == 22
    assert trainable == 22

    # Freeze weight
    model.fc.weight.requires_grad = False
    total2, trainable2 = count_parameters(model)
    assert total2 == 22
    assert trainable2 == 2  # Only bias trainable


def test_get_model_summary() -> None:
    """Test model summary dictionary structure."""
    model = SimpleModel()
    summary = get_model_summary(model)
    assert summary["model_name"] == "SimpleModel"
    assert summary["total_parameters"] == 22
    assert summary["trainable_parameters"] == 22
    assert summary["frozen_parameters"] == 0


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """Test checkpoint save and load round-trip."""
    ckpt_path = tmp_path / "test_model.pt"
    model = SimpleModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # Save
    saved_path = save_checkpoint(
        model=model,
        path=ckpt_path,
        optimizer=opt,
        epoch=5,
        metrics={"val_acc": 0.92},
        config={"lr": 0.01},
    )
    assert saved_path.exists()

    # Load into fresh model
    model2 = SimpleModel()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    ckpt = load_checkpoint(
        saved_path, model2, optimizer=opt2, device=torch.device("cpu")
    )

    assert ckpt["epoch"] == 5
    assert ckpt["metrics"]["val_acc"] == 0.92

    # Verify identical parameters
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
