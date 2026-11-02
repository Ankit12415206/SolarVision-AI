"""
Module: test_segmentation_models
Description: Unit tests for lightweight U-Net and numerically stable BCE+Dice loss
Author: SolarVision AI Team
Created: 2026-11-02
"""

import torch

from src.models.segmentation import (
    BCEDiceLoss,
    DiceLoss,
    SolarLightweightUNet,
    build_segmentation_model,
)


def test_unet_forward_grayscale() -> None:
    """Test U-Net forward pass with 1-channel grayscale EL input."""
    model = SolarLightweightUNet(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(2, 1, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 1, 128, 128)
    assert not torch.isnan(logits).any()


def test_dice_loss_values() -> None:
    """Test DiceLoss on identical and disjoint target masks."""
    loss_fn = DiceLoss()
    # High confidence match
    logits_perfect = torch.ones((1, 1, 32, 32)) * 10.0
    targets_perfect = torch.ones((1, 1, 32, 32))
    loss_perfect = loss_fn(logits_perfect, targets_perfect)
    assert loss_perfect.item() < 0.1

    # Disjoint match
    logits_bad = torch.ones((1, 1, 32, 32)) * (-10.0)
    loss_bad = loss_fn(logits_bad, targets_perfect)
    assert loss_bad.item() > 0.8


def test_bce_dice_loss_numerical_stability() -> None:
    """Verify numerical stability of BCEDiceLoss with moderate and extreme pos_weights."""
    logits = torch.randn(2, 1, 64, 64)
    # Binary targets with sparse defects
    targets = (torch.rand(2, 1, 64, 64) > 0.98).float()

    # 1. Test with moderate pos_weight (recommended 15.0)
    loss_mod = BCEDiceLoss(pos_weight=15.0)
    val_mod = loss_mod(logits, targets)
    assert not torch.isnan(val_mod)
    assert not torch.isinf(val_mod)
    assert val_mod.item() > 0

    # 2. Test with raw extreme calculated pos_weight (1939.69)
    # The loss class caps it to max_pos_weight to avoid numerical explosion
    loss_extreme = BCEDiceLoss(pos_weight=1939.69, max_pos_weight=50.0)
    val_extreme = loss_extreme(logits, targets)
    assert not torch.isnan(val_extreme)
    assert not torch.isinf(val_extreme)
    assert val_extreme.item() < 100.0  # Kept bounded and stable


def test_build_segmentation_model_factory() -> None:
    """Test factory constructor builds SolarLightweightUNet."""
    model = build_segmentation_model(
        "unet", in_channels=1, out_channels=1, base_channels=8
    )
    assert isinstance(model, SolarLightweightUNet)
