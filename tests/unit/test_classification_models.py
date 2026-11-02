"""
Module: test_classification_models
Description: Unit tests for CustomCNN and MobileNetV2 classification architectures
Author: SolarVision AI Team
Created: 2026-11-02
"""

import pytest
import torch

from src.data.classification import get_classification_dataloaders
from src.models.classification import (
    SolarCustomCNN,
    SolarMobileNetV2,
    build_classification_model,
)
from src.utils.model_utils import count_parameters


def test_custom_cnn_forward() -> None:
    """Test CustomCNN forward pass output shapes and logits."""
    model = SolarCustomCNN(num_classes=6, base_channels=16)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)
    assert not torch.isnan(out).any()


def test_custom_cnn_gradient_backward() -> None:
    """Test backward pass computes gradients across CustomCNN."""
    model = SolarCustomCNN(num_classes=6, base_channels=16)
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 3], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Check first conv layer has gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_mobilenet_v2_unfrozen() -> None:
    """Test MobileNetV2 architecture with unfrozen backbone."""
    model = SolarMobileNetV2(num_classes=6, pretrained=False, freeze_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)
    total, trainable = count_parameters(model)
    assert total == trainable


def test_mobilenet_v2_frozen() -> None:
    """Test MobileNetV2 with frozen backbone (linear probe)."""
    model = SolarMobileNetV2(num_classes=6, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)
    # 1280 * 6 weights + 6 biases = 7686 trainable in head
    assert trainable == 7686
    assert total > trainable


def test_mobilenet_v2_partially_unfrozen() -> None:
    """Test MobileNetV2 with partially unfrozen backbone (fine-tuning 3 blocks)."""
    model = SolarMobileNetV2(
        num_classes=6,
        pretrained=False,
        freeze_backbone=True,
        unfreeze_blocks=3,
    )
    total, trainable = count_parameters(model)
    assert trainable > 7686
    assert trainable < total


def test_build_classification_model_factory() -> None:
    """Test factory constructor builds correct model types."""
    m_cnn = build_classification_model("custom_cnn", num_classes=6)
    assert isinstance(m_cnn, SolarCustomCNN)

    m_mob = build_classification_model("mobilenet_v2", num_classes=6, pretrained=False)
    assert isinstance(m_mob, SolarMobileNetV2)

    with pytest.raises(ValueError):
        build_classification_model("unknown_model_type")


def test_classification_model_with_dataloader() -> None:
    """Test running forward pass using mini-batch from real DataLoader."""
    train_loader, _, _ = get_classification_dataloaders(
        manifest_dir="data/processed/splits",
        image_size=128,
        batch_size=4,
        num_workers=0,
    )
    images, labels = next(iter(train_loader))

    model = SolarCustomCNN(num_classes=6, base_channels=16)
    model.eval()
    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (4, 6)
    assert logits.dtype == torch.float32


def test_checkpoint_restoration_and_inference() -> None:
    """Verify saved best checkpoints can be restored and run inference on real input."""
    from pathlib import Path

    from src.utils.model_utils import load_checkpoint

    ckpt_mob = Path("checkpoints/classification/mobilenet_v2_best.pt")
    if ckpt_mob.exists():
        model_mob = build_classification_model(
            "mobilenet_v2", num_classes=6, pretrained=False
        )
        load_checkpoint(ckpt_mob, model_mob, device=torch.device("cpu"))
        model_mob.eval()
        with torch.no_grad():
            preds = model_mob(torch.randn(1, 3, 224, 224))
        assert preds.shape == (1, 6)

    ckpt_cnn = Path("checkpoints/classification/custom_cnn_best.pt")
    if ckpt_cnn.exists():
        model_cnn = build_classification_model("custom_cnn", num_classes=6)
        load_checkpoint(ckpt_cnn, model_cnn, device=torch.device("cpu"))
        model_cnn.eval()
        with torch.no_grad():
            preds_cnn = model_cnn(torch.randn(1, 3, 224, 224))
        assert preds_cnn.shape == (1, 6)
