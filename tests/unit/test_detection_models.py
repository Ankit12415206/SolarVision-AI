"""
Module: test_detection_models
Description: Unit tests for lightweight YOLOv8n detection architecture
Author: SolarVision AI Team
Created: 2026-11-02
"""

import torch

from src.models.detection import SolarYOLOv8n, build_detection_model


def test_yolov8n_forward_shapes() -> None:
    """Test YOLOv8n forward pass produces expected multi-scale feature head shapes."""
    model = SolarYOLOv8n(num_classes=2, in_channels=3, base_channels=8)
    x = torch.randn(2, 3, 320, 320)
    p3, p4, p5 = model(x)

    # Output dimensions: 4 bbox coords + 1 objectness + 2 classes = 7 channels
    assert p3.shape == (2, 7, 40, 40)  # 320 / 8 = 40
    assert p4.shape == (2, 7, 20, 20)  # 320 / 16 = 20
    assert p5.shape == (2, 7, 10, 10)  # 320 / 32 = 10
    assert not torch.isnan(p3).any()


def test_build_detection_model_factory() -> None:
    """Test factory constructor builds SolarYOLOv8n."""
    model = build_detection_model("yolov8n", num_classes=2, base_channels=8)
    assert isinstance(model, SolarYOLOv8n)
