"""
Module: test_transforms
Description: Unit tests for classification and paired segmentation transforms
Author: SolarVision AI Team
Created: 2026-11-02
"""

import numpy as np
import torch
from PIL import Image

from src.data.transforms import DualSegmentationTransform, get_classification_transforms


def test_classification_transforms_train() -> None:
    """Test classification training transform pipeline."""
    tf = get_classification_transforms(image_size=224, is_training=True)
    img = Image.new("RGB", (300, 400), color="blue")
    tensor = tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_classification_transforms_eval() -> None:
    """Test classification evaluation transform pipeline."""
    tf = get_classification_transforms(image_size=128, is_training=False)
    img = Image.new("RGB", (100, 100), color="red")
    tensor = tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_dual_segmentation_transform() -> None:
    """Test dual transform preserving discrete binary mask labels with nearest interpolation."""
    tf = DualSegmentationTransform(image_size=256, is_training=False)
    img = Image.new("L", (640, 590), color=100)
    # Mask with only 0 and 1 values
    mask_arr = np.zeros((590, 640), dtype=np.uint8)
    mask_arr[100:150, 100:150] = 1
    mask = Image.fromarray(mask_arr, mode="L")

    img_tensor, mask_tensor = tf(img, mask)

    assert img_tensor.shape == (1, 256, 256)
    assert mask_tensor.shape == (1, 256, 256)
    # Mask values must strictly remain 0.0 or 1.0 (no intermediate fractional values)
    unique_vals = torch.unique(mask_tensor).tolist()
    assert set(unique_vals).issubset({0.0, 1.0})
