"""
Module: transforms
Description: Composable preprocessing and data transforms for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def get_classification_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    rotation_degrees: int = 15,
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.2,
    saturation_jitter: float = 0.1,
) -> transforms.Compose:
    """Build preprocessing and augmentation transform pipeline for image classification.

    Args:
        image_size: Target square image dimension (height and width).
        is_training: If True, include training augmentations (flip, jitter, rotation).
        mean: Normalization mean per channel. Defaults to ImageNet [0.485, 0.456, 0.406].
        std: Normalization std per channel. Defaults to ImageNet [0.229, 0.224, 0.225].
        rotation_degrees: Maximum degrees for random rotation.
        brightness_jitter: Brightness jitter factor.
        contrast_jitter: Contrast jitter factor.
        saturation_jitter: Saturation jitter factor.

    Returns:
        transforms.Compose: Composed PyTorch transform.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transform_list: List[Callable[[Any], Any]] = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]

    if is_training:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(
                    brightness=brightness_jitter,
                    contrast=contrast_jitter,
                    saturation=saturation_jitter,
                ),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(transform_list)


class DualSegmentationTransform:
    """Paired transform for segmentation image and mask.

    Ensures nearest-neighbor interpolation on masks so discrete integer classes
    (0 for background, 1 for defect) are strictly preserved without smoothing artifacts.
    """

    def __init__(
        self,
        image_size: int = 256,
        is_training: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ) -> None:
        """Initialize dual segmentation transform.

        Args:
            image_size: Target square spatial dimension (H, W).
            is_training: Whether to apply synchronous geometric augmentations.
            mean: Grayscale image normalization mean.
            std: Grayscale image normalization standard deviation.
            hflip_prob: Probability of horizontal flip in training.
            vflip_prob: Probability of vertical flip in training.
        """
        self.image_size = (image_size, image_size)
        self.is_training = is_training
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply paired transforms to image and mask.

        Args:
            image: PIL Image of the EL panel (mode 'L' or 'RGB').
            mask: PIL Image of the binary mask (mode 'L', values 0 and 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image_tensor: FloatTensor of shape (1, H, W) normalized.
                - mask_tensor: FloatTensor of shape (1, H, W) with values {0.0, 1.0}.
        """
        # Resize image with bilinear interpolation, mask with nearest-neighbor
        image = TF.resize(
            image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, self.image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # Synchronous geometric augmentations during training
        if self.is_training:
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Convert image to FloatTensor and normalize
        image_arr = np.array(image, dtype=np.float32)
        if image_arr.ndim == 2:
            image_arr = np.expand_dims(image_arr, axis=0)  # (1, H, W)
        elif image_arr.ndim == 3:
            image_arr = np.transpose(image_arr, (2, 0, 1))  # (C, H, W)

        # Normalize to [0, 1] then apply mean/std
        image_tensor = torch.from_numpy(image_arr / 255.0).float()
        image_tensor = (image_tensor - self.mean) / self.std

        # Convert mask to FloatTensor (1, H, W) with discrete binary values {0.0, 1.0}
        mask_arr = np.array(mask, dtype=np.float32)
        if mask_arr.ndim == 2:
            mask_arr = np.expand_dims(mask_arr, axis=0)
        # Ensure binary values: any non-zero becomes 1.0
        mask_arr = (mask_arr > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr).float()

        return image_tensor, mask_tensor
