"""
Module: custom_cnn
Description: Lightweight baseline Convolutional Neural Network for Solar Panel Classification
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm2d, ReLU, and optional MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
    ) -> None:
        """Initialize ConvBlock.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Convolution kernel dimension. Defaults to 3.
            stride: Convolution stride. Defaults to 1.
            padding: Spatial padding. Defaults to 1.
            pool: Whether to append a 2x2 MaxPool2d layer. Defaults to True.
        """
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        return self.block(x)


class SolarCustomCNN(nn.Module):
    """Custom 4-stage CNN baseline for solar panel defect classification.

    Lightweight architecture optimized for CPU training and inference on standard laptops.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """Initialize SolarCustomCNN.

        Args:
            num_classes: Number of defect categories. Defaults to 6.
            in_channels: Input image channels. Defaults to 3 (RGB).
            base_channels: Number of channels in first conv block. Defaults to 32.
            dropout: Dropout probability in the classifier head. Defaults to 0.3.
        """
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, c, pool=True),  # 32 channels, /2
            ConvBlock(c, c * 2, pool=True),  # 64 channels, /4
            ConvBlock(c * 2, c * 4, pool=True),  # 128 channels, /8
            ConvBlock(c * 4, c * 8, pool=True),  # 256 channels, /16
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        feat = self.features(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits
