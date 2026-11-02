"""
Module: yolo_lightweight
Description: Pure-PyTorch lightweight YOLO architecture for Aerial Soiling Object Detection
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm2d + SiLU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize ConvBNSiLU block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size. Defaults to 3.
            stride: Stride. Defaults to 1.
            padding: Padding. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Lightweight residual bottleneck with 1x1 and 3x3 convolutions."""

    def __init__(self, channels: int, shortcut: bool = True) -> None:
        """Initialize Bottleneck.

        Args:
            channels: Feature channels.
            shortcut: Whether to apply residual skip connection. Defaults to True.
        """
        super().__init__()
        self.cv1 = ConvBNSiLU(channels, channels, kernel_size=1, padding=0)
        self.cv2 = ConvBNSiLU(channels, channels, kernel_size=3, padding=1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        res = self.cv2(self.cv1(x))
        return x + res if self.shortcut else res


class C2fBlock(nn.Module):
    """YOLOv8-style C2f cross-stage partial block."""

    def __init__(self, in_channels: int, out_channels: int, n: int = 1) -> None:
        """Initialize C2fBlock.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            n: Number of bottleneck layers. Defaults to 1.
        """
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.cv2 = ConvBNSiLU(
            out_channels + n * hidden, out_channels, kernel_size=1, padding=0
        )
        self.bottlenecks = nn.ModuleList([Bottleneck(hidden) for _ in range(n)])
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = self.cv1(x)
        y_split = list(torch.split(y, self.hidden, dim=1))
        for b in self.bottlenecks:
            y_split.append(b(y_split[-1]))
        return self.cv2(torch.cat(y_split, dim=1))


class SolarYOLOv8n(nn.Module):
    """Lightweight anchor-free YOLOv8n detector for aerial drone PV defect inspection.

    Predicts bounding boxes (cx, cy, w, h) and class probabilities across 3 feature scales.
    Total parameter footprint is ~1.5 million parameters, keeping inference fast on CPUs.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        base_channels: int = 16,
    ) -> None:
        """Initialize SolarYOLOv8n detector.

        Args:
            num_classes: Number of defect classes (0: Dust, 1: bird-droppings). Defaults to 2.
            in_channels: Image channels (RGB). Defaults to 3.
            base_channels: Base channel scaling factor. Defaults to 16.
        """
        super().__init__()
        self.num_classes = num_classes
        c = base_channels

        # Backbone (P1 - P5)
        self.stem = ConvBNSiLU(
            in_channels, c, kernel_size=3, stride=2, padding=1
        )  # 320x320
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c, c * 2, stride=2),  # 160x160
            C2fBlock(c * 2, c * 2, n=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c * 2, c * 4, stride=2),  # 80x80 (P3)
            C2fBlock(c * 4, c * 4, n=2),
        )
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c * 4, c * 8, stride=2),  # 40x40 (P4)
            C2fBlock(c * 8, c * 8, n=2),
        )
        self.stage4 = nn.Sequential(
            ConvBNSiLU(c * 8, c * 16, stride=2),  # 20x20 (P5)
            C2fBlock(c * 16, c * 16, n=1),
        )

        # Decoupled detection heads for P3 (80x80), P4 (40x40), P5 (20x20)
        # Each head predicts: 4 bbox coordinates + 1 objectness + num_classes
        out_dim = 4 + 1 + num_classes
        self.head_p3 = nn.Conv2d(c * 4, out_dim, kernel_size=1)
        self.head_p4 = nn.Conv2d(c * 8, out_dim, kernel_size=1)
        self.head_p5 = nn.Conv2d(c * 16, out_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute forward pass.

        Args:
            x: Input image tensor of shape (B, 3, 640, 640).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out_p3: Predictions at stride 8 (B, 4 + 1 + num_classes, H/8, W/8).
                - out_p4: Predictions at stride 16 (B, 4 + 1 + num_classes, H/16, W/16).
                - out_p5: Predictions at stride 32 (B, 4 + 1 + num_classes, H/32, W/32).
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)  # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32

        pred_p3 = self.head_p3(p3)
        pred_p4 = self.head_p4(p4)
        pred_p5 = self.head_p5(p5)

        return pred_p3, pred_p4, pred_p5
