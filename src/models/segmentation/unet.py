"""
Module: unet
Description: Lightweight U-Net architecture for Electroluminescence defect segmentation
Author: SolarVision AI Team
Created: 2026-11-02
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential Conv2d + BatchNorm2d + ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize DoubleConv.

        Args:
            in_channels: Input channel depth.
            out_channels: Output channel depth.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling followed by feature concatenation with skip connections and DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize UpBlock.

        Args:
            in_channels: Incoming channels from previous decoder layer.
            out_channels: Output channel depth.
        """
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Upward decoder feature tensor.
            x2: Skip connection feature tensor from encoder.

        Returns:
            torch.Tensor: Decoded feature map.
        """
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SolarLightweightUNet(nn.Module):
    """Lightweight U-Net for solar cell electroluminescence (EL) defect segmentation.

    Operates on single-channel grayscale EL crops and produces single-channel binary logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        """Initialize SolarLightweightUNet.

        Args:
            in_channels: Input channels (1 for grayscale EL). Defaults to 1.
            out_channels: Output logits channels (1 for binary crack segmentation). Defaults to 1.
            base_channels: Initial conv channel multiplier. Defaults to 16.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)  # 16
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))  # 32
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))  # 64
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))  # 128
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(c * 8, c * 16)
        )  # 256 (Bottleneck)

        # Decoder
        self.up1 = UpBlock(c * 16, c * 8)  # 256 -> 128
        self.up2 = UpBlock(c * 8, c * 4)  # 128 -> 64
        self.up3 = UpBlock(c * 4, c * 2)  # 64 -> 32
        self.up4 = UpBlock(c * 2, c)  # 32 -> 16

        # Output projection
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input image tensor of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Unnormalized raw logits of shape (B, 1, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
