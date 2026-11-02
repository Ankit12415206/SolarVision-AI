"""
Module: model_utils
Description: Model persistence, device configuration, parameter counting utilities
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the appropriate torch computation device with safe CPU fallback.

    Args:
        preferred: Preferred device string ('cuda', 'cpu'). If None, checks CUDA.

    Returns:
        torch.device: Available device object.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred is not None and preferred != "cuda":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch module.

    Args:
        model: Target PyTorch module.

    Returns:
        Tuple[int, int]: (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get structural parameter summary for a model.

    Args:
        model: Target PyTorch module.

    Returns:
        Dict[str, Any]: Dictionary containing class name, parameter counts, and modules.
    """
    total, trainable = count_parameters(model)
    return {
        "model_name": model.__class__.__name__,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model weights and training state to a checkpoint file.

    Args:
        model: Target PyTorch model.
        path: Path where checkpoint will be saved (.pt or .pth).
        optimizer: Optional optimizer whose state dict will be saved.
        epoch: Optional integer current epoch number.
        metrics: Optional dictionary of evaluation metrics.
        config: Optional dictionary of training configurations.

    Returns:
        Path: Resolved Path to saved checkpoint file.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer state from a checkpoint file.

    Ensures safe loading across CPU and GPU devices using map_location.

    Args:
        path: Path to checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: Optional optimizer instance to restore state into.
        device: Target computation device. Defaults to get_device().

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    target_device = device or get_device()
    checkpoint = torch.load(ckpt_path, map_location=target_device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Checkpoint might be direct state_dict
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
