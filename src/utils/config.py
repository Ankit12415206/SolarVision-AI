"""
Module: config
Description: Configuration loading and reproducibility utilities for SolarVision AI
Author: SolarVision AI Team
Created: 2026-11-02
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration key-value dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries to ensure full reproducibility.

    Args:
        seed: Integer random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
