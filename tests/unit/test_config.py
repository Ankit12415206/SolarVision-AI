"""
Module: test_config
Description: Unit tests for configuration loader and seeding utilities
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pytest
import torch

from src.utils.config import load_yaml_config, set_seed


def test_load_yaml_config_success(tmp_path: Path) -> None:
    """Test loading a valid YAML file."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("project:\n  name: Test\n  seed: 123\n", encoding="utf-8")

    cfg = load_yaml_config(config_file)
    assert cfg["project"]["name"] == "Test"
    assert cfg["project"]["seed"] == 123


def test_load_yaml_config_not_found() -> None:
    """Test error when config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("non_existent_config.yaml")


def test_set_seed_reproducibility() -> None:
    """Test that set_seed ensures reproducible random tensor generation."""
    set_seed(42)
    t1 = torch.randn(5, 5)

    set_seed(42)
    t2 = torch.randn(5, 5)

    assert torch.equal(t1, t2)
