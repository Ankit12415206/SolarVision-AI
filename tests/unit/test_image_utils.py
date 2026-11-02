"""
Module: test_image_utils
Description: Unit tests for image loading, validation, and content hashing
Author: SolarVision AI Team
Created: 2026-11-02
"""

from pathlib import Path

import pytest
from PIL import Image

from src.utils.image import compute_file_hash, get_image_info, load_image


def test_compute_file_hash(tmp_path: Path) -> None:
    """Test file SHA-256 hash calculation."""
    f1 = tmp_path / "img1.bin"
    f2 = tmp_path / "img2.bin"
    f1.write_bytes(b"image_content_a")
    f2.write_bytes(b"image_content_a")

    h1 = compute_file_hash(f1)
    h2 = compute_file_hash(f2)
    assert h1 == h2
    assert len(h1) == 64


def test_load_image_rgb(tmp_path: Path) -> None:
    """Test loading and converting image to RGB."""
    img_path = tmp_path / "test.png"
    img = Image.new("L", (50, 50), color=128)
    img.save(img_path)

    loaded = load_image(img_path, mode="RGB")
    assert loaded.mode == "RGB"
    assert loaded.size == (50, 50)


def test_load_image_not_found() -> None:
    """Test error when loading non-existent image."""
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image.jpg")


def test_get_image_info(tmp_path: Path) -> None:
    """Test reading image dimensions without loading full buffer."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (120, 80), color="white")
    img.save(img_path, format="JPEG")

    size, fmt = get_image_info(img_path)
    assert size == (120, 80)
    assert fmt.upper() in ["JPEG", "JPG"]
