"""
Module: image
Description: Image loading, conversion, and hashing utilities for duplicate detection
Author: SolarVision AI Team
Created: 2026-11-02
"""

import hashlib
from pathlib import Path
from typing import Tuple, Union

from PIL import Image


def compute_file_hash(path: Union[str, Path], block_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file on disk.

    Args:
        path: Path to the target file.
        block_size: Read buffer block size in bytes.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image(
    path: Union[str, Path],
    mode: str = "RGB",
) -> Image.Image:
    """Safely open an image and convert it to the desired color mode.

    Args:
        path: Path to the image file.
        mode: Target PIL mode (e.g., 'RGB', 'L'). Defaults to 'RGB'.

    Returns:
        Image.Image: Loaded PIL Image in the specified color mode.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file is not a readable image.
    """
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    try:
        with Image.open(img_path) as img:
            return img.convert(mode)
    except Exception as exc:
        raise ValueError(f"Failed to read image at {img_path}: {exc}") from exc


def get_image_info(path: Union[str, Path]) -> Tuple[Tuple[int, int], str]:
    """Get dimensions (width, height) and format of an image without loading full pixel buffer.

    Args:
        path: Path to the image file.

    Returns:
        Tuple[Tuple[int, int], str]: ((width, height), format_name).
    """
    with Image.open(path) as img:
        return img.size, img.format or "UNKNOWN"
