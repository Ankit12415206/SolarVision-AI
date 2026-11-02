"""
Module: visualizer
Description: Visual annotation and overlay generators for inference results
Author: SolarVision AI Team
Created: 2026-11-02
"""

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.inference.types import BoundingBox

CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Clean": (40, 167, 69),  # Green
    "Dusty": (255, 159, 28),  # Orange
    "Bird-drop": (230, 57, 70),  # Red
    "Electrical-damage": (155, 34, 38),  # Dark Red
    "Physical-Damage": (208, 0, 0),  # Bright Red
    "Snow-Covered": (0, 180, 216),  # Cyan
    "Dust": (255, 159, 28),  # Orange
    "bird-droppings": (230, 57, 70),  # Red
}


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Load default PIL font."""
    try:
        return ImageFont.load_default()
    except Exception:
        return None  # type: ignore


def annotate_classification(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    class_probabilities: Dict[str, float],
) -> Image.Image:
    """Annotate classification image with top prediction badge and probability bars.

    Args:
        image: Original input PIL Image.
        predicted_class: Predicted defect class label.
        confidence: Confidence score of top class in [0.0, 1.0].
        class_probabilities: Mapping from class names to softmax probabilities.

    Returns:
        Image.Image: New annotated RGB PIL Image with banner.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    font = _get_font(14)

    # Banner height: 60px
    banner_h = 60
    annotated = Image.new("RGB", (w, h + banner_h), color=(24, 28, 36))
    annotated.paste(img_rgb, (0, banner_h))

    draw = ImageDraw.Draw(annotated)
    tag_color = CLASS_COLOR_PALETTE.get(predicted_class, (40, 167, 69))

    # Top prediction badge
    badge_text = f"Condition: {predicted_class} ({confidence * 100:.1f}%)"
    draw.rectangle([10, 8, min(w - 10, 320), 32], fill=tag_color)
    draw.text((16, 12), badge_text, fill=(255, 255, 255), font=font)

    # Top 3 runner-up probabilities
    sorted_probs = sorted(
        class_probabilities.items(), key=lambda item: item[1], reverse=True
    )[:3]
    prob_str = " | ".join(f"{c}: {p*100:.1f}%" for c, p in sorted_probs)
    draw.text((12, 38), prob_str, fill=(180, 190, 205), font=font)

    return annotated


def annotate_detection(
    image: Image.Image,
    boxes: List[BoundingBox],
) -> Image.Image:
    """Draw defect bounding boxes with class labels and confidence tags onto image.

    Args:
        image: Original input PIL Image.
        boxes: List of BoundingBox objects in normalized coordinates [0.0, 1.0].

    Returns:
        Image.Image: New annotated RGB PIL Image.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size
    font = _get_font(12)

    for box in boxes:
        # Convert normalized [0, 1] to pixel coordinates
        x1 = max(0.0, min(float(w_img), box.x1 * w_img))
        y1 = max(0.0, min(float(h_img), box.y1 * h_img))
        x2 = max(0.0, min(float(w_img), box.x2 * w_img))
        y2 = max(0.0, min(float(h_img), box.y2 * h_img))

        color = CLASS_COLOR_PALETTE.get(box.class_name, (0, 255, 0))

        # Draw 2px rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label tag
        label_text = f"{box.class_name} {box.score:.2f}"
        tag_w = len(label_text) * 7 + 8
        tag_h = 16
        tag_y1 = max(0.0, y1 - tag_h)
        draw.rectangle([x1, tag_y1, x1 + tag_w, tag_y1 + tag_h], fill=color)
        draw.text((x1 + 4, tag_y1 + 2), label_text, fill=(255, 255, 255), font=font)

    return annotated


def annotate_segmentation(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.40,
) -> Image.Image:
    """Create multi-panel visualization of EL image, predicted mask, and highlighted overlay.

    Panels (left-to-right):
        1. Original Grayscale EL Image
        2. Color Highlight Overlay (Cyan/Green defect overlay on EL)
        3. Binary Defect Mask (White defects on black background)

    Args:
        image: Original input PIL Image (Grayscale or RGB).
        mask: 2D binary uint8 numpy array {0, 1} with matching aspect ratio.
        alpha: Blend transparency for defect overlay. Defaults to 0.40.

    Returns:
        Image.Image: Composite multi-panel RGB PIL Image.
    """
    # Resize mask if dimensions differ from original image
    w_orig, h_orig = image.size
    h_m, w_m = mask.shape
    if (w_orig, h_orig) != (w_m, h_m):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (w_orig, h_orig), resample=Image.NEAREST
        )
        mask_bin = np.array(mask_pil) > 127
    else:
        mask_bin = mask > 0

    img_gray = image.convert("L")
    img_rgb = Image.merge("RGB", (img_gray, img_gray, img_gray))
    arr_rgb = np.array(img_rgb, dtype=np.float32)

    # Panel 2: Highlight overlay (Bright Green: 0, 255, 128)
    overlay_arr = arr_rgb.copy()
    overlay_color = np.array([0, 255, 128], dtype=np.float32)
    overlay_arr[mask_bin] = (1.0 - alpha) * overlay_arr[
        mask_bin
    ] + alpha * overlay_color
    overlay_img = Image.fromarray(overlay_arr.astype(np.uint8))

    # Panel 3: Binary Mask (White on Black)
    mask_arr = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    mask_arr[mask_bin] = [255, 255, 255]
    mask_img = Image.fromarray(mask_arr)

    # Combine into 3-panel side-by-side composite
    composite = Image.new("RGB", (w_orig * 3, h_orig + 30), color=(20, 24, 30))
    composite.paste(img_rgb, (0, 30))
    composite.paste(overlay_img, (w_orig, 30))
    composite.paste(mask_img, (w_orig * 2, 30))

    # Add header titles
    draw = ImageDraw.Draw(composite)
    font = _get_font(12)
    draw.text((10, 8), "Original EL Image", fill=(200, 210, 225), font=font)
    draw.text(
        (w_orig + 10, 8), "Defect Highlight Overlay", fill=(0, 255, 128), font=font
    )
    draw.text(
        (w_orig * 2 + 10, 8), "Binary Defect Mask", fill=(255, 255, 255), font=font
    )

    return composite
