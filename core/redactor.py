import cv2
import numpy as np
from typing import List, Tuple


def apply_mosaic(image: np.ndarray, regions: List[Tuple[int, int, int, int]], block_size: int = 12) -> np.ndarray:
    """
    Apply mosaic/pixelation effect to regions.
    regions: list of (x, y, w, h)
    """
    result = image.copy()
    height, width = result.shape[:2]

    for (x, y, w, h) in regions:
        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Resize down then up to create pixelation
        small_w = max(1, (x2 - x1) // block_size)
        small_h = max(1, (y2 - y1) // block_size)

        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        result[y1:y2, x1:x2] = mosaic

    return result


def apply_blur(image: np.ndarray, regions: List[Tuple[int, int, int, int]], ksize: int = 35) -> np.ndarray:
    """
    Apply Gaussian blur to regions.
    """
    result = image.copy()
    height, width = result.shape[:2]

    for (x, y, w, h) in regions:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Ensure ksize is odd
        k = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        result[y1:y2, x1:x2] = blurred

    return result


def apply_solid(image: np.ndarray, regions: List[Tuple[int, int, int, int]], color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Apply solid color block to regions.
    """
    result = image.copy()
    height, width = result.shape[:2]

    for (x, y, w, h) in regions:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        result[y1:y2, x1:x2] = color

    return result


def redact_image(image: np.ndarray, regions: List[Tuple[int, int, int, int]], mode: str = "mosaic", **kwargs) -> np.ndarray:
    """
    Main entry for applying redaction.
    mode: 'mosaic', 'blur', 'solid'
    """
    if mode == "mosaic":
        block_size = kwargs.get("block_size", 12)
        return apply_mosaic(image, regions, block_size)
    elif mode == "blur":
        ksize = kwargs.get("ksize", 35)
        return apply_blur(image, regions, ksize)
    elif mode == "solid":
        color = kwargs.get("color", (128, 128, 128))
        return apply_solid(image, regions, color)
    else:
        return apply_mosaic(image, regions)
