from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from paths containing non-ASCII characters on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image to paths containing non-ASCII characters on Windows."""
    suffix = Path(path).suffix or ".png"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        data.tofile(str(path))
    except OSError:
        return False
    return True
