"""
Image I/O utilities for PyGFWX.  # cm:f1a2b3c — image_io module: load_image / save_image / get_bit_depth

Thin Pillow wrapper for loading and saving images as NumPy arrays.
Supports PNG, JPEG, and TIFF in 8-bit and 16-bit depths.

Shape convention (matches encode/decode API):
  - Grayscale:  (H, W)        uint8 or uint16
  - Multi-channel: (H, W, C)  uint8 or uint16
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Pillow internal mode strings that indicate 16-bit grayscale data
_16BIT_GRAY_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16S"}

# File extensions that support 16-bit depths
_16BIT_FORMATS = {".png", ".tif", ".tiff"}


def load_image(path: str | Path) -> np.ndarray:  # cm:d4e5f6a — load_image(): read PNG/JPEG/TIFF → (H,W) or (H,W,C) uint8/uint16
    """Load an image file as a NumPy array.

    Supports PNG, JPEG, and TIFF.  8-bit images return uint8 arrays;
    16-bit images (PNG/TIFF) return uint16 arrays.  Palette-mode images
    are converted to RGB or RGBA automatically.

    Args:
        path: Path to the image file.

    Returns:
        Shape (H, W) for grayscale, (H, W, C) for multi-channel.
        dtype is uint8 for 8-bit images, uint16 for 16-bit images.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The image mode is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)

    # Promote palette mode to RGB/RGBA
    if img.mode == "P":
        has_transparency = img.info.get("transparency") is not None
        img = img.convert("RGBA" if has_transparency else "RGB")

    # 16-bit grayscale: Pillow uses mode 'I' (int32 container) or 'I;16'
    if img.mode in _16BIT_GRAY_MODES:
        arr = np.array(img)
        # Pillow stores 16-bit data in an int32 container; reinterpret as uint16
        return arr.astype(np.uint16)

    # 16-bit multi-channel TIFF: Pillow may report mode 'RGB' or 'RGBA' with uint16 data
    arr = np.array(img)
    if arr.dtype == np.uint16:
        return arr

    # Standard 8-bit image
    return arr.astype(np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> None:  # cm:b7c8d9e — save_image(): write (H,W)/(H,W,C) uint8/uint16 → PNG/JPEG/TIFF
    """Save a NumPy image array to a file.

    Format is inferred from the file extension.  JPEG only supports 8-bit;
    16-bit images must be saved as PNG or TIFF.

    Args:
        image: Shape (H, W) for grayscale or (H, W, C) for multi-channel.
               dtype must be uint8 or uint16.
        path: Destination file path.

    Raises:
        ValueError: Unsupported dtype, channel count, or format/depth combination.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if image.dtype == np.uint8:
        _save_8bit(image, path, suffix)
    elif image.dtype == np.uint16:
        if suffix not in _16BIT_FORMATS:
            raise ValueError(f"16-bit images require PNG or TIFF, got {suffix!r}")
        if suffix in (".jpg", ".jpeg"):
            raise ValueError("JPEG does not support 16-bit images. Use PNG or TIFF.")
        _save_16bit(image, path)
    else:
        raise ValueError(f"Unsupported dtype {image.dtype!r}. Expected uint8 or uint16.")


def get_bit_depth(image: np.ndarray) -> int:
    """Return the bit depth of an image array.

    Args:
        image: NumPy array with dtype uint8 or uint16.

    Returns:
        8 for uint8, 16 for uint16.

    Raises:
        ValueError: dtype is not uint8 or uint16.
    """
    if image.dtype == np.uint8:
        return 8
    if image.dtype == np.uint16:
        return 16
    raise ValueError(f"Cannot determine bit depth for dtype {image.dtype!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_8bit(image: np.ndarray, path: Path, suffix: str) -> None:
    """Save a uint8 array using the appropriate Pillow mode."""
    pil_img = _array_to_pil_8bit(image)

    # JPEG cannot store an alpha channel — silently drop it
    if suffix in (".jpg", ".jpeg") and pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")

    pil_img.save(path)


def _save_16bit(image: np.ndarray, path: Path) -> None:
    """Save a uint16 array as a 16-bit PNG or TIFF.

    Only grayscale (1-channel) 16-bit images are supported.  Pillow does not
    provide reliable multi-channel 16-bit save support; use the ``tifffile``
    package for 16-bit RGB/RGBA TIFF files.
    """
    channels = 1 if image.ndim == 2 else image.shape[2]

    if channels == 1:
        mono = image if image.ndim == 2 else image[:, :, 0]
        h, w = mono.shape
        # frombuffer('I;16', ...) works for both PNG and TIFF without deprecation warnings
        pil_img = Image.frombuffer("I;16", (w, h), mono.tobytes(), "raw", "I;16", 0, 1)
        pil_img.save(path)

    else:
        raise ValueError(
            f"Saving 16-bit multi-channel images ({channels} channels) is not supported by Pillow. "
            "Use the `tifffile` package: `tifffile.imwrite(path, image)`."
        )


def _array_to_pil_8bit(image: np.ndarray) -> Image.Image:
    """Convert a uint8 numpy array to a Pillow Image."""
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")

    channels = image.shape[2]
    if channels == 1:
        return Image.fromarray(image[:, :, 0], mode="L")
    if channels == 3:
        return Image.fromarray(image, mode="RGB")
    if channels == 4:
        return Image.fromarray(image, mode="RGBA")
    raise ValueError(f"Unsupported channel count: {channels}")
