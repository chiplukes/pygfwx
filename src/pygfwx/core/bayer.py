"""
GFWX Bayer Mode Support.

This module provides utilities for handling Bayer (CFA - Color Filter Array)
images in GFWX. Bayer images contain raw sensor data from digital cameras
where each pixel has only one color value (R, G, or B) arranged in a pattern.

Common Bayer Patterns (2x2 cell):
    RGGB:  R G    BGGR:  B G    GRBG:  G R    GBRG:  G B
           G B           G R           B G           R G

GFWX Bayer Mode Features:
- Processes each of the 4 sub-images separately (2x2 decimation)
- Uses step=2 for lifting/quantization to handle sub-pixel grids
- Treats green channels as luma, red/blue as chroma for quantization
- Preserves raw Bayer data without demosaicing

The sub-image layout (ox, oy offsets):
    (0,0)  (1,0)
    (0,1)  (1,1)

For RGGB pattern:
    R(0,0)  G(1,0)   <- Green1 (diagonal pair with G at 0,1)
    G(0,1)  B(1,1)   <- Green2 (diagonal pair with G at 1,0)
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from pygfwx.core.header import Intent


class BayerPattern(IntEnum):
    """
    Bayer CFA (Color Filter Array) patterns.

    Each pattern defines the color at position (0,0) of the 2x2 cell.
    The pattern repeats across the entire sensor.
    """

    RGGB = 2  # R at (0,0), G at (1,0) and (0,1), B at (1,1)
    BGGR = 3  # B at (0,0), G at (1,0) and (0,1), R at (1,1)
    GRBG = 4  # G at (0,0), R at (1,0), B at (0,1), G at (1,1)
    GBRG = 5  # G at (0,0), B at (1,0), R at (0,1), G at (1,1)
    GENERIC = 6  # Generic Bayer (no specific pattern)


@dataclass
class BayerSubImage:
    """
    Represents one of the 4 sub-images in a Bayer pattern.

    Each sub-image contains pixels at every other position (step=2).

    Attributes:
        ox: X offset (0 or 1) in the 2x2 cell.
        oy: Y offset (0 or 1) in the 2x2 cell.
        color: The color at this position ('R', 'G', or 'B').
        is_chroma: Whether this sub-image is treated as chroma.
    """

    ox: int
    oy: int
    color: str
    is_chroma: bool


def intent_is_bayer(intent: Intent | int) -> bool:
    """
    Check if an intent indicates a Bayer pattern.

    Args:
        intent: The intent value from the GFWX header.

    Returns:
        True if the intent is one of the Bayer patterns.
    """
    return Intent.BAYER_RGGB <= intent <= Intent.BAYER_GENERIC


def get_bayer_pattern(intent: Intent | int) -> BayerPattern | None:
    """
    Get the Bayer pattern from an intent value.

    Args:
        intent: The intent value from the GFWX header.

    Returns:
        The BayerPattern enum value, or None if not a Bayer intent.
    """
    if Intent.BAYER_RGGB <= intent <= Intent.BAYER_GENERIC:
        return BayerPattern(intent)
    return None


def get_bayer_sub_images(pattern: BayerPattern) -> list[BayerSubImage]:
    """
    Get the 4 sub-images for a Bayer pattern.

    Each sub-image represents one position in the 2x2 Bayer cell.
    Green positions are treated as luma (is_chroma=False),
    Red and Blue positions are treated as chroma (is_chroma=True).

    Args:
        pattern: The Bayer pattern.

    Returns:
        List of 4 BayerSubImage descriptors.

    The sub-images are returned in the order required for processing:
    - First (0,0) - the top-left pixel
    - Then (1,0), (0,1), (1,1) - rest of the 2x2 cell
    """
    # Define color at each position for each pattern
    # Order: (0,0), (1,0), (0,1), (1,1) - row-major 2x2 cell
    patterns = {
        BayerPattern.RGGB: [("R", True), ("G", False), ("G", False), ("B", True)],
        BayerPattern.BGGR: [("B", True), ("G", False), ("G", False), ("R", True)],
        BayerPattern.GRBG: [("G", False), ("R", True), ("B", True), ("G", False)],
        BayerPattern.GBRG: [("G", False), ("B", True), ("R", True), ("G", False)],
        BayerPattern.GENERIC: [("G", False), ("C", True), ("C", True), ("G", False)],  # Generic: assume green diagonal
    }

    colors = patterns.get(pattern, patterns[BayerPattern.GENERIC])

    # Positions in row-major order: (0,0), (1,0), (0,1), (1,1)
    # This matches: row0=[ox=0, ox=1], row1=[ox=0, ox=1]
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

    return [
        BayerSubImage(ox=pos[0], oy=pos[1], color=colors[i][0], is_chroma=colors[i][1])
        for i, pos in enumerate(positions)
    ]


def iter_bayer_offsets() -> Iterator[tuple[int, int]]:
    """
    Iterate over Bayer sub-image offsets in SDK order.

    The SDK processes sub-images with nested loops:
        for ox in [0, 1]:
            for oy in [0, 1]:

    Yields:
        Tuple of (ox, oy) offsets.
    """
    for ox in range(2):
        for oy in range(2):
            yield (ox, oy)


def iter_bayer_offsets_for_lifting() -> Iterator[tuple[int, int]]:
    """
    Iterate over Bayer sub-image offsets for the additional lifting passes.

    The SDK applies extra lifting passes for Bayer with a specific order:
        for ox in [0, 1]:
            for oy in [1-ox, ..., 1]:  # Skips (0,0) which is done first

    This yields offsets in order: (0,1), (1,0), (1,1) - skipping (0,0).

    Yields:
        Tuple of (ox, oy) offsets for additional Bayer lifting.
    """
    for ox in range(2):
        for oy in range(1 - ox, 2):
            yield (ox, oy)


def is_chroma_subimage(ox: int, oy: int, pattern: BayerPattern) -> bool:
    """
    Determine if a Bayer sub-image position is chroma.

    For Bayer patterns, the green positions are treated as luma
    and red/blue positions are treated as chroma.

    Args:
        ox: X offset in the 2x2 cell (0 or 1).
        oy: Y offset in the 2x2 cell (0 or 1).
        pattern: The Bayer pattern.

    Returns:
        True if this position is chroma (red or blue).
    """
    # For common patterns, (0,0) and (1,1) are diagonal
    # RGGB/BGGR: diagonal is non-green (R or B) at corners
    # GRBG/GBRG: diagonal is green (G) at corners

    if pattern == BayerPattern.RGGB:
        # R at (0,0), B at (1,1) - both chroma
        # G at (0,1), G at (1,0) - both luma
        return (ox == 0 and oy == 0) or (ox == 1 and oy == 1)

    elif pattern == BayerPattern.BGGR:
        # B at (0,0), R at (1,1) - both chroma
        # G at (0,1), G at (1,0) - both luma
        return (ox == 0 and oy == 0) or (ox == 1 and oy == 1)

    elif pattern == BayerPattern.GRBG:
        # G at (0,0), G at (1,1) - both luma
        # R at (1,0), B at (0,1) - both chroma
        return (ox == 1 and oy == 0) or (ox == 0 and oy == 1)

    elif pattern == BayerPattern.GBRG:
        # G at (0,0), G at (1,1) - both luma
        # B at (1,0), R at (0,1) - both chroma
        return (ox == 1 and oy == 0) or (ox == 0 and oy == 1)

    else:
        # Generic: assume green at diagonal (0,0) and (1,1)
        return (ox == 1 and oy == 0) or (ox == 0 and oy == 1)


def extract_bayer_subimage(
    image: np.ndarray,
    ox: int,
    oy: int,
) -> np.ndarray:
    """
    Extract a sub-image from a Bayer image.

    Takes every other pixel starting at offset (ox, oy), producing
    an image at half resolution in each dimension.

    Args:
        image: 2D Bayer image array.
        ox: X offset (0 or 1).
        oy: Y offset (0 or 1).

    Returns:
        Sub-image at half resolution.
    """
    return image[oy::2, ox::2].copy()


def insert_bayer_subimage(
    image: np.ndarray,
    subimage: np.ndarray,
    ox: int,
    oy: int,
) -> None:
    """
    Insert a sub-image back into a Bayer image.

    Places pixels at every other position starting at offset (ox, oy).

    Args:
        image: 2D Bayer image array (modified in-place).
        subimage: Sub-image data to insert.
        ox: X offset (0 or 1).
        oy: Y offset (0 or 1).
    """
    image[oy::2, ox::2] = subimage


def create_bayer_test_image(
    width: int,
    height: int,
    pattern: BayerPattern = BayerPattern.RGGB,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Create a synthetic Bayer test image.

    Generates a Bayer image where each color channel has a distinct
    gradient pattern for easy verification:
    - Red: horizontal gradient
    - Green: diagonal gradient (both G positions)
    - Blue: vertical gradient

    Args:
        width: Image width (should be even for complete Bayer cells).
        height: Image height (should be even for complete Bayer cells).
        pattern: The Bayer pattern to use.
        dtype: Output data type.

    Returns:
        2D Bayer image array.
    """
    # Determine max value based on dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        max_val = info.max
    else:
        max_val = 1.0

    image = np.zeros((height, width), dtype=dtype)

    # Create 2D gradients for each color
    # Horizontal gradient (varies with x)
    x_grad = np.tile(np.linspace(0, max_val, width, dtype=np.float64), (height, 1))
    # Vertical gradient (varies with y)
    y_grad = np.tile(np.linspace(0, max_val, height, dtype=np.float64).reshape(-1, 1), (1, width))
    # Diagonal gradient (average of x and y)
    diag_grad = (x_grad + y_grad) / 2

    # Map colors to positions based on pattern
    color_positions = {
        BayerPattern.RGGB: {"R": (0, 0), "G1": (1, 0), "G2": (0, 1), "B": (1, 1)},
        BayerPattern.BGGR: {"B": (0, 0), "G1": (1, 0), "G2": (0, 1), "R": (1, 1)},
        BayerPattern.GRBG: {"G1": (0, 0), "R": (1, 0), "B": (0, 1), "G2": (1, 1)},
        BayerPattern.GBRG: {"G1": (0, 0), "B": (1, 0), "R": (0, 1), "G2": (1, 1)},
        BayerPattern.GENERIC: {"G1": (0, 0), "C1": (1, 0), "C2": (0, 1), "G2": (1, 1)},
    }

    positions = color_positions.get(pattern, color_positions[BayerPattern.RGGB])

    for color, (ox, oy) in positions.items():
        if color == "R":
            # Red: horizontal gradient
            gradient = x_grad
        elif color == "B":
            # Blue: vertical gradient
            gradient = y_grad
        else:
            # Green (G1, G2) or generic chroma: diagonal gradient
            gradient = diag_grad

        # Extract the subimage portion of the gradient and assign
        image[oy::2, ox::2] = gradient[oy::2, ox::2].astype(dtype)

    return image


def create_uniform_bayer_image(
    width: int,
    height: int,
    r_value: int,
    g_value: int,
    b_value: int,
    pattern: BayerPattern = BayerPattern.RGGB,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Create a Bayer image with uniform color values.

    Useful for testing with known constant values.

    Args:
        width: Image width (should be even).
        height: Image height (should be even).
        r_value: Red channel value.
        g_value: Green channel value (used for both G positions).
        b_value: Blue channel value.
        pattern: The Bayer pattern.
        dtype: Output data type.

    Returns:
        2D Bayer image array.
    """
    image = np.zeros((height, width), dtype=dtype)

    # Map colors to positions based on pattern
    color_values = {
        BayerPattern.RGGB: {(0, 0): r_value, (1, 0): g_value, (0, 1): g_value, (1, 1): b_value},
        BayerPattern.BGGR: {(0, 0): b_value, (1, 0): g_value, (0, 1): g_value, (1, 1): r_value},
        BayerPattern.GRBG: {(0, 0): g_value, (1, 0): r_value, (0, 1): b_value, (1, 1): g_value},
        BayerPattern.GBRG: {(0, 0): g_value, (1, 0): b_value, (0, 1): r_value, (1, 1): g_value},
        BayerPattern.GENERIC: {(0, 0): g_value, (1, 0): r_value, (0, 1): b_value, (1, 1): g_value},
    }

    values = color_values.get(pattern, color_values[BayerPattern.RGGB])

    for (ox, oy), val in values.items():
        image[oy::2, ox::2] = val

    return image


def validate_bayer_dimensions(width: int, height: int) -> None:
    """
    Validate that dimensions are suitable for Bayer images.

    Bayer images should have even dimensions to contain complete
    2x2 cells.

    Args:
        width: Image width.
        height: Image height.

    Raises:
        ValueError: If dimensions are not valid for Bayer.
    """
    if width < 2 or height < 2:
        raise ValueError(f"Bayer image must be at least 2x2, got {width}x{height}")

    if width % 2 != 0:
        # Warning but not error - SDK handles odd dimensions
        pass

    if height % 2 != 0:
        # Warning but not error - SDK handles odd dimensions
        pass


def get_quality_for_subimage(
    ox: int,
    oy: int,
    luma_quality: int,
    chroma_quality: int,
    _pattern: BayerPattern = BayerPattern.GENERIC,
) -> int:
    """
    Get the appropriate quality value for a Bayer sub-image.

    In GFWX Bayer mode, green pixels use luma quality while
    red/blue pixels use chroma quality.

    Args:
        ox: X offset in the 2x2 cell.
        oy: Y offset in the 2x2 cell.
        luma_quality: Quality for luma (green) sub-images.
        chroma_quality: Quality for chroma (red/blue) sub-images.
        _pattern: The Bayer pattern (unused - SDK uses simpler logic).

    Returns:
        The quality value to use for this sub-image.

    Note:
        The SDK uses a simpler check: (ox | oy) ? chroma : luma
        This treats (0,0) as luma and others as chroma regardless of pattern.
    """
    # SDK logic: (ox | oy) ? chroma_quality : luma_quality
    # This means (0,0) is always treated as luma, others as chroma
    if ox | oy:
        return chroma_quality
    else:
        return luma_quality
