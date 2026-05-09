"""
GFWX Quantization Module.

Implements scalar quantization and dequantization for lossy GFWX compression.

The quantization scheme approximates the JPEG 2000 baseline quantizer by
doubling the quality parameter at each wavelet level, resulting in finer
quantization for detail coefficients and coarser for larger-scale features.

Reference: gfwx.h quantize<T, dequantize>() lines 406-426

Key concepts:
- quality: Quantization step size (1-1024, higher = more compression/loss)
- maxQ: Maximum quality value (QualityMax * boost = 1024 * 8 for lossy)
- boost: 8 for lossy compression, 1 for lossless
- Traversal pattern: Processes coefficients at positions where (x | y) & skip != 0
"""

import numpy as np

from pygfwx.core.header import QUALITY_MAX


def quantize(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: int,
    quality: int,
    min_q: int,
    max_q: int,
) -> None:
    """
    Apply forward quantization to wavelet coefficients (encoding).

    Quantizes coefficients in-place using: coef * q / maxQ

    The traversal pattern matches the wavelet decomposition structure,
    processing coefficients at positions where (x | y) & skip != 0.
    Quality doubles at each level to approximate JPEG 2000 baseline.

    Args:
        image: 2D coefficient array (modified in-place).
        x0, y0: Top-left corner of region.
        x1, y1: Bottom-right corner (exclusive).
        step: Initial step size (1 for full image, 2 for Bayer sub-grids).
        quality: Base quality parameter (1-1024).
        min_q: Minimum quality threshold.
        max_q: Maximum quality * boost (usually 1024 * 8 = 8192).

    Note:
        For lossless encoding (quality=1024), no actual quantization occurs
        since q >= maxQ causes early exit.
    """
    sizex = x1 - x0
    sizey = y1 - y0
    skip = step

    while skip < sizex and skip < sizey:
        q = max(max(1, min_q), quality)
        if q >= max_q:
            break

        for y in range(0, sizey, skip):
            # xStep alternates based on y to match (x | y) & skip pattern
            x_step = skip if (y & skip) else skip * 2

            for x in range(x_step - skip, sizex, x_step):
                coef = int(image[y0 + y, x0 + x])
                # Forward quantization: coef * q / maxQ
                # Use int() truncation (toward zero) to match C++ integer division
                # Python's // is floor division which rounds toward -infinity
                quant = int(coef * q / max_q)
                image[y0 + y, x0 + x] = quant

        skip *= 2
        quality = min(max_q, quality * 2)  # [MAGIC] Approximates JPEG 2000 baseline


def dequantize(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: int,
    quality: int,
    min_q: int,
    max_q: int,
) -> None:
    """
    Apply inverse quantization to wavelet coefficients (decoding).

    Dequantizes coefficients in-place with rounding toward original value:
    - Positive coef: (coef * maxQ + maxQ/2) / q
    - Negative coef: (coef * maxQ - maxQ/2) / q
    - Zero coef: 0 (preserved exactly)

    The asymmetric rounding (+/- maxQ/2) helps reduce quantization bias.

    Args:
        image: 2D coefficient array (modified in-place).
        x0, y0: Top-left corner of region.
        x1, y1: Bottom-right corner (exclusive).
        step: Initial step size (1 for full image, 2 for Bayer sub-grids).
        quality: Base quality parameter (may be shifted for downsampling).
        min_q: Minimum quality threshold.
        max_q: Maximum quality * boost (usually 1024 * 8 = 8192).

    Note:
        For lossless decoding (quality >= 1024), no dequantization occurs
        since coefficients were not quantized during encoding.
    """
    sizex = x1 - x0
    sizey = y1 - y0
    skip = step

    while skip < sizex and skip < sizey:
        q = max(max(1, min_q), quality)
        if q >= max_q:
            break

        for y in range(0, sizey, skip):
            # xStep alternates based on y to match (x | y) & skip pattern
            x_step = skip if (y & skip) else skip * 2

            for x in range(x_step - skip, sizex, x_step):
                coef = int(image[y0 + y, x0 + x])
                if coef < 0:
                    # Negative: coef * maxQ - maxQ/2, then divide by q
                    dequant = (coef * max_q - max_q // 2) // q
                elif coef > 0:
                    # Positive: coef * maxQ + maxQ/2, then divide by q
                    dequant = (coef * max_q + max_q // 2) // q
                else:
                    dequant = 0
                image[y0 + y, x0 + x] = dequant

        skip *= 2
        quality = min(max_q, quality * 2)  # [MAGIC] Approximates JPEG 2000 baseline


def quantize_channel(
    image: np.ndarray,
    width: int,
    height: int,
    quality: int,
    is_chroma: bool = False,
    boost: int = 8,
) -> None:
    """
    Quantize a single channel's wavelet coefficients.

    High-level wrapper around quantize() for encoding.

    Args:
        image: 2D coefficient array (modified in-place).
        width: Image width.
        height: Image height.
        quality: Quality parameter (1-1024, higher = more loss).
        is_chroma: True for chroma channels (uses chromaQuality = quality * 2).
        boost: 8 for lossy, 1 for lossless.
    """
    max_q = QUALITY_MAX * boost
    effective_quality = quality * 2 if is_chroma else quality
    quantize(image, 0, 0, width, height, 1, effective_quality, 0, max_q)


def dequantize_channel(
    image: np.ndarray,
    width: int,
    height: int,
    quality: int,
    is_chroma: bool = False,
    boost: int = 8,
    downsampling: int = 0,
) -> None:
    """
    Dequantize a single channel's wavelet coefficients.

    High-level wrapper around dequantize() for decoding.

    Args:
        image: 2D coefficient array (modified in-place).
        width: Image width (at current resolution after downsampling).
        height: Image height (at current resolution).
        quality: Quality parameter (1-1024).
        is_chroma: True for chroma channels (uses chromaQuality = quality * 2).
        boost: 8 for lossy, 1 for lossless.
        downsampling: Downsampling level (quality << downsampling).
    """
    max_q = QUALITY_MAX * boost
    effective_quality = quality * 2 if is_chroma else quality
    # Shift quality for downsampling (progressive decode)
    effective_quality = effective_quality << downsampling
    dequantize(image, 0, 0, width, height, 1, effective_quality, 0, max_q)


def quantize_bayer(
    image: np.ndarray,
    width: int,
    height: int,
    quality: int,
    boost: int = 8,
) -> None:
    """
    Quantize Bayer pattern image (4 sub-grids).

    Bayer images have 2x2 sub-grids with different quantization:
    - (0,0) sub-grid uses base quality (green)
    - Other sub-grids use chromaQuality = quality * 2 (R, B, other G)

    Args:
        image: 2D coefficient array (modified in-place).
        width: Image width.
        height: Image height.
        quality: Base quality parameter.
        boost: 8 for lossy, 1 for lossless.
    """
    max_q = QUALITY_MAX * boost
    chroma_quality = quality * 2

    for ox in range(2):
        for oy in range(2):
            q = quality if (ox | oy) == 0 else chroma_quality
            min_q = quality  # Bayer uses quality as minQ
            quantize(image, ox, oy, width, height, 2, q, min_q, max_q)


def dequantize_bayer(
    image: np.ndarray,
    width: int,
    height: int,
    quality: int,
    boost: int = 8,
    downsampling: int = 0,
) -> None:
    """
    Dequantize Bayer pattern image (4 sub-grids).

    Args:
        image: 2D coefficient array (modified in-place).
        width: Image width.
        height: Image height.
        quality: Base quality parameter.
        boost: 8 for lossy, 1 for lossless.
        downsampling: Downsampling level for progressive decode.
    """
    max_q = QUALITY_MAX * boost
    chroma_quality = quality * 2

    for ox in range(2):
        for oy in range(2):
            q = quality if (ox | oy) == 0 else chroma_quality
            # Apply downsampling shift
            q = q << downsampling
            min_q = quality  # Bayer uses quality as minQ
            dequantize(image, ox, oy, width, height, 2, q, min_q, max_q)


def compute_effective_quality(
    base_quality: int,
    level: int,
    max_quality: int = QUALITY_MAX * 8,
) -> int:
    """
    Compute effective quality at a specific wavelet level.

    Quality doubles at each level, approximating JPEG 2000 baseline quantizer.

    Args:
        base_quality: Starting quality parameter.
        level: Wavelet decomposition level (0 = finest detail).
        max_quality: Maximum quality cap.

    Returns:
        Effective quality at the given level.
    """
    q = base_quality
    for _ in range(level):
        q = min(max_quality, q * 2)
    return q


def is_lossless(quality: int) -> bool:
    """
    Check if the given quality setting results in lossless compression.

    Args:
        quality: Quality parameter (1-1024).

    Returns:
        True if quality >= QUALITY_MAX (no quantization will occur).
    """
    return quality >= QUALITY_MAX


def get_quantization_info(
    quality: int,
    width: int,
    height: int,
    boost: int = 8,
) -> dict:
    """
    Get information about quantization for given parameters.

    Useful for understanding compression behavior at different quality levels.

    Args:
        quality: Quality parameter (1-1024).
        width: Image width.
        height: Image height.
        boost: Boost factor.

    Returns:
        Dictionary with quantization analysis:
        - is_lossless: Whether quantization will be skipped
        - max_q: Maximum quality value
        - num_levels: Number of wavelet levels that will be quantized
        - level_qualities: Quality at each level
    """
    max_q = QUALITY_MAX * boost

    # Count wavelet levels
    max_dim = max(width, height)
    num_levels = 0
    skip = 1
    while skip < max_dim:
        skip *= 2
        num_levels += 1

    # Compute quality at each level
    level_qualities = []
    q = quality
    for _ in range(num_levels):
        q_eff = max(1, q)
        if q_eff >= max_q:
            break
        level_qualities.append(q_eff)
        q = min(max_q, q * 2)

    return {
        "is_lossless": is_lossless(quality),
        "max_q": max_q,
        "num_levels": num_levels,
        "levels_quantized": len(level_qualities),
        "level_qualities": level_qualities,
    }
