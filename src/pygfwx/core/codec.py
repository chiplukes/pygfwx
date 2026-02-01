"""
PyGFWX High-Level Codec API.

This module provides the main user-facing API for GFWX compression.
It wraps the lower-level block encoder/decoder with a simple interface.

Example:
    >>> from pygfwx import encode, decode
    >>> import numpy as np
    >>>
    >>> # Create test image
    >>> image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    >>>
    >>> # Lossless compression
    >>> compressed = encode(image)
    >>> decoded = decode(compressed)
    >>> assert np.array_equal(image, decoded)
    >>>
    >>> # Lossy compression (better ratio, lower quality)
    >>> compressed = encode(image, quality=256)
    >>> decoded = decode(compressed)
"""

from typing import Optional

import numpy as np

from pygfwx.core.block_decoder import DecodeResult, decode_image
from pygfwx.core.block_encoder import EncodeResult, encode_image
from pygfwx.core.header import (
    QUALITY_MAX,
    Encoder,
    Filter,
    GFWXHeader,
    Intent,
    parse_header,
)

__all__ = [
    # Main functions
    "encode",
    "decode",
    "get_header",
    # Enums
    "Filter",
    "Encoder",
    "Intent",
    # Classes
    "GFWXHeader",
    "EncodeResult",
    "DecodeResult",
    # Constants
    "QUALITY_MAX",
]


def encode(
    image: np.ndarray,
    quality: int = QUALITY_MAX,
    filter: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Optional[Intent] = None,
    chroma_scale: int = 1,
    metadata: bytes = b"",
) -> bytes:
    """
    Encode an image to GFWX format.

    Args:
        image: Input image as numpy array.
            - Shape (H, W) for grayscale
            - Shape (H, W, 3) for RGB
            - Shape (H, W, 4) for RGBA
            - dtype: uint8 or uint16
        quality: Quality parameter (1-1024).
            - 1024 = lossless (default)
            - 512 = high quality lossy
            - 256 = medium quality lossy
            - Lower values = more compression, more loss
        filter: Wavelet filter type.
            - Filter.LINEAR = 5/3 wavelet, best for lossless
            - Filter.CUBIC = 9/7 wavelet, best for lossy
        encoder: Encoding mode.
            - Encoder.CONTEXTUAL = best compression (default)
            - Encoder.FAST = faster, less compression
            - Encoder.HIGH_BITRATE = best for high quality
        intent: Color intent (auto-detected from channel count if None).
        chroma_scale: Chroma quality divisor (1 = same quality as luma).
        metadata: Optional metadata bytes (must be multiple of 4 bytes).

    Returns:
        Compressed GFWX data as bytes.

    Raises:
        ValueError: If input is invalid.

    Example:
        >>> img = np.zeros((64, 64), dtype=np.uint8)
        >>> compressed = encode(img)
        >>> len(compressed) < img.nbytes
        True
    """
    result = encode_image(
        image=image,
        quality=quality,
        filter_type=filter,
        encoder=encoder,
        intent=intent,
        chroma_scale=chroma_scale,
        metadata=metadata,
    )
    return result.data


def decode(
    data: bytes,
    downsampling: int = 0,
) -> np.ndarray:
    """
    Decode GFWX compressed data to an image.

    Args:
        data: Compressed GFWX data.
        downsampling: Downsampling factor for reduced resolution decode.
            - 0 = full resolution (default)
            - 1 = half resolution (2x smaller)
            - 2 = quarter resolution (4x smaller)
            - n = 2^n times smaller

    Returns:
        Decoded image as numpy array.
        - Shape (H, W) for grayscale
        - Shape (H, W, C) for multi-channel
        - dtype: uint8 for 8-bit, uint16 for 16-bit

    Raises:
        ValueError: If data is malformed or unsupported.

    Example:
        >>> compressed = encode(np.zeros((64, 64), dtype=np.uint8))
        >>> decoded = decode(compressed)
        >>> decoded.shape
        (64, 64)
    """
    result = decode_image(data=data, downsampling=downsampling)
    return result.image


def get_header(data: bytes) -> GFWXHeader:
    """
    Parse header from GFWX data without full decode.

    Useful for inspecting compressed data properties without
    the cost of full decompression.

    Args:
        data: Compressed GFWX data.

    Returns:
        Parsed GFWXHeader with image properties.

    Raises:
        ValueError: If header is malformed.

    Example:
        >>> compressed = encode(np.zeros((64, 64, 3), dtype=np.uint8))
        >>> header = get_header(compressed)
        >>> header.sizex, header.sizey, header.channels
        (64, 64, 3)
    """
    header, _ = parse_header(data)
    return header


def encode_full(
    image: np.ndarray,
    quality: int = QUALITY_MAX,
    filter: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Optional[Intent] = None,
    chroma_scale: int = 1,
    metadata: bytes = b"",
) -> EncodeResult:
    """
    Encode image and return full result with header.

    Same as encode() but returns EncodeResult with both
    compressed data and header information.

    Args:
        image: Input image as numpy array.
        quality: Quality parameter (1-1024, 1024=lossless).
        filter: Wavelet filter type.
        encoder: Encoding mode.
        intent: Color intent (auto-detected if None).
        chroma_scale: Chroma quality divisor.
        metadata: Optional metadata bytes.

    Returns:
        EncodeResult with .data and .header attributes.
    """
    return encode_image(
        image=image,
        quality=quality,
        filter_type=filter,
        encoder=encoder,
        intent=intent,
        chroma_scale=chroma_scale,
        metadata=metadata,
    )


def decode_full(
    data: bytes,
    downsampling: int = 0,
) -> DecodeResult:
    """
    Decode GFWX data and return full result with header.

    Same as decode() but returns DecodeResult with image,
    header, and truncation status.

    Args:
        data: Compressed GFWX data.
        downsampling: Downsampling factor.

    Returns:
        DecodeResult with .image, .header, and .is_truncated attributes.
    """
    return decode_image(data=data, downsampling=downsampling)
