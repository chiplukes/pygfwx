"""
PyGFWX - Python implementation of the GFWX wavelet codec.

This package provides an educational implementation of the GFWX (Good, Fast Wavelet Codec)
for learning wavelet-based image compression.

Example:
    >>> from pygfwx import encode, decode
    >>> import numpy as np
    >>>
    >>> # Create a test image
    >>> image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    >>>
    >>> # Lossless compression
    >>> compressed = encode(image)
    >>> decoded = decode(compressed)
    >>> assert np.array_equal(image, decoded)
    >>>
    >>> # Lossy compression for better ratio
    >>> compressed = encode(image, quality=256)
    >>> decoded = decode(compressed)
"""

__version__ = "0.1.0"

# Import high-level API from core
from pygfwx.core.codec import (
    QUALITY_MAX,
    DecodeResult,
    EncodeResult,
    Encoder,
    Filter,
    GFWXHeader,
    Intent,
    decode,
    decode_full,
    encode,
    encode_full,
    get_header,
)

# Import progressive decoding
from pygfwx.streaming.progressive import (
    ProgressiveDecoder,
    ProgressiveResult,
    ProgressiveStatus,
    decode_progressive,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Main functions
    "encode",
    "decode",
    "encode_full",
    "decode_full",
    "get_header",
    # Progressive decoding
    "decode_progressive",
    "ProgressiveDecoder",
    "ProgressiveResult",
    "ProgressiveStatus",
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
