"""
Core codec implementation for PyGFWX.

This module contains the fundamental building blocks of the GFWX codec:
- High-level encode/decode API
- Bitstream I/O
- Header parsing/writing
- Wavelet lifting transforms
- Quantization
- Golomb-Rice entropy coding
- Context modeling
- Coefficient encoding/decoding
"""

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.block_decoder import DecodeResult, decode_image
from pygfwx.core.block_encoder import EncodeResult, encode_image
from pygfwx.core.codec import decode, decode_full, encode, encode_full, get_header
from pygfwx.core.context import (
    Context,
    compute_run_coder,
    get_context,
    select_coding_mode,
    update_fast_context,
)
from pygfwx.core.decoder import decode_block, decode_coefficients
from pygfwx.core.encoder import encode_block, encode_coefficients
from pygfwx.core.golomb_rice import (
    interleaved_decode,
    interleaved_encode,
    signed_decode,
    signed_encode,
    unsigned_decode,
    unsigned_encode,
)
from pygfwx.core.header import (
    GFWX_MAGIC,
    QUALITY_MAX,
    Encoder,
    Filter,
    GFWXHeader,
    Intent,
    parse_header,
)
from pygfwx.core.lifting import lift, lift_full, unlift, unlift_full

__all__ = [
    # High-level API
    "encode",
    "decode",
    "encode_full",
    "decode_full",
    "get_header",
    "encode_image",
    "decode_image",
    "EncodeResult",
    "DecodeResult",
    # Bitstream
    "BitReader",
    "BitWriter",
    # Header
    "GFWX_MAGIC",
    "QUALITY_MAX",
    "Encoder",
    "Filter",
    "GFWXHeader",
    "Intent",
    "parse_header",
    # Lifting transforms
    "lift",
    "lift_full",
    "unlift",
    "unlift_full",
    # Golomb-Rice
    "unsigned_decode",
    "unsigned_encode",
    "interleaved_decode",
    "interleaved_encode",
    "signed_decode",
    "signed_encode",
    # Context
    "Context",
    "get_context",
    "select_coding_mode",
    "update_fast_context",
    "compute_run_coder",
    # Encoder/Decoder
    "encode_coefficients",
    "encode_block",
    "decode_coefficients",
    "decode_block",
]
