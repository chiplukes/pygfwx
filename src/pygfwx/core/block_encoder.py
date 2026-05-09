"""
GFWX Block Encoder - High-level encoding pipeline.

This module implements the full GFWX encode pipeline that:
1. Validates input and creates header
2. Applies forward color transform (if present)
3. Applies forward wavelet transform (lift)
4. Applies quantization for lossy compression
5. Encodes coefficient blocks
6. Writes header, transform program, block sizes, and block data

The encoding order mirrors the decoder for bit-exact roundtrip.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pygfwx.core.bitstream import BitWriter
from pygfwx.core.encoder import encode_coefficients
from pygfwx.core.golomb_rice import signed_encode
from pygfwx.core.header import (
    QUALITY_MAX,
    Encoder,
    Filter,
    GFWXHeader,
    Intent,
    create_default_header,
    write_header,
)
from pygfwx.core.lifting import lift
from pygfwx.core.quantization import quantize


@dataclass
class EncodeResult:  # cm:b8c9d0 — EncodeResult dataclass: compressed bytes + header from encode
    """Result of encoding operation."""

    data: bytes
    """Compressed GFWX data."""

    header: GFWXHeader
    """Header used for encoding."""

    @property
    def compressed_size(self) -> int:
        """Size of compressed data in bytes."""
        return len(self.data)


def encode_image(  # cm:e1f2a3 — encode_image(): full encode pipeline (validate→lift→quantize→entropy-code)
    image: np.ndarray,
    quality: int = QUALITY_MAX,
    filter_type: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Optional[Intent] = None,
    chroma_scale: int = 1,
    metadata: bytes = b"",
) -> EncodeResult:
    """
    Encode an image to GFWX format.

    This is the main entry point for encoding. It handles the complete
    encode pipeline from numpy array to compressed bytes.

    Args:
        image: Input image as numpy array.
            - Shape (H, W) for mono
            - Shape (H, W, C) for multi-channel (C=3 for RGB, C=4 for RGBA)
            - dtype: uint8 or uint16
        quality: Quality parameter (1-1024, 1024=lossless).
        filter_type: Wavelet filter (LINEAR for lossless, CUBIC for lossy).
        encoder: Encoder mode (CONTEXTUAL default, FAST, HIGH_BITRATE).
        intent: Color intent (auto-detected if None).
        chroma_scale: Chroma quality divisor (1=same as luma).
        metadata: Optional metadata bytes (must be multiple of 4).

    Returns:
        EncodeResult containing compressed data and header.

    Raises:
        ValueError: If input is invalid.
    """
    # Validate and normalize input
    image, height, width, channels, bit_depth, is_signed = _validate_input(image)

    # Auto-detect intent if not specified
    if intent is None:
        intent = _auto_detect_intent(channels)

    # Create header
    header = create_default_header(
        width=width,
        height=height,
        channels=channels,
        layers=1,
        quality=quality,
        bit_depth=bit_depth,
        is_signed=is_signed,
        filter_type=filter_type,
        encoder=encoder,
        intent=intent,
        chroma_scale=chroma_scale,
    )

    # Convert to internal format (int32 for wavelet processing)
    total_channels = header.layers * header.channels
    aux_data = np.zeros((total_channels, height, width), dtype=np.int32)

    # Copy input data to aux buffer
    boost = 1 if quality == QUALITY_MAX else 8
    for c in range(total_channels):
        if total_channels == 1:
            aux_data[c] = image.astype(np.int32) * boost
        else:
            aux_data[c] = image[:, :, c].astype(np.int32) * boost

    # Track which channels are chroma (for transform program)
    is_chroma = [0] * total_channels

    # Apply forward color transform (currently no transform - identity)
    # In the future, we can add UYV/A710 transforms here

    # Apply forward wavelet transform to each channel
    for c in range(total_channels):
        lift(aux_data[c], 0, 0, width, height, 1, Filter(header.filter))

    # Apply quantization (for lossy compression)
    if quality < QUALITY_MAX:
        chroma_quality = max(1, (quality + chroma_scale // 2) // chroma_scale)
        max_q = QUALITY_MAX * boost

        for c in range(total_channels):
            channel_quality = chroma_quality if is_chroma[c] else quality
            quantize(aux_data[c], 0, 0, width, height, 1, channel_quality, 0, max_q)

    # Encode all blocks
    encoded_data = _encode_all_levels(
        aux_data=aux_data,
        header=header,
        is_chroma=is_chroma,
    )

    # Build final output: header + transform program + encoded blocks
    header_bytes = write_header(header, metadata)

    # Write transform program (minimal: just end marker for identity transform)
    transform_writer = BitWriter(4)  # 4 words should be plenty
    signed_encode(2, -1, transform_writer)  # End marker
    transform_writer.flush_write_word()
    transform_bytes = transform_writer.get_data()

    # Combine all parts
    result_data = header_bytes + transform_bytes + encoded_data

    return EncodeResult(data=result_data, header=header)


def _validate_input(
    image: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int, bool]:
    """
    Validate input image and extract parameters.

    Args:
        image: Input numpy array.

    Returns:
        Tuple of (normalized_image, height, width, channels, bit_depth, is_signed).

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

    if height < 1 or width < 1:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")

    if channels < 1 or channels > 4:
        raise ValueError(f"Channels must be 1-4, got {channels}")

    # Determine bit depth and signedness from dtype
    if image.dtype == np.uint8:
        bit_depth = 8
        is_signed = False
    elif image.dtype == np.int8:
        bit_depth = 8
        is_signed = True
    elif image.dtype == np.uint16:
        bit_depth = 16
        is_signed = False
    elif image.dtype == np.int16:
        bit_depth = 16
        is_signed = True
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}, use uint8/int8/uint16/int16")

    return image, height, width, channels, bit_depth, is_signed


def _auto_detect_intent(channels: int) -> Intent:
    """Auto-detect color intent based on channel count."""
    if channels == 1:
        return Intent.MONO
    elif channels == 3:
        return Intent.RGB
    elif channels == 4:
        return Intent.RGBA
    else:
        return Intent.GENERIC


def _encode_all_levels(  # cm:b4c5d6 — _encode_all_levels(): resolution-level loop (coarse→fine block encoding)
    aux_data: np.ndarray,
    header: GFWXHeader,
    is_chroma: list[int],
) -> bytes:
    """
    Encode all resolution levels for all channels.

    Processes levels from coarsest (DC) to finest, encoding blocks
    for each channel at each level.

    The output format per level is:
    1. Block sizes (4 bytes each, little-endian)
    2. Block data (concatenated, each padded to 4-byte boundary)

    Args:
        aux_data: Coefficient arrays shape (channels, height, width).
        header: Header with encoding parameters.
        is_chroma: Per-channel chroma flags.

    Returns:
        Encoded block data (all levels concatenated).
    """
    total_channels = header.layers * header.channels
    sizex = header.sizex
    sizey = header.sizey
    chroma_quality = max(1, (header.quality + header.chroma_scale // 2) // header.chroma_scale)

    # Find maximum step (coarsest level)
    step = 1
    while step * 2 < sizex or step * 2 < sizey:
        step *= 2

    # Accumulate all encoded data
    output = bytearray()

    # Encode each resolution level
    has_dc = True
    while step >= 1:
        block_size_log = header.block_size

        # Calculate block dimensions for this level
        bs = step << block_size_log

        block_count_x = (sizex + bs - 1) // bs
        block_count_y = (sizey + bs - 1) // bs

        level_block_sizes = []
        level_block_data = []

        # Encode each block (order: channel, then by, then bx)
        for c in range(total_channels):
            for by in range(block_count_y):
                for bx in range(block_count_x):
                    # Calculate block coordinates
                    x0 = bx * bs
                    y0 = by * bs
                    x1 = min((bx + 1) * bs, sizex)
                    y1 = min((by + 1) * bs, sizey)

                    if x0 >= sizex or y0 >= sizey:
                        level_block_sizes.append(0)
                        level_block_data.append(b"")
                        continue

                    # Determine quality for this channel
                    quality = chroma_quality if is_chroma[c] else header.quality

                    # Create writer for this block (estimate max size)
                    block_width = x1 - x0
                    block_height = y1 - y0
                    max_block_words = block_width * block_height * 2 + 16
                    writer = BitWriter(max_block_words)

                    # Encode coefficients for this block
                    encode_coefficients(
                        image=aux_data[c],
                        stream=writer,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        step=step,
                        scheme=Encoder(header.encoder),
                        quality=quality,
                        has_dc=has_dc and bx == 0 and by == 0,
                        is_chroma=is_chroma[c] != 0,
                    )

                    # Flush and get block data
                    writer.flush_write_word()
                    block_bytes = writer.get_data()

                    # Size in 32-bit words
                    block_size_words = (len(block_bytes) + 3) // 4
                    level_block_sizes.append(block_size_words)

                    # Pad to word boundary
                    while len(block_bytes) % 4 != 0:
                        block_bytes += b"\x00"

                    level_block_data.append(block_bytes)

        # Write this level's block sizes first
        for size in level_block_sizes:
            output.extend(size.to_bytes(4, "little"))

        # Then write this level's block data
        for data in level_block_data:
            output.extend(data)

        has_dc = False
        step //= 2

    return bytes(output)
