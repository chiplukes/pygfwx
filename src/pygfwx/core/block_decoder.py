"""
GFWX Block Decoder - High-level decoding pipeline.

This module implements the full GFWX decode pipeline that:
1. Parses the file header and metadata
2. Iterates through resolution levels (coarse to fine)
3. Decodes coefficient blocks for each channel
4. Applies dequantization for lossy files
5. Applies inverse wavelet transform (unlift)
6. Applies inverse color transform (if present)

The block structure allows for:
- Progressive decoding (decode partial file for preview)
- Parallel decoding of independent blocks
- Memory-efficient processing

Bayer Mode Support:
- Detects Bayer intents (RGGB, BGGR, GRBG, GBRG, GENERIC)
- Processes 4 sub-images separately with step=2
- Uses chroma quality for (ox|oy) != 0 sub-images
"""

from dataclasses import dataclass

import numpy as np

from pygfwx.core.bayer import (
    get_quality_for_subimage,
    intent_is_bayer,
    iter_bayer_offsets,
    iter_bayer_offsets_for_lifting,
)
from pygfwx.core.bitstream import BitReader
from pygfwx.core.decoder import decode_coefficients
from pygfwx.core.header import GFWX_MAGIC, Encoder, Filter, GFWXHeader, parse_header
from pygfwx.core.lifting import unlift
from pygfwx.core.quantization import QUALITY_MAX, dequantize


@dataclass
class DecodeResult:  # cm:e7f8a9 — DecodeResult dataclass: decoded image + header + truncation flag
    """Result of decoding operation."""

    image: np.ndarray
    """Decoded image data with shape (height, width, channels) or (height, width) for mono."""

    header: GFWXHeader
    """Parsed header information."""

    is_truncated: bool = False
    """True if the input data was truncated before complete decode."""


@dataclass
class BlockInfo:  # cm:b0c1d2 — BlockInfo dataclass: per-block spatial coordinates and word-size
    """Information about a single block."""

    bx: int
    """Block x index."""

    by: int
    """Block y index."""

    channel: int
    """Channel index."""

    x0: int
    """Block left coordinate."""

    y0: int
    """Block top coordinate."""

    x1: int
    """Block right coordinate (exclusive)."""

    y1: int
    """Block bottom coordinate (exclusive)."""

    size_words: int
    """Size of block data in 32-bit words."""


def decode_image(  # cm:e3f4a5 — decode_image(): full decode pipeline (header→transform→entropy→unlift→output)
    data: bytes,
    downsampling: int = 0,
) -> DecodeResult:
    """
    Decode a GFWX compressed image.

    This is the main entry point for decoding GFWX files. It handles the
    complete decode pipeline from raw bytes to reconstructed image.

    Args:
        data: Raw GFWX file data.
        downsampling: Downsampling factor (0 = full size, 1 = half, 2 = quarter, etc.).

    Returns:
        DecodeResult containing the decoded image and header.

    Raises:
        ValueError: If the data is malformed or unsupported.
    """
    # Parse header (returns tuple of header and header size)
    header, header_end = parse_header(data)

    # Calculate output dimensions with downsampling
    sizex_down = (header.sizex + (1 << downsampling) - 1) >> downsampling
    sizey_down = (header.sizey + (1 << downsampling) - 1) >> downsampling

    # Allocate output buffer for all channels
    total_channels = header.layers * header.channels
    aux_data = np.zeros((total_channels, sizey_down, sizex_down), dtype=np.int32)

    # Track which channels are chroma (from transform program)
    is_chroma = [0] * total_channels

    # Verify magic
    if len(data) < 28:
        raise ValueError("Data too short for GFWX header")
    if int.from_bytes(data[0:4], "little") != GFWX_MAGIC:
        raise ValueError("Invalid GFWX magic number")

    # Parse transform program (after header/metadata)
    stream = BitReader(data[header_end:])
    transform_program, transform_steps, is_chroma = _parse_transform_program(
        stream, total_channels
    )
    stream.flush_read_word()

    # Calculate quality values
    chroma_quality = max(1, (header.quality + header.chroma_scale // 2) // header.chroma_scale)
    boost = 1 if header.quality == 1024 else 8

    # Calculate offset after transform program (word-aligned)
    # stream.word_index gives us the word position after flush
    transform_end = header_end + stream.word_index * 4

    # Check if this is Bayer mode
    is_bayer = intent_is_bayer(header.intent)

    # Decode coefficient blocks for each resolution level
    is_truncated = _decode_all_levels(
        aux_data=aux_data,
        data=data,
        stream_offset=transform_end,
        header=header,
        sizex_down=sizex_down,
        sizey_down=sizey_down,
        downsampling=downsampling,
        is_chroma=is_chroma,
        chroma_quality=chroma_quality,
        is_bayer=is_bayer,
    )

    # Dequantize and unlift each channel
    for c in range(total_channels):
        channel_data = aux_data[c]

        if is_bayer:
            # Bayer mode: process 4 sub-images separately
            _dequantize_and_unlift_bayer(
                channel_data,
                sizex_down,
                sizey_down,
                header,
                downsampling,
                chroma_quality,
                boost,
            )
        else:
            # Normal mode: process full image
            # Dequantize (for lossy compression)
            if header.quality < QUALITY_MAX:
                dequantize(
                    channel_data,
                    0,
                    0,
                    sizex_down,
                    sizey_down,
                    1,
                    (chroma_quality if is_chroma[c] else header.quality) << downsampling,
                    0,
                    QUALITY_MAX * boost,
                )

            # Inverse wavelet transform
            unlift(channel_data, 0, 0, sizex_down, sizey_down, 1, Filter(header.filter))

    # Apply inverse color transform (if present)
    if transform_program and transform_steps:
        _apply_inverse_transform(
            aux_data, transform_program, transform_steps, is_chroma, boost
        )

    # Convert to output format
    image = _convert_to_output(aux_data, header, sizex_down, sizey_down, boost)

    return DecodeResult(image=image, header=header, is_truncated=is_truncated)


def _dequantize_and_unlift_bayer(
    channel_data: np.ndarray,
    sizex_down: int,
    sizey_down: int,
    header: GFWXHeader,
    downsampling: int,
    chroma_quality: int,
    boost: int,
) -> None:
    """
    Apply Bayer-specific dequantization and inverse wavelet transform.

    In Bayer mode, each of the 4 sub-images (2x2 decimation) is processed
    separately with step=2. The sub-image at (0,0) uses luma quality,
    while others use chroma quality.

    Args:
        channel_data: 2D coefficient array (modified in-place).
        sizex_down: Downsampled width.
        sizey_down: Downsampled height.
        header: Parsed header.
        downsampling: Downsampling factor.
        chroma_quality: Quality for chroma sub-images.
        boost: Quality boost factor (1 for lossless, 8 for lossy).

    Note:
        The processing order matches the SDK:
        1. Dequantize each sub-image with appropriate quality
        2. Unlift additional sub-images (0,1), (1,0), (1,1)
        3. Unlift full image starting at (0,0)
    """
    # Dequantize each sub-image (for lossy compression)
    if header.quality < QUALITY_MAX:
        for ox, oy in iter_bayer_offsets():
            # (0,0) uses luma quality, others use chroma quality
            sub_quality = get_quality_for_subimage(ox, oy, header.quality, chroma_quality)
            dequantize(
                channel_data,
                ox,
                oy,
                sizex_down,
                sizey_down,
                2,  # step=2 for Bayer sub-images
                sub_quality << downsampling,
                header.quality,
                QUALITY_MAX * boost,
            )

    # Unlift additional sub-images first (order: (0,1), (1,0), (1,1))
    for ox, oy in iter_bayer_offsets_for_lifting():
        unlift(channel_data, ox, oy, sizex_down, sizey_down, 2, Filter(header.filter))

    # Then unlift the full image starting at (0,0)
    unlift(channel_data, 0, 0, sizex_down, sizey_down, 1, Filter(header.filter))


def _parse_transform_program(
    stream: BitReader, num_channels: int
) -> tuple[list[int], list[int], list[int]]:
    """
    Parse the color transform program from the bitstream.

    The transform program defines how channels are combined. Common
    transforms include YUV (luminance-chrominance) and A710.

    Args:
        stream: BitReader positioned at the transform program.
        num_channels: Total number of channels.

    Returns:
        Tuple of (program, step_indices, is_chroma_flags).
    """
    from pygfwx.core.golomb_rice import signed_decode

    program = []
    steps = []
    is_chroma = [0] * num_channels

    while True:
        channel = signed_decode(2, stream)
        program.append(channel)

        if channel < 0:
            # End of program
            break
        if channel >= num_channels:
            raise ValueError(f"Transform channel {channel} >= num_channels {num_channels}")

        steps.append(len(program) - 1)

        # Read term sequence for this step
        while True:
            other_channel = signed_decode(2, stream)
            program.append(other_channel)

            if other_channel < 0:
                break
            if other_channel >= num_channels:
                raise ValueError(
                    f"Transform other channel {other_channel} >= num_channels"
                )

            factor = signed_decode(2, stream)
            program.append(factor)

        denominator = signed_decode(2, stream)
        program.append(denominator)

        chroma_flag = signed_decode(2, stream)
        program.append(chroma_flag)
        is_chroma[channel] = chroma_flag

    return program, steps, is_chroma


def _decode_all_levels(
    aux_data: np.ndarray,
    data: bytes,
    stream_offset: int,
    header: GFWXHeader,
    sizex_down: int,
    sizey_down: int,
    downsampling: int,
    is_chroma: list[int],
    chroma_quality: int,
    is_bayer: bool = False,
) -> bool:
    """
    Decode all resolution levels for all channels.

    Processes levels from coarsest (DC) to finest, decoding blocks
    for each channel at each level.

    Args:
        aux_data: Output coefficient arrays shape (channels, height, width).
        data: Raw compressed data.
        stream_offset: Byte offset where block data starts.
        header: Parsed header.
        sizex_down: Downsampled width.
        sizey_down: Downsampled height.
        downsampling: Downsampling factor.
        is_chroma: Per-channel chroma flags.
        chroma_quality: Quality for chroma channels.
        is_bayer: If True, decode with Bayer sub-image offsets.

    Returns:
        True if data was truncated.
    """
    total_channels = header.layers * header.channels
    is_truncated = False

    # Find maximum step (coarsest level)
    step = 1
    while step * 2 < header.sizex or step * 2 < header.sizey:
        step *= 2

    # Current position in the data
    pos = stream_offset

    # Decode each resolution level
    has_dc = True
    while (step >> downsampling) >= 1:
        step_down = step >> downsampling
        block_size_log = header.block_size  # block_size is already parsed as log2 + 2

        # Calculate block dimensions for this level
        bs = step << block_size_log
        bs_down = step_down << block_size_log

        block_count_x = (header.sizex + bs - 1) // bs
        block_count_y = (header.sizey + bs - 1) // bs
        block_count = block_count_x * block_count_y * total_channels

        # Check if we have enough data for block sizes
        block_sizes_end = pos + block_count * 4
        if block_sizes_end > len(data):
            is_truncated = True
            break

        # Read block sizes
        block_sizes = []
        for _ in range(block_count):
            size = int.from_bytes(data[pos : pos + 4], "little")
            block_sizes.append(size)
            pos += 4

        # Calculate block data positions
        block_data_start = pos
        block_positions = []
        current_pos = block_data_start
        for size in block_sizes:
            block_positions.append(current_pos)
            current_pos += size * 4

        # Check if we have all block data
        if current_pos > len(data):
            is_truncated = True
            # Process available blocks anyway
            current_pos = min(current_pos, len(data))

        # Decode each block
        for block_idx in range(block_count):
            bx = block_idx % block_count_x
            by = (block_idx // block_count_x) % block_count_y
            c = block_idx // (block_count_x * block_count_y)

            # Calculate block coordinates in downsampled space
            x0 = int(bx * bs_down)
            y0 = int(by * bs_down)
            x1 = int(min((bx + 1) * bs_down, sizex_down))
            y1 = int(min((by + 1) * bs_down, sizey_down))

            if x0 >= sizex_down or y0 >= sizey_down:
                continue

            # Check if block data is available
            block_pos = block_positions[block_idx]
            block_end = block_pos + block_sizes[block_idx] * 4
            if block_end > len(data):
                continue

            # Create reader for this block
            block_data = data[block_pos:block_end]
            if len(block_data) == 0:
                continue

            block_stream = BitReader(block_data)

            # Determine quality for this channel
            quality = chroma_quality if is_chroma[c] else header.quality

            # Decode coefficients for this block
            try:
                if is_bayer:
                    # Bayer mode: decode each of the 4 sub-images
                    for ox, oy in iter_bayer_offsets():
                        sub_quality = get_quality_for_subimage(ox, oy, header.quality, chroma_quality)
                        decode_coefficients(
                            image=aux_data[c],
                            stream=block_stream,
                            x0=x0 + ox,
                            y0=y0 + oy,
                            x1=x1,
                            y1=y1,
                            step=2 * step_down,  # Double step for sub-images
                            scheme=Encoder(header.encoder),
                            quality=sub_quality,
                            has_dc=has_dc and bx == 0 and by == 0,
                            is_chroma=(ox | oy) != 0,  # (0,0) is luma, others are chroma
                        )
                else:
                    # Normal mode
                    decode_coefficients(
                        image=aux_data[c],
                        stream=block_stream,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        step=step_down,
                        scheme=Encoder(header.encoder),
                        quality=quality,
                        has_dc=has_dc and bx == 0 and by == 0,
                        is_chroma=is_chroma[c] != 0,
                    )
            except Exception:
                # Block decode failed, likely truncated
                is_truncated = True

        # Move position past all block data
        pos = current_pos
        has_dc = False
        step //= 2

    return is_truncated


def _apply_inverse_transform(
    aux_data: np.ndarray,
    program: list[int],
    steps: list[int],
    is_chroma: list[int],
    _boost: int,
) -> None:
    """
    Apply inverse color transform.

    Runs the transform program in reverse order to convert from
    transformed space (e.g., YUV) back to original (e.g., RGB).

    Args:
        aux_data: Coefficient arrays shape (channels, height, width).
        program: Transform program instructions.
        steps: Indices into program for each step.
        is_chroma: Per-channel chroma flags.
        _boost: Boost factor (1 for lossless, 8 for lossy) - unused in transform.
    """
    height, width = aux_data.shape[1], aux_data.shape[2]
    buffer_size = height * width

    # Process steps in reverse order
    for s in range(len(steps) - 1, -1, -1):
        pc = steps[s]
        channel = program[pc]
        pc += 1

        # Compute transform term
        transform_temp = np.zeros(buffer_size, dtype=np.int64)

        while program[pc] >= 0:
            other_channel = program[pc]
            pc += 1
            factor = program[pc]
            pc += 1

            if is_chroma[other_channel] == -1:
                # Source channel (not yet transformed)
                # This case shouldn't happen in decode
                pass
            else:
                # Use aux_data channel
                other_data = aux_data[other_channel].reshape(-1)
                transform_temp += other_data.astype(np.int64) * factor

        # Skip the -1 marker
        pc += 1

        # Read denominator
        denom = program[pc]

        # Apply division
        if denom == 2:
            transform_temp >>= 1
        elif denom == 4:
            transform_temp >>= 2
        elif denom == 8:
            transform_temp >>= 3
        elif denom > 1:
            transform_temp //= denom

        # Subtract from destination channel
        dest = aux_data[channel].reshape(-1)
        dest -= transform_temp.astype(np.int32)


def _convert_to_output(
    aux_data: np.ndarray,
    header: GFWXHeader,
    width: int,
    height: int,
    boost: int,
) -> np.ndarray:
    """
    Convert internal representation to output image format.

    Applies boost factor division and clips to valid range.

    Args:
        aux_data: Coefficient arrays shape (channels, height, width).
        header: Parsed header.
        width: Output width.
        height: Output height.
        boost: Boost factor (1 for lossless, 8 for lossy).

    Returns:
        Output image array.
    """
    total_channels = header.layers * header.channels

    # Determine output dtype based on bit depth
    if header.bit_depth <= 8:
        if header.is_signed:
            dtype = np.int8
            min_val, max_val = -128, 127
        else:
            dtype = np.uint8
            min_val, max_val = 0, 255
    else:
        if header.is_signed:
            dtype = np.int16
            min_val, max_val = -32768, 32767
        else:
            dtype = np.uint16
            min_val, max_val = 0, 65535

    # Create output array
    if total_channels == 1:
        output = np.zeros((height, width), dtype=dtype)
    else:
        output = np.zeros((height, width, total_channels), dtype=dtype)

    # Convert each channel
    for c in range(total_channels):
        source = aux_data[c]

        if boost == 1:
            # Lossless: direct copy with clipping
            clipped = np.clip(source, min_val, max_val)
        else:
            # Lossy: divide by boost, then clip
            divided = source // boost
            clipped = np.clip(divided, min_val, max_val)

        if total_channels == 1:
            output[:, :] = clipped.astype(dtype)
        else:
            output[:, :, c] = clipped.astype(dtype)

    return output


def get_block_info(
    header: GFWXHeader,
    step: int,
    downsampling: int = 0,
) -> list[BlockInfo]:
    """
    Get information about blocks for a given resolution level.

    Useful for understanding the block structure or implementing
    partial decoding.

    Args:
        header: Parsed header.
        step: Resolution level step.
        downsampling: Downsampling factor.

    Returns:
        List of BlockInfo for all blocks at this level.
    """
    total_channels = header.layers * header.channels
    step_down = step >> downsampling

    sizex_down = (header.sizex + (1 << downsampling) - 1) >> downsampling
    sizey_down = (header.sizey + (1 << downsampling) - 1) >> downsampling

    bs = step << header.block_size
    bs_down = step_down << header.block_size

    block_count_x = (header.sizex + bs - 1) // bs
    block_count_y = (header.sizey + bs - 1) // bs

    blocks = []
    for c in range(total_channels):
        for by in range(block_count_y):
            for bx in range(block_count_x):
                x0 = int(bx * bs_down)
                y0 = int(by * bs_down)
                x1 = int(min((bx + 1) * bs_down, sizex_down))
                y1 = int(min((by + 1) * bs_down, sizey_down))

                blocks.append(
                    BlockInfo(
                        bx=bx,
                        by=by,
                        channel=c,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        size_words=0,
                    )
                )

    return blocks
