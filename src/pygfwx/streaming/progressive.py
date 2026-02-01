"""
GFWX Progressive Decoding.

This module provides progressive decoding capabilities for GFWX files:
- Decode at reduced resolution for faster preview
- Handle truncated data gracefully
- Report "next point of interest" for streaming applications

Progressive decoding works because GFWX stores coefficients in a
hierarchical manner, from coarsest (DC) to finest detail. Stopping
early gives a lower-resolution but valid image.

Reference: gfwx.h decompress() with downsampling parameter
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from pygfwx.core.bitstream import BitReader
from pygfwx.core.decoder import decode_coefficients
from pygfwx.core.golomb_rice import signed_decode
from pygfwx.core.header import (
    GFWX_MAGIC,
    Filter,
    GFWXHeader,
    HeaderParseError,
    Intent,
    parse_header,
)
from pygfwx.core.lifting import unlift
from pygfwx.core.quantization import QUALITY_MAX, dequantize


class ProgressiveStatus(IntEnum):
    """Status codes for progressive decoding."""

    OK = 0
    """Decode completed successfully."""

    NEED_MORE_DATA = 1
    """More data needed, next_point_of_interest indicates how much."""

    MALFORMED = -1
    """Data is malformed/corrupted."""

    UNSUPPORTED = -2
    """Unsupported file version or feature."""


@dataclass
class ProgressiveResult:
    """Result from progressive decode operation."""

    status: ProgressiveStatus
    """Decode status."""

    image: np.ndarray | None = None
    """Decoded image, may be partial if truncated. Shape (H, W) or (H, W, C)."""

    header: GFWXHeader | None = None
    """Parsed header, or None if header incomplete."""

    next_point_of_interest: int = 0
    """Bytes needed for next meaningful decode step."""

    levels_decoded: int = 0
    """Number of resolution levels successfully decoded."""

    max_levels: int = 0
    """Total number of resolution levels available."""

    actual_downsampling: int = 0
    """Actual downsampling achieved (may be > requested if truncated)."""


@dataclass
class ProgressiveDecoder:
    """
    Stateful progressive decoder for streaming applications.

    Maintains decode state across multiple calls as more data arrives.
    Call feed() with new data chunks and get() to retrieve current image.

    Example:
        decoder = ProgressiveDecoder()
        while streaming:
            data = receive_chunk()
            decoder.feed(data)
            result = decoder.get()
            if result.image is not None:
                display_preview(result.image)
    """

    _data: bytearray = field(default_factory=bytearray)
    """Accumulated data buffer."""

    _header: GFWXHeader | None = None
    """Parsed header (cached)."""

    _header_end: int = 0
    """Byte offset where header ends."""

    _last_result: ProgressiveResult | None = None
    """Most recent decode result."""

    def feed(self, chunk: bytes) -> None:
        """
        Add more data to the decoder.

        Args:
            chunk: New data bytes to append.
        """
        self._data.extend(chunk)
        self._last_result = None  # Invalidate cached result

    def get(self, downsampling: int = 0) -> ProgressiveResult:
        """
        Get current decode result with accumulated data.

        Args:
            downsampling: Desired downsampling (0=full, 1=half, 2=quarter, etc.)

        Returns:
            ProgressiveResult with current image or status.
        """
        return decode_progressive(bytes(self._data), downsampling)

    def reset(self) -> None:
        """Clear all accumulated data and state."""
        self._data.clear()
        self._header = None
        self._header_end = 0
        self._last_result = None

    @property
    def bytes_received(self) -> int:
        """Total bytes accumulated so far."""
        return len(self._data)


def decode_progressive(
    data: bytes,
    downsampling: int = 0,
) -> ProgressiveResult:
    """
    Decode GFWX data progressively with truncation handling.

    This function attempts to decode as much as possible from the provided
    data, returning a usable image even if the data is incomplete.

    Args:
        data: GFWX compressed data (may be truncated).
        downsampling: Desired downsampling factor:
            - 0: Full resolution
            - 1: Half resolution (1/2 x 1/2)
            - 2: Quarter resolution (1/4 x 1/4)
            - etc.

    Returns:
        ProgressiveResult with decoded image and status.

    Example:
        # Streaming decode
        result = decode_progressive(partial_data, downsampling=2)
        if result.status == ProgressiveStatus.NEED_MORE_DATA:
            print(f"Need at least {result.next_point_of_interest} bytes")
        elif result.image is not None:
            show_preview(result.image)
    """
    # Need at least 28 bytes for header
    if len(data) < 28:
        return ProgressiveResult(
            status=ProgressiveStatus.NEED_MORE_DATA,
            next_point_of_interest=28,
        )

    # Check magic
    if int.from_bytes(data[0:4], "little") != GFWX_MAGIC:
        return ProgressiveResult(status=ProgressiveStatus.MALFORMED)

    # Check version before full parse (version is at offset 4-8)
    version = int.from_bytes(data[4:8], "little")
    if version > 1:
        return ProgressiveResult(
            status=ProgressiveStatus.UNSUPPORTED,
            next_point_of_interest=0,
        )

    # Parse header
    try:
        header, header_end = parse_header(data)
    except HeaderParseError as e:
        if "Unsupported version" in str(e):
            return ProgressiveResult(status=ProgressiveStatus.UNSUPPORTED)
        return ProgressiveResult(status=ProgressiveStatus.MALFORMED)
    except ValueError:
        return ProgressiveResult(status=ProgressiveStatus.MALFORMED)

    # Calculate output dimensions
    sizex_down = (header.sizex + (1 << downsampling) - 1) >> downsampling
    sizey_down = (header.sizey + (1 << downsampling) - 1) >> downsampling
    total_channels = header.layers * header.channels

    # Calculate max levels
    max_step = 1
    while max_step * 2 < header.sizex or max_step * 2 < header.sizey:
        max_step *= 2
    max_levels = 0
    temp_step = max_step
    while (temp_step >> downsampling) >= 1:
        max_levels += 1
        temp_step //= 2

    # Allocate output buffer
    aux_data = np.zeros((total_channels, sizey_down, sizex_down), dtype=np.int32)
    is_chroma = [0] * total_channels

    # Parse transform program
    if header_end >= len(data):
        return ProgressiveResult(
            status=ProgressiveStatus.NEED_MORE_DATA,
            header=header,
            next_point_of_interest=header_end + 64,  # Guess
            max_levels=max_levels,
        )

    stream = BitReader(data[header_end:])

    try:
        transform_program, transform_steps, is_chroma = _parse_transform_program_safe(stream, total_channels)
    except _TruncationError:
        return ProgressiveResult(
            status=ProgressiveStatus.NEED_MORE_DATA,
            header=header,
            next_point_of_interest=len(data) + 256,
            max_levels=max_levels,
        )
    except ValueError:
        return ProgressiveResult(status=ProgressiveStatus.MALFORMED, header=header)

    stream.flush_read_word()

    # Calculate quality values
    chroma_quality = max(1, (header.quality + header.chroma_scale // 2) // header.chroma_scale)
    boost = 1 if header.quality == QUALITY_MAX else 8

    # Offset after transform program
    transform_end = header_end + stream.word_index * 4

    # Decode coefficient blocks progressively
    levels_decoded, is_truncated, next_poi = _decode_levels_progressive(
        aux_data=aux_data,
        data=data,
        stream_offset=transform_end,
        header=header,
        sizex_down=sizex_down,
        sizey_down=sizey_down,
        downsampling=downsampling,
        is_chroma=is_chroma,
        chroma_quality=chroma_quality,
    )

    # Calculate actual downsampling achieved
    actual_downsampling = downsampling
    if is_truncated and levels_decoded < max_levels:
        # We decoded fewer levels than requested
        actual_downsampling = downsampling + (max_levels - levels_decoded)

    # Dequantize and unlift each channel (only for decoded levels)
    if levels_decoded > 0:
        for c in range(total_channels):
            channel_data = aux_data[c]

            # Determine step based on levels decoded
            current_step = max_step
            for _ in range(levels_decoded):
                current_step //= 2
            min_step = max(1, current_step >> downsampling)

            # Check for Bayer mode
            is_bayer = Intent.BAYER_RGGB <= header.intent <= Intent.BAYER_GENERIC

            if is_bayer:
                # Bayer mode: process 4 sub-grids separately
                for ox in range(2):
                    for oy in range(2):
                        if ox == 0 and oy == 0:
                            continue  # Skip first, done below with unlift
                        q = chroma_quality if (ox or oy) else header.quality
                        dequantize(
                            channel_data,
                            ox,
                            oy,
                            sizex_down,
                            sizey_down,
                            2,
                            q << actual_downsampling,
                            header.quality,
                            QUALITY_MAX * boost,
                        )
                        unlift(channel_data, ox, oy, sizex_down, sizey_down, 2, Filter(header.filter))

            # Standard dequantize for non-Bayer or for main grid
            if header.quality < QUALITY_MAX:
                q = chroma_quality if is_chroma[c] else header.quality
                dequantize(
                    channel_data,
                    0,
                    0,
                    sizex_down,
                    sizey_down,
                    1,
                    q << actual_downsampling,
                    0,
                    QUALITY_MAX * boost,
                )

            # Inverse wavelet transform
            unlift(channel_data, 0, 0, sizex_down, sizey_down, min_step, Filter(header.filter))

        # Apply inverse color transform (if present)
        if transform_program and transform_steps:
            _apply_inverse_transform(aux_data, transform_program, transform_steps, is_chroma, boost)

    # Convert to output format
    image = _convert_to_output(aux_data, header, sizex_down, sizey_down, boost)

    status = ProgressiveStatus.NEED_MORE_DATA if is_truncated else ProgressiveStatus.OK

    return ProgressiveResult(
        status=status,
        image=image,
        header=header,
        next_point_of_interest=next_poi,
        levels_decoded=levels_decoded,
        max_levels=max_levels,
        actual_downsampling=actual_downsampling,
    )


class _TruncationError(Exception):
    """Internal exception for truncation during parsing."""

    pass


def _parse_transform_program_safe(stream: BitReader, num_channels: int) -> tuple[list[int], list[int], list[int]]:
    """
    Parse transform program with truncation detection.

    Raises _TruncationError if data is truncated.
    """
    program = []
    steps = []
    is_chroma = [0] * num_channels

    while True:
        if stream.overflow:
            raise _TruncationError()

        channel = signed_decode(2, stream)
        program.append(channel)

        if channel < 0:
            break
        if channel >= num_channels:
            raise ValueError(f"Transform channel {channel} >= num_channels")

        steps.append(len(program) - 1)

        while True:
            if stream.overflow:
                raise _TruncationError()

            other_channel = signed_decode(2, stream)
            program.append(other_channel)

            if other_channel < 0:
                break
            if other_channel >= num_channels:
                raise ValueError(f"Transform other channel {other_channel} >= num_channels")

            factor = signed_decode(2, stream)
            program.append(factor)

        denominator = signed_decode(2, stream)
        program.append(denominator)

        chroma_flag = signed_decode(2, stream)
        program.append(chroma_flag)
        is_chroma[channel] = chroma_flag

    return program, steps, is_chroma


def _decode_levels_progressive(
    aux_data: np.ndarray,
    data: bytes,
    stream_offset: int,
    header: GFWXHeader,
    sizex_down: int,
    sizey_down: int,
    downsampling: int,
    is_chroma: list[int],
    chroma_quality: int,
) -> tuple[int, bool, int]:
    """
    Decode resolution levels progressively with truncation handling.

    Returns:
        Tuple of (levels_decoded, is_truncated, next_point_of_interest).
    """
    total_channels = header.layers * header.channels
    is_truncated = False
    levels_decoded = 0

    # Find maximum step
    step = 1
    while step * 2 < header.sizex or step * 2 < header.sizey:
        step *= 2

    pos = stream_offset
    next_poi = len(data) + 1024  # Initial guess

    has_dc = True
    while (step >> downsampling) >= 1:
        step_down = step >> downsampling
        block_size_log = header.block_size

        # Calculate block dimensions
        bs = step << block_size_log
        bs_down = step_down << block_size_log

        block_count_x = (header.sizex + bs - 1) // bs
        block_count_y = (header.sizey + bs - 1) // bs
        block_count = block_count_x * block_count_y * total_channels

        # Check if we have block sizes
        block_sizes_end = pos + block_count * 4
        if block_sizes_end > len(data):
            is_truncated = True
            next_poi = block_sizes_end
            break

        # Read block sizes
        block_sizes = []
        for _ in range(block_count):
            size = int.from_bytes(data[pos : pos + 4], "little")
            block_sizes.append(size)
            pos += 4

        # Calculate total block data size
        total_block_data = sum(block_sizes) * 4
        block_data_end = pos + total_block_data

        # Estimate next point of interest
        if (step >> downsampling) > 1:
            next_poi = block_data_end + block_count * 4
        else:
            next_poi = block_data_end

        # Check if we have all block data
        if block_data_end > len(data):
            is_truncated = True
            break

        # Decode each block
        block_start = pos
        for block_idx in range(block_count):
            bx = block_idx % block_count_x
            by = (block_idx // block_count_x) % block_count_y
            c = block_idx // (block_count_x * block_count_y)

            block_end = block_start + block_sizes[block_idx] * 4

            if block_end > len(data):
                is_truncated = True
                break

            # Calculate block bounds
            x0 = int(bx * bs_down)
            y0 = int(by * bs_down)
            x1 = int(min((bx + 1) * bs_down, sizex_down))
            y1 = int(min((by + 1) * bs_down, sizey_down))

            if x0 < x1 and y0 < y1:
                block_data = data[block_start:block_end]
                stream = BitReader(block_data)

                try:
                    # Check for Bayer mode
                    is_bayer = Intent.BAYER_RGGB <= header.intent <= Intent.BAYER_GENERIC

                    if is_bayer:
                        # Bayer: decode 4 sub-grids
                        for ox in range(2):
                            for oy in range(2):
                                q = chroma_quality if (ox or oy) else header.quality
                                decode_coefficients(
                                    aux_data[c],
                                    stream,
                                    x0 + ox,
                                    y0 + oy,
                                    x1,
                                    y1,
                                    2 * step_down,
                                    header.encoder,
                                    q,
                                    has_dc and bx == 0 and by == 0,
                                    bool(ox or oy),
                                )
                    else:
                        decode_coefficients(
                            aux_data[c],
                            stream,
                            x0,
                            y0,
                            x1,
                            y1,
                            step_down,
                            header.encoder,
                            chroma_quality if is_chroma[c] else header.quality,
                            has_dc and bx == 0 and by == 0,
                            bool(is_chroma[c]),
                        )
                except Exception:
                    # Block decode failed, data likely corrupted
                    is_truncated = True
                    break

            block_start = block_end

        if is_truncated:
            break

        pos = block_start
        levels_decoded += 1
        has_dc = False
        step //= 2

    return levels_decoded, is_truncated, next_poi


def _apply_inverse_transform(
    aux_data: np.ndarray,
    program: list[int],
    steps: list[int],
    is_chroma: list[int],
    boost: int,
) -> None:
    """Apply inverse color transform in reverse order."""
    # Process steps in reverse
    for s in range(len(steps) - 1, -1, -1):
        pc = steps[s]
        dest_channel = program[pc]
        pc += 1

        # Compute transform term
        transform_temp = np.zeros((aux_data.shape[1], aux_data.shape[2]), dtype=np.int32)

        while program[pc] >= 0:
            src_channel = program[pc]
            pc += 1
            factor = program[pc]
            pc += 1

            if is_chroma[src_channel] == -1:
                # Source from original image (not implemented for progressive)
                transform_temp += aux_data[src_channel] * (boost * factor)
            else:
                transform_temp += aux_data[src_channel] * factor

        pc += 1  # Skip -1 terminator
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

        # Subtract from destination
        aux_data[dest_channel] -= transform_temp


def _convert_to_output(
    aux_data: np.ndarray,
    header: GFWXHeader,
    sizex: int,
    sizey: int,
    boost: int,
) -> np.ndarray:
    """Convert internal aux data to output image format."""
    total_channels = header.layers * header.channels

    # Determine output dtype based on bit depth
    if header.bit_depth <= 8:
        out_dtype = np.uint8 if not header.is_signed else np.int8
        min_val = -128 if header.is_signed else 0
        max_val = 127 if header.is_signed else 255
    else:
        out_dtype = np.uint16 if not header.is_signed else np.int16
        min_val = -32768 if header.is_signed else 0
        max_val = 32767 if header.is_signed else 65535

    # Create output array
    if total_channels == 1:
        output = np.zeros((sizey, sizex), dtype=out_dtype)
    else:
        output = np.zeros((sizey, sizex, header.channels), dtype=out_dtype)

    # Copy channels with boost division and clamping
    for c in range(total_channels):
        layer = c // header.channels
        channel_in_layer = c % header.channels

        src = aux_data[c]
        if boost > 1:
            src = src // boost

        src = np.clip(src, min_val, max_val).astype(out_dtype)

        if total_channels == 1:
            output[:, :] = src
        else:
            # For now, only handle layer 0
            if layer == 0:
                output[:, :, channel_in_layer] = src

    return output
