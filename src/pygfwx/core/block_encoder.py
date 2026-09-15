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

import numpy as np

from pygfwx.core.bitstream import BitWriter
from pygfwx.core.encoder import encode_coefficients
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
from pygfwx.core.transforms import (
    TRANSFORM_A710_RGB,
    forward_transform_generic,
    forward_transform_uyv,
    get_chroma_flags,
    write_transform_program,
)


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


def encode_image(  # cm:e1f2a3 — encode_image(): full encode pipeline (validate→color-transform→lift→quantize→entropy-code)
    image: np.ndarray,
    quality: int = QUALITY_MAX,
    filter_type: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Intent | None = None,
    chroma_scale: int = 1,
    metadata: bytes = b"",
    color_transform: str | None = None,
    max_levels: int | None = None,
    remainder_raw: bool = False,
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
        color_transform: Optional color transform to apply before lifting.
            - None: identity (no transform, default)
            - "uyv": UYV/YUV-like (R-=G, B-=G, G+=(R'+B')/4)
            - "a710": A710 higher-quality color transform
        max_levels: None (default) for standard, unbounded GFWX recursion
            (bit-for-bit unchanged from before this parameter existed).
            A positive integer for the capped-recursion divergence — see
            `lift()`'s own docstring (pygfwx/core/lifting.py) and
            gfwx-fpga's notes/gfwx_capped_recursion_explainer.md for the
            full rationale and real measured cost. Recorded in the
            header so `decode()` doesn't need it passed back in.
            NOTE: not yet verified in combination with Bayer mode or
            `downsampling=` in `decode()` — flagged as a known gap, not
            silently assumed to work.
        remainder_raw: False (default) — the capped remainder is
            recursively transformed and entropy-coded (pygfwx's own
            existing behavior). True — store the remainder as literal,
            uncompressed int16 values instead (no transform, no entropy
            coding); see `header.py`'s own `GFWXHeader.remainder_raw`
            docstring. Only meaningful when `max_levels` is set.
            Recorded in the header; `decode()` needs no corresponding
            argument.

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
        max_levels=max_levels,
        remainder_raw=remainder_raw,
    )

    # Convert to internal format (int32 for wavelet processing)
    total_channels = header.layers * header.channels
    aux_data = np.zeros((total_channels, height, width), dtype=np.int32)

    # Copy input data to aux buffer
    boost = 1 if quality == QUALITY_MAX else 8

    # Apply forward color transform (populates aux_data, returns program + chroma flags)
    transform_program, is_chroma = _apply_forward_transform(image, aux_data, boost, total_channels, color_transform)

    # Apply forward wavelet transform to each channel
    for c in range(total_channels):
        lift(aux_data[c], 0, 0, width, height, 1, Filter(header.filter), max_levels=max_levels, remainder_raw=remainder_raw)

    # Apply quantization (for lossy compression)
    if quality < QUALITY_MAX:
        chroma_quality = max(1, (quality + chroma_scale // 2) // chroma_scale)
        max_q = QUALITY_MAX * boost

        for c in range(total_channels):
            channel_quality = chroma_quality if is_chroma[c] else quality
            quantize(
                aux_data[c], 0, 0, width, height, 1, channel_quality, 0, max_q, max_levels=max_levels, remainder_raw=remainder_raw
            )

    # Encode all blocks
    encoded_data = _encode_all_levels(
        aux_data=aux_data,
        header=header,
        is_chroma=is_chroma,
        max_levels=max_levels,
    )

    # Build final output: header + transform program + encoded blocks
    header_bytes = write_header(header, metadata)

    # Write transform program (identity end-marker, or actual program if transform used)
    transform_writer = BitWriter(16)  # Sufficient for any standard program
    write_transform_program(transform_writer, transform_program)
    transform_bytes = transform_writer.get_data()

    # Combine all parts
    result_data = header_bytes + transform_bytes + encoded_data

    return EncodeResult(data=result_data, header=header)


def _apply_forward_transform(
    image: np.ndarray,
    aux_data: np.ndarray,
    boost: int,
    total_channels: int,
    color_transform: str | None,
) -> tuple[list[int] | None, list[int]]:
    """
    Apply forward color transform and fill aux_data.

    For the identity case (no transform), simply copies image * boost into aux_data.
    For UYV or A710, delegates to the appropriate transform function.

    Args:
        image: Input image array (H, W) or (H, W, C).
        aux_data: Pre-allocated output buffer (C, H, W) int32 — modified in-place.
        boost: Scale factor (1 for lossless, 8 for lossy).
        total_channels: Number of channels.
        color_transform: "uyv", "a710", or None.

    Returns:
        Tuple of (transform_program, is_chroma_flags).
        transform_program is None for identity; a list[int] for named transforms.
    """
    if color_transform is None or total_channels < 3:
        # Identity: copy with boost, no transform
        for c in range(total_channels):
            if total_channels == 1:
                aux_data[c] = image.astype(np.int32) * boost
            else:
                aux_data[c] = image[:, :, c].astype(np.int32) * boost
        return None, [0] * total_channels

    if color_transform == "uyv":
        result, program = forward_transform_uyv(image, boost)
        is_chroma = get_chroma_flags(total_channels, True, program)
    elif color_transform == "a710":
        result, is_chroma = forward_transform_generic(image, TRANSFORM_A710_RGB, boost)
        program = TRANSFORM_A710_RGB
    else:
        raise ValueError(f"Unknown color_transform {color_transform!r}. Use 'uyv', 'a710', or None.")

    # Copy result (C, H, W) into aux_data (same shape)
    for c in range(total_channels):
        aux_data[c] = result[c]

    return program, is_chroma


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


def _encode_remainder_raw(sub: np.ndarray) -> bytes:  # cm:f1a2b3 — raw (uncompressed) remainder storage
    """
    Encode a capped-recursion remainder as literal, uncompressed int16
    values instead of recursively transforming/entropy-coding it — the
    simpler "CineForm-style" alternative documented in gfwx-fpga's own
    notes/gfwx_capped_recursion_explainer.md (used there as the
    pessimistic cost baseline; pygfwx's own default behavior does
    better, see `_encode_all_levels`).

    No length prefix is written: the decoder already knows `sub_h`/
    `sub_w` deterministically from the same header fields the encoder
    used (`remainder_step` is a pure function of `sizex`/`sizey`/
    `max_levels`), so it can compute the exact same byte count itself —
    consistent with this format's existing "no separate length field,
    both sides derive it identically" convention (see
    `_encode_all_levels`'s own docstring on remainder-section framing).

    Args:
        sub: 2D coefficient array (the remainder sub-image for one
            channel), row-major, any numpy integer dtype.

    Returns:
        Little-endian int16 values in row-major order, padded with zero
        bytes to a 4-byte boundary (matching every other section's own
        padding convention).

    Raises:
        ValueError: If any value doesn't fit in a signed 16-bit int —
            fail loudly rather than silently truncate real coefficient
            data.
    """
    lo, hi = int(sub.min()), int(sub.max())
    if lo < -32768 or hi > 32767:
        raise ValueError(
            f"remainder_raw storage requires all remainder values to fit in int16 "
            f"(range -32768..32767); got a value range of {lo}..{hi}"
        )
    # `sub` is a strided view (remainder_step spacing) -- force row-major
    # (C) order explicitly so byte layout doesn't silently follow
    # whatever memory layout the view happens to have.
    raw_bytes = bytearray(sub.astype("<i2", order="C").tobytes(order="C"))
    while len(raw_bytes) % 4 != 0:
        raw_bytes.append(0)
    return bytes(raw_bytes)


def _encode_all_levels(  # cm:b4c5d6 — _encode_all_levels(): resolution-level loop (coarse→fine block encoding)
    aux_data: np.ndarray,
    header: GFWXHeader,
    is_chroma: list[int],
    sizex: int | None = None,
    sizey: int | None = None,
    max_levels: int | None = None,
    total_channels: int | None = None,
) -> bytes:
    """
    Encode all resolution levels for all channels.

    Processes levels from coarsest (DC) to finest, encoding blocks
    for each channel at each level.

    The output format per level is:
    1. Block sizes (4 bytes each, little-endian)
    2. Block data (concatenated, each padded to 4-byte boundary)

    Args:
        aux_data: Coefficient arrays shape (channels, height, width) — for
            a recursive remainder call (see `max_levels` below), a numpy
            strided view into the parent's own array at that channel's
            remainder sub-region, still shape (channels, sub_h, sub_w).
        header: Header with encoding parameters. `header.sizex`/`sizey`
            are used only for the TOP-level call (see `sizex`/`sizey`
            below); quality/chroma_scale/block_size/encoder are reused
            as-is at every recursion depth.
        is_chroma: Per-channel chroma flags.
        sizex, sizey: Region dimensions to encode. Defaults to
            `header.sizex`/`header.sizey` (the top-level call). A
            recursive remainder call passes the remainder's own smaller
            dimensions here instead — `header` itself is NOT copied/
            resized, since nothing here needs its own sizex/sizey once
            these parameters are supplied explicitly.
        max_levels: None (default) for standard, unbounded encoding —
            bit-for-bit unchanged from before this parameter existed.
            When set (from `header.max_levels_or_none`), encodes normally
            through `max_levels` levels for every channel, then — for
            EACH channel independently — recurses into this same
            function on that channel's own remainder sub-array (a numpy
            strided view, matching `lift()`'s own convention), appending
            that channel's own fully-encoded remainder bytes immediately
            after the capped main levels. See `lift()`'s own docstring
            for the full rationale; this must stay structurally
            consistent with however `lift()`/`quantize()` were actually
            called on `aux_data` beforehand, or the block boundaries
            here won't line up with where real (non-padding) coefficients
            actually live.

    Returns:
        Encoded block data (all levels concatenated; remainder sections,
        if any, appended per-channel after the capped main levels).
    """
    if total_channels is None:
        total_channels = header.layers * header.channels
    if sizex is None:
        sizex = header.sizex
    if sizey is None:
        sizey = header.sizey
    chroma_quality = max(1, (header.quality + header.chroma_scale // 2) // header.chroma_scale)

    # Determine the coarsest step ACTUALLY PROCESSED by lift() on this
    # exact (sizex, sizey, max_levels) -- MUST replicate lift()'s own
    # `while step < sizex or step < sizey` condition and counting exactly
    # (NOT the superficially-similar `step*2<sizex` shape below, which
    # finds a different quantity and only coincides with it in the
    # unbounded case -- confirmed by direct trace, this was a real bug in
    # an earlier version of this code). See unlift()'s own matching
    # derivation in lifting.py for why `min_step * 2**(levels_done-1)`
    # is the right closed form once `levels_done` is counted this way.
    step = 1
    levels_done = 0
    capped = False
    while step < sizex or step < sizey:
        if max_levels is not None and levels_done >= max_levels:
            capped = True
            break
        step *= 2
        levels_done += 1
    remainder_step = step  # lift()'s own post-loop step == remainder granularity
    coarsest_step = 1 << (levels_done - 1) if levels_done > 0 else 0

    # Accumulate all encoded data
    output = bytearray()

    if max_levels is not None and capped:
        # Real dependency, not just a layout choice: a "coarsest main
        # level" position's own ancestor context (get_context()'s own
        # `image[py,px]` read) lives at spacing `remainder_step` -- i.e.
        # in the remainder, which is COARSER than anything the main
        # levels below process. `lift()`/`quantize()` already populated
        # every value correctly regardless of order (they ran to
        # completion on the whole array before this function ever
        # started), but `encode_coefficients()`'s own byte-stream
        # ordering doesn't matter for correctness here either way -- the
        # reason this MUST come first is symmetry with `_decode_all_levels`,
        # which genuinely cannot compute correct context for the main
        # levels until the remainder's own (coarser) values exist in its
        # own `aux_data`. Encoding the remainder first here keeps the
        # byte stream in the same order the decoder needs to consume it.
        for c in range(total_channels):
            sub = aux_data[c, 0:sizey:remainder_step, 0:sizex:remainder_step]
            sub_h, sub_w = sub.shape
            if sub_h > 1 or sub_w > 1:
                if header.remainder_raw:
                    output.extend(_encode_remainder_raw(sub))
                else:
                    remainder_bytes = _encode_all_levels(
                        sub[np.newaxis, :, :],
                        header,
                        is_chroma=[is_chroma[c]],
                        sizex=sub_w,
                        sizey=sub_h,
                        max_levels=None,
                        total_channels=1,
                    )
                    output.extend(remainder_bytes)

    # Encode each resolution level, starting at the coarsest step actually
    # processed (see derivation above) rather than whatever `step`
    # happened to be after the tracking loop.
    step = coarsest_step
    # The true DC belongs to whatever recursion actually bottoms out --
    # when capped, that's the remainder's own (just-encoded-above)
    # innermost level, NOT this region's own coarsest processed level.
    has_dc = not (max_levels is not None and capped)
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
