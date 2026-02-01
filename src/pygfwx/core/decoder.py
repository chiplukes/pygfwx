"""
GFWX Coefficient Decoder.

This module implements the main coefficient decoding loop that combines:
- Context modeling for adaptive coding
- Golomb-Rice decoding (interleaved and signed variants)
- Run-length decoding for zero coefficients

The decoder processes wavelet coefficients in a specific order
(arranged so that (x | y) & step == 1) and uses context from
previously decoded neighbors to select the optimal decoding mode.
"""

import numpy as np

from pygfwx.core.bitstream import BitReader
from pygfwx.core.context import Context, get_context, update_fast_context
from pygfwx.core.golomb_rice import interleaved_decode, signed_decode, unsigned_decode
from pygfwx.core.header import Encoder


def decode_coefficients(
    image: np.ndarray,
    stream: BitReader,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: int,
    scheme: Encoder,
    quality: int,
    has_dc: bool,
    is_chroma: bool,
) -> None:
    """
    Decode wavelet coefficients from the bitstream into the image array.

    This function decodes coefficients for a block of the image, using
    context-adaptive Golomb-Rice coding with run-length encoding for zeros.

    Args:
        image: 2D numpy array to store decoded coefficients (modified in-place).
        stream: BitReader positioned at the start of coefficient data.
        x0, y0: Top-left corner of the block.
        x1, y1: Bottom-right corner (exclusive) of the block.
        step: Wavelet level step size (1 for full resolution).
        scheme: Encoder type (TURBO, FAST, CONTEXTUAL, HIGH_BITRATE).
        quality: Quality parameter (1024 = lossless).
        has_dc: True if this block contains the DC coefficient.
        is_chroma: True if this is a chroma channel (affects thresholds).

    Note:
        The decoding order is arranged so that (x | y) & step == 1,
        which ensures we process detail coefficients after their
        dependencies (approximation coefficients) are available.
    """
    sizex = x1 - x0
    sizey = y1 - y0

    # Decode DC coefficient if present
    if has_dc and sizex > 0 and sizey > 0:
        image[y0, x0] = signed_decode(4, stream)

    # Initialize context (first moment, second moment)
    context = Context(sum=0, sum2=0)

    # Run-length state
    # run = -1 means we need to decode a new run length
    # run > 0 means we have that many zeros left in the run
    # run = 0 means run just ended, next non-zero has been shifted
    run = -1

    # Initial run coder parameter
    if scheme == Encoder.TURBO:
        # Turbo mode: simple run coding based on quality and step
        # Avoid overflow by checking q * step < 2048
        if quality == 0 or (step < 2048 and quality * step < 2048):
            run_coder = 1
        else:
            run_coder = 0
    else:
        run_coder = 0

    # Process coefficients in the specific order
    for y in range(0, sizey, step):
        # xStep alternates based on y position
        # This ensures (x | y) & step == 1 for all processed positions
        x_step = step if (y & step) else step * 2

        for x in range(x_step - step, sizex, x_step):
            s = 0

            # Check if we need to decode a new run length
            if run_coder and run == -1:
                run = unsigned_decode(run_coder, stream)

            if run > 0:
                # Consume a zero from the run
                run -= 1
            else:
                # Decode a coefficient
                if scheme == Encoder.TURBO:
                    s = interleaved_decode(1, stream)

                elif scheme == Encoder.HIGH_BITRATE:
                    # Use bit_width of context for pot selection
                    bit_width = context.sum.bit_length() if context.sum > 0 else 0
                    pot = max(0, bit_width - 4)
                    s = interleaved_decode(pot, stream)
                    # Update context with decaying moment
                    t = abs(s)
                    new_sum = ((context.sum * 15 + 7) >> 4) + t
                    context = Context(sum=new_sum, sum2=context.sum2)

                else:
                    # FAST or CONTEXTUAL mode
                    if scheme == Encoder.CONTEXTUAL:
                        context = get_context(image, x0, y0, x1, y1, x, y, step)

                    # Select coding mode based on context statistics
                    s = _decode_with_context(context, stream, is_chroma)

                    # Update context and run coder
                    if scheme == Encoder.FAST:
                        context = update_fast_context(context, s)
                        # Update run coder if s and runCoder have same zero-ness
                        if bool(s) == bool(run_coder):
                            run_coder = _compute_run_coder_fast(context)
                    else:
                        # CONTEXTUAL mode run coder update
                        if bool(s) == bool(run_coder):
                            run_coder = _compute_run_coder_contextual(
                                context, quality
                            )

                # Handle run break: if run was 0 and s <= 0, shift negatives
                if run == 0 and s <= 0:
                    s -= 1
                run = -1

            # Store the decoded coefficient
            image[y0 + y, x0 + x] = s


def _decode_with_context(context: Context, stream: BitReader, is_chroma: bool) -> int:
    """
    Decode a coefficient using context-adaptive mode selection.

    The decision tree selects between interleaved and signed Golomb-Rice
    coding based on the relationship between sum² and sum2 (second moment).

    Args:
        context: Current context statistics.
        stream: BitReader to decode from.
        is_chroma: True if chroma channel (uses higher threshold).

    Returns:
        The decoded coefficient value.
    """
    sum_sq = context.sum * context.sum
    sum2 = context.sum2
    threshold = 250 if is_chroma else 100

    if sum_sq < 2 * sum2 + threshold:
        return interleaved_decode(0, stream)
    elif sum_sq < 2 * sum2 + 950:
        return interleaved_decode(1, stream)
    elif sum_sq < 3 * sum2 + 3000:
        if sum_sq < 5 * sum2 + 400:
            return signed_decode(1, stream)
        else:
            return interleaved_decode(2, stream)
    elif sum_sq < 3 * sum2 + 12000:
        if sum_sq < 5 * sum2 + 3000:
            return signed_decode(2, stream)
        else:
            return interleaved_decode(3, stream)
    elif sum_sq < 4 * sum2 + 44000:
        if sum_sq < 6 * sum2 + 12000:
            return signed_decode(3, stream)
        else:
            return interleaved_decode(4, stream)
    else:
        return signed_decode(4, stream)


def _compute_run_coder_fast(context: Context) -> int:
    """
    Compute run coder parameter for FAST encoder mode.

    Uses simple first-moment thresholds for quick decisions.

    Args:
        context: Current context statistics.

    Returns:
        Run coder pot value (0-4).
    """
    if context.sum < 1:
        return 4
    elif context.sum < 2:
        return 3
    elif context.sum < 4:
        return 2
    elif context.sum < 8:
        return 1
    else:
        return 0


def _compute_run_coder_contextual(context: Context, quality: int) -> int:
    """
    Compute run coder parameter for CONTEXTUAL encoder mode.

    Uses both first and second moments with quality-dependent thresholds.

    Args:
        context: Current context statistics.
        quality: Quality parameter (1024 = lossless).

    Returns:
        Run coder pot value (0-4).
    """
    sum_sq = context.sum * context.sum

    if quality == 1024:
        # Lossless mode: simple threshold
        return 1 if context.sum < 2 else 0
    else:
        # Lossy mode: complex thresholds
        if context.sum < 4 and context.sum2 < 2:
            return 4
        elif context.sum < 8 and context.sum2 < 4:
            return 3
        elif 2 * sum_sq < 3 * context.sum2 + 48:
            return 2
        elif 2 * sum_sq < 5 * context.sum2 + 32:
            return 1
        else:
            return 0


def decode_block(
    image: np.ndarray,
    stream: BitReader,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    scheme: Encoder,
    quality: int,
    is_chroma: bool,
) -> None:
    """
    Decode all wavelet levels for a block.

    This decodes the coefficient hierarchy starting from the coarsest
    level (DC) down to the finest detail coefficients.

    Args:
        image: 2D numpy array to store decoded coefficients.
        stream: BitReader positioned at block data.
        x0, y0: Top-left corner of the block.
        x1, y1: Bottom-right corner (exclusive) of the block.
        scheme: Encoder type.
        quality: Quality parameter.
        is_chroma: True if chroma channel.
    """
    sizex = x1 - x0
    sizey = y1 - y0

    # Find the maximum step (coarsest level)
    step = 1
    while step < sizex or step < sizey:
        step *= 2

    # Decode from coarsest to finest
    has_dc = True
    while step >= 1:
        decode_coefficients(
            image, stream, x0, y0, x1, y1, step, scheme, quality, has_dc, is_chroma
        )
        has_dc = False  # Only first level has DC
        step //= 2
