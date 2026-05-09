"""
GFWX Coefficient Encoder.

This module implements the main coefficient encoding loop that combines:
- Context modeling for adaptive coding
- Golomb-Rice encoding (interleaved and signed variants)
- Run-length encoding for zero coefficients

The encoder processes wavelet coefficients in the same order as the decoder
(arranged so that (x | y) & step == 1) and uses context from previously
encoded neighbors to select the optimal encoding mode.

Reference: gfwx.h encode() lines 476-558
"""

import numpy as np

from pygfwx.core.bitstream import BitWriter
from pygfwx.core.context import (
    Context,
    compute_run_coder_contextual,
    compute_run_coder_fast,
    get_context,
    update_fast_context,
)
from pygfwx.core.golomb_rice import interleaved_encode, signed_encode, unsigned_encode
from pygfwx.core.header import Encoder


def encode_coefficients(
    image: np.ndarray,
    stream: BitWriter,
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
    Encode wavelet coefficients from the image array into the bitstream.

    This function encodes coefficients for a block of the image, using
    context-adaptive Golomb-Rice coding with run-length encoding for zeros.

    Args:
        image: 2D numpy array containing wavelet coefficients.
        stream: BitWriter to write encoded data.
        x0, y0: Top-left corner of the block.
        x1, y1: Bottom-right corner (exclusive) of the block.
        step: Wavelet level step size (1 for full resolution).
        scheme: Encoder type (TURBO, FAST, CONTEXTUAL, HIGH_BITRATE).
        quality: Quality parameter (1024 = lossless).
        has_dc: True if this block contains the DC coefficient.
        is_chroma: True if this is a chroma channel (affects thresholds).

    Note:
        The encoding order is arranged so that (x | y) & step == 1,
        matching the decoder's traversal pattern exactly.
    """
    sizex = x1 - x0
    sizey = y1 - y0

    # Encode DC coefficient if present
    if has_dc and sizex > 0 and sizey > 0:
        signed_encode(4, int(image[y0, x0]), stream)

    # Initialize context (first moment, second moment)
    context = Context(sum=0, sum2=0)

    # Run-length state
    # run counts consecutive zeros (encoder starts at 0, counts up)
    # runCoder = 0 means run coding is disabled
    run = 0
    run_coder = 0

    # Process coefficients in the specific order
    for y in range(0, sizey, step):
        # xStep alternates based on y position
        # This ensures (x | y) & step == 1 for all processed positions
        x_step = step if (y & step) else step * 2

        for x in range(x_step - step, sizex, x_step):
            s = int(image[y0 + y, x0 + x])

            if run_coder and s == 0:
                # Continue the run
                run += 1
            else:
                # Handle HIGH_BITRATE mode (no run coding, simpler context)
                if scheme == Encoder.HIGH_BITRATE:
                    bit_width = context.sum.bit_length() if context.sum > 0 else 0
                    pot = max(0, bit_width - 4)
                    interleaved_encode(pot, s, stream)
                    # Update context with decaying moment
                    t = abs(s)
                    new_sum = ((context.sum * 15 + 7) >> 4) + t
                    context = Context(sum=new_sum, sum2=context.sum2)
                    continue

                # Break the run if we had one active
                if run_coder:
                    unsigned_encode(run_coder, run, stream)
                    run = 0
                    # s can't be zero here, so shift negatives by 1
                    if s < 0:
                        s += 1

                # Get context for CONTEXTUAL mode
                if scheme == Encoder.CONTEXTUAL:
                    context = get_context(image, x0, y0, x1, y1, x, y, step)

                # Encode coefficient using context-adaptive mode selection
                _encode_with_context(context, s, stream, is_chroma)

                # Update context and run coder
                if scheme == Encoder.FAST:
                    context = update_fast_context(context, s)
                    # Update run coder if s and runCoder have same zero-ness
                    if bool(s) == bool(run_coder):
                        run_coder = compute_run_coder_fast(context)
                elif scheme != Encoder.TURBO:
                    # CONTEXTUAL mode run coder update
                    if bool(s) == bool(run_coder):
                        run_coder = compute_run_coder_contextual(context, quality)

    # Flush remaining run at end of block
    if run > 0:
        unsigned_encode(run_coder, run, stream)


def _encode_with_context(
    context: Context, s: int, stream: BitWriter, is_chroma: bool
) -> None:
    """
    Encode a coefficient using context-adaptive mode selection.

    The decision tree selects between interleaved and signed Golomb-Rice
    coding based on the relationship between sum² and sum2 (second moment).
    This MUST match the decoder's decision tree exactly.

    Args:
        context: Current context statistics.
        s: The coefficient value to encode.
        stream: BitWriter to encode to.
        is_chroma: True if chroma channel (uses higher threshold).
    """
    sum_sq = context.sum * context.sum
    sum2 = context.sum2
    threshold = 250 if is_chroma else 100

    if sum_sq < 2 * sum2 + threshold:
        interleaved_encode(0, s, stream)
    elif sum_sq < 2 * sum2 + 950:
        interleaved_encode(1, s, stream)
    elif sum_sq < 3 * sum2 + 3000:
        if sum_sq < 5 * sum2 + 400:
            signed_encode(1, s, stream)
        else:
            interleaved_encode(2, s, stream)
    elif sum_sq < 3 * sum2 + 12000:
        if sum_sq < 5 * sum2 + 3000:
            signed_encode(2, s, stream)
        else:
            interleaved_encode(3, s, stream)
    elif sum_sq < 4 * sum2 + 44000:
        if sum_sq < 6 * sum2 + 12000:
            signed_encode(3, s, stream)
        else:
            interleaved_encode(4, s, stream)
    else:
        signed_encode(4, s, stream)


def encode_block(
    image: np.ndarray,
    stream: BitWriter,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    scheme: Encoder,
    quality: int,
    is_chroma: bool,
) -> None:
    """
    Encode all wavelet levels for a block.

    This encodes the coefficient hierarchy starting from the coarsest
    level (DC) down to the finest detail coefficients.

    Args:
        image: 2D numpy array containing wavelet coefficients.
        stream: BitWriter to write encoded data.
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

    # Encode from coarsest to finest
    has_dc = True
    while step >= 1:
        encode_coefficients(
            image, stream, x0, y0, x1, y1, step, scheme, quality, has_dc, is_chroma
        )
        has_dc = False  # Only first level has DC
        step //= 2
