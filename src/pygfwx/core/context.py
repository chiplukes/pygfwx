"""
GFWX Context Modeling.

This module implements the adaptive context modeling used in GFWX for
entropy coding. The context is computed from previously decoded coefficients
(ancestor, siblings, neighbors) to predict the distribution of the current
coefficient and select the optimal Golomb-Rice parameter.

The context consists of two values:
- sum: weighted sum of absolute values of neighbors (first moment)
- sum2: weighted sum of squared values of neighbors (second moment)

These statistics are used to select between different coding modes
(interleaved vs signed, different pot values) for optimal compression.
"""

from typing import NamedTuple

import numpy as np


class Context(NamedTuple):
    """Context statistics for entropy coding.

    Attributes:
        sum: First moment - weighted average of |neighbors| * 16
        sum2: Second moment - weighted average of neighbors^2 * 16
    """

    sum: int
    sum2: int


def _add_context(
    x: int, weight: int, sum_val: int, sum2_val: int, count: int
) -> tuple[int, int, int]:
    """
    Add a coefficient value to the running context statistics.

    Args:
        x: The coefficient value (will use absolute value).
        weight: The weight for this coefficient.
        sum_val: Running sum of weighted |values|.
        sum2_val: Running sum of weighted values^2.
        count: Running count of weights.

    Returns:
        Updated (sum, sum2, count) tuple.
    """
    abs_x = abs(x)
    sum_val += abs_x * weight
    # Clamp to 4096 to avoid overflow in sum2 calculation
    clamped = min(abs_x, 4096)
    sum2_val += (clamped * clamped) * weight
    count += weight
    return sum_val, sum2_val, count


def get_context(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    x: int,
    y: int,
    skip: int,
) -> Context:
    """
    Compute context statistics for a coefficient position.

    The context is built from:
    - Ancestor coefficient (weight 2)
    - Upper and left siblings (weight 2 each, if available)
    - Neighbors at distance 2*skip (weights 4, 4, 2, 2)
    - Neighbors at distance 4*skip (weights 2, 2, 1, 1)

    Args:
        image: 2D array of coefficient values.
        x0, y0: Top-left corner of the current block.
        x1, y1: Bottom-right corner (exclusive) of the current block.
        x: Local x coordinate within block (relative to x0).
        y: Local y coordinate within block (relative to y0).
        skip: Current wavelet level step size.

    Returns:
        Context containing weighted sum and sum-of-squares,
        normalized to 16 counts.

    Note:
        The image array should have the coefficients in row-major order.
        Positions outside the block boundaries are handled by wrapping
        the ancestor position.
    """
    sizex = x1 - x0

    # Calculate ancestor position
    # px = x0 + (x & ~(skip * 2)) + (x & skip)
    px = x0 + (x & ~(skip * 2)) + (x & skip)
    if px >= x1:
        px -= skip * 2

    py = y0 + (y & ~(skip * 2)) + (y & skip)
    if py >= y1:
        py -= skip * 2

    count = 0
    sum_val = 0
    sum2_val = 0

    # Ancestor (weight 2)
    ancestor_val = int(image[py, px])
    sum_val, sum2_val, count = _add_context(abs(ancestor_val), 2, sum_val, sum2_val, count)

    # Siblings
    if (y & skip) and (x | skip) < sizex:
        # Upper sibling (weight 2)
        upper_sibling = int(image[y0 + y - skip, x0 + (x | skip)])
        sum_val, sum2_val, count = _add_context(upper_sibling, 2, sum_val, sum2_val, count)

        if x & skip:
            # Left sibling (weight 2)
            left_sibling = int(image[y0 + y, x0 + x - skip])
            sum_val, sum2_val, count = _add_context(left_sibling, 2, sum_val, sum2_val, count)

    # Neighbors at distance 2*skip
    if y >= skip * 2 and x >= skip * 2:
        # North neighbor (weight 4)
        north = int(image[y0 + y - skip * 2, x0 + x])
        sum_val, sum2_val, count = _add_context(north, 4, sum_val, sum2_val, count)

        # West neighbor (weight 4)
        west = int(image[y0 + y, x0 + x - skip * 2])
        sum_val, sum2_val, count = _add_context(west, 4, sum_val, sum2_val, count)

        # Northwest neighbor (weight 2)
        northwest = int(image[y0 + y - skip * 2, x0 + x - skip * 2])
        sum_val, sum2_val, count = _add_context(northwest, 2, sum_val, sum2_val, count)

        # Northeast neighbor (weight 2, if available)
        if x + skip * 2 < sizex:
            northeast = int(image[y0 + y - skip * 2, x0 + x + skip * 2])
            sum_val, sum2_val, count = _add_context(northeast, 2, sum_val, sum2_val, count)

        # Neighbors at distance 4*skip
        if y >= skip * 4 and x >= skip * 4:
            # North-far (weight 2)
            north_far = int(image[y0 + y - skip * 4, x0 + x])
            sum_val, sum2_val, count = _add_context(north_far, 2, sum_val, sum2_val, count)

            # West-far (weight 2)
            west_far = int(image[y0 + y, x0 + x - skip * 4])
            sum_val, sum2_val, count = _add_context(west_far, 2, sum_val, sum2_val, count)

            # Northwest-far (weight 1)
            northwest_far = int(image[y0 + y - skip * 4, x0 + x - skip * 4])
            sum_val, sum2_val, count = _add_context(northwest_far, 1, sum_val, sum2_val, count)

            # Northeast-far (weight 1, if available)
            if x + skip * 4 < sizex:
                northeast_far = int(image[y0 + y - skip * 4, x0 + x + skip * 4])
                sum_val, sum2_val, count = _add_context(northeast_far, 1, sum_val, sum2_val, count)

    # Normalize to 16 counts (with rounding)
    norm_sum = (sum_val * 16 + count // 2) // count
    norm_sum2 = (sum2_val * 16 + count // 2) // count

    return Context(norm_sum, norm_sum2)


def select_coding_mode(
    context: Context, is_chroma: bool = False
) -> tuple[str, int]:
    """
    Select the optimal coding mode based on context statistics.

    The coding mode determines:
    - Whether to use interleaved or signed Golomb-Rice coding
    - The pot (power-of-two) parameter for the code

    Args:
        context: The context statistics (sum, sum2).
        is_chroma: True if this is a chroma channel (uses different thresholds).

    Returns:
        Tuple of (coding_type, pot) where:
        - coding_type is "interleaved" or "signed"
        - pot is the Golomb-Rice parameter (0-4)
    """
    sum_sq = context.sum * context.sum
    sum2 = context.sum2

    # Decision tree from SDK decode() function
    threshold_base = 250 if is_chroma else 100

    if sum_sq < 2 * sum2 + threshold_base:
        return ("interleaved", 0)
    elif sum_sq < 2 * sum2 + 950:
        return ("interleaved", 1)
    elif sum_sq < 3 * sum2 + 3000:
        if sum_sq < 5 * sum2 + 400:
            return ("signed", 1)
        else:
            return ("interleaved", 2)
    elif sum_sq < 3 * sum2 + 12000:
        if sum_sq < 5 * sum2 + 3000:
            return ("signed", 2)
        else:
            return ("interleaved", 3)
    elif sum_sq < 4 * sum2 + 44000:
        if sum_sq < 6 * sum2 + 12000:
            return ("signed", 3)
        else:
            return ("interleaved", 4)
    else:
        return ("signed", 4)


def update_fast_context(context: Context, value: int) -> Context:
    """
    Update context using the FAST encoder's decaying moment method.

    In FAST mode, context is updated after each coefficient using
    exponential decay: new = (old * 15 + 7) >> 4 + current

    Args:
        context: Current context statistics.
        value: The decoded coefficient value.

    Returns:
        Updated context.
    """
    t = abs(value)
    new_sum = ((context.sum * 15 + 7) >> 4) + t
    clamped_t = min(t, 4096)
    new_sum2 = ((context.sum2 * 15 + 7) >> 4) + clamped_t * clamped_t
    return Context(new_sum, new_sum2)


def compute_run_coder(
    context: Context, value: int, current_run_coder: int, quality: int, encoder_fast: bool
) -> int:
    """
    Compute the run-length coder parameter based on context.

    The run coder parameter determines how many bits to use for
    encoding runs of zeros (0 = no run coding, 1-4 = pot for run length).

    Args:
        context: Current context statistics.
        value: The current coefficient value (used to check if zero).
        current_run_coder: Current run coder state.
        quality: Quality setting (1024 = lossless).
        encoder_fast: True if using FAST encoder mode.

    Returns:
        New run coder parameter (0-4).
    """
    # Only update when value and current_run_coder have same zero-ness
    if bool(value) != bool(current_run_coder):
        return current_run_coder

    if encoder_fast:
        # FAST encoder uses simple first-moment threshold
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
    else:
        # CONTEXTUAL encoder uses both moments
        sum_sq = context.sum * context.sum
        if quality == 1024:
            # Lossless mode - simpler threshold
            return 1 if context.sum < 2 else 0
        else:
            # Lossy mode - more complex thresholds
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
