"""
GFWX Wavelet Lifting Transforms.

This module implements the lifting scheme wavelet transforms used in GFWX:
- Forward transform (lift): For encoding, decomposes image into wavelet coefficients
- Inverse transform (unlift): For decoding, reconstructs image from coefficients

Two filter types are supported:
- FilterLinear (5/3): Integer-exact, suitable for lossless compression
- FilterCubic (9/7): Better frequency separation, suitable for lossy compression

The lifting scheme works in-place without requiring auxiliary memory buffers.
"""

import numpy as np

from pygfwx.core.header import Filter


def _trunc_div(num: int, denom: int) -> int:  # cm:f6a7b8c — _trunc_div(): C++ truncation-toward-zero integer division
    """
    Integer division with truncation toward zero (like C++).

    Python's // operator floors the result, which gives different results
    for negative numbers. C++ truncates toward zero.

    Examples:
        -5 // 2 = -3 in Python (floor)
        -5 / 2 = -2 in C++ (truncate toward zero)

    Args:
        num: Numerator.
        denom: Denominator (assumed positive).

    Returns:
        num / denom truncated toward zero.
    """
    return int(num / denom)


def _round_fraction(num: int, denom: int) -> int:
    """
    Round a fraction to nearest integer (away from zero for .5).

    This matches C++ integer division behavior which truncates toward zero,
    not Python's floor division which truncates toward negative infinity.

    Args:
        num: Numerator.
        denom: Denominator.

    Returns:
        Rounded result.
    """
    if num < 0:
        # C++ uses truncation toward zero: (num - denom/2) / denom
        # We compute: adjusted = num - denom // 2, then truncate toward zero
        adjusted = num - denom // 2
        # int() truncates toward zero like C++
        return int(adjusted / denom)
    else:
        return (num + denom // 2) // denom


def _median(a: int, b: int, c: int) -> int:
    """
    Return the median of three values.

    Args:
        a, b, c: Three integer values.

    Returns:
        The median value.
    """
    if a <= b:
        if b <= c:
            return b
        elif a <= c:
            return c
        else:
            return a
    else:
        if a <= c:
            return a
        elif b <= c:
            return c
        else:
            return b


def _cubic(c0: int, c1: int, c2: int, c3: int) -> int:  # cm:d9e0f1 — _cubic(): 9/7 wavelet interpolation with median clamp
    """
    Cubic interpolation with median clamping.

    Computes: median(round((-c0 + 9*(c1+c2) - c3) / 16), c1, c2)

    The median clamping prevents overshoots at edges which would
    cause ringing artifacts.

    Args:
        c0, c1, c2, c3: Four neighboring sample values.

    Returns:
        Interpolated value clamped to [min(c1,c2), max(c1,c2)].
    """
    interpolated = _round_fraction(-c0 + 9 * (c1 + c2) - c3, 16)
    return _median(interpolated, c1, c2)


def lift(  # cm:a2b3c4 — lift(): forward wavelet transform (spatial → wavelet domain, in-place)
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: int,
    filter_type: Filter,
) -> None:
    """
    Forward wavelet transform using the lifting scheme.

    Transforms image data in-place from spatial domain to wavelet domain.
    Applies horizontal lifting followed by vertical lifting, then doubles
    the step and repeats until the entire image is processed.

    Args:
        image: 2D numpy array (modified in-place).
        x0, y0: Top-left corner of region to transform.
        x1, y1: Bottom-right corner (exclusive).
        step: Initial step size (usually 1).
        filter_type: Filter.LINEAR (5/3) or Filter.CUBIC (9/7).

    Note:
        After transform, even indices contain approximation coefficients (L)
        and odd indices contain detail coefficients (H), interleaved.
    """
    sizex = x1 - x0
    sizey = y1 - y0

    while step < sizex or step < sizey:
        # Horizontal lifting
        if step < sizex:
            _lift_horizontal(image, x0, y0, sizex, sizey, step, filter_type)

        # Vertical lifting
        if step < sizey:
            _lift_vertical(image, x0, y0, sizex, sizey, step, filter_type)

        step *= 2


def _lift_horizontal(
    image: np.ndarray,
    x0: int,
    y0: int,
    sizex: int,
    sizey: int,
    step: int,
    filter_type: Filter,
) -> None:
    """Apply horizontal lifting (predict and update) to all rows."""
    for y in range(0, sizey, step):
        row = image[y0 + y]

        if filter_type == Filter.CUBIC:
            _lift_horizontal_cubic(row, x0, sizex, step)
        else:
            _lift_horizontal_linear(row, x0, sizex, step)


def _lift_horizontal_linear(row: np.ndarray, x0: int, sizex: int, step: int) -> None:
    """Horizontal lifting with 5/3 linear filter."""
    # Predict step: odd -= (left + right) / 2
    # Use truncation division to match C++ behavior
    x = step
    while x < sizex - step:
        row[x0 + x] -= _trunc_div(int(row[x0 + x - step]) + int(row[x0 + x + step]), 2)
        x += step * 2
    # Boundary: last odd position
    if x < sizex:
        row[x0 + x] -= int(row[x0 + x - step])

    # Update step: even += (left + right) / 4
    # Use truncation division to match C++ behavior
    x = step * 2
    while x < sizex - step:
        row[x0 + x] += _trunc_div(int(row[x0 + x - step]) + int(row[x0 + x + step]), 4)
        x += step * 2
    # Boundary: last even position
    if x < sizex:
        row[x0 + x] += _trunc_div(int(row[x0 + x - step]), 2)


def _lift_horizontal_cubic(row: np.ndarray, x0: int, sizex: int, step: int) -> None:
    """Horizontal lifting with 9/7 cubic filter."""
    # Predict step with cubic interpolation
    # For odd position at x, we use 4 even neighbors:
    # c0 = x - step*3, c1 = x - step, c2 = x + step, c3 = x + step*3
    # Initialize sliding window
    c0 = int(row[x0])  # Actually c0 should be before c1, starts as c1
    c1 = int(row[x0])
    c2 = int(row[x0 + step * 2]) if step * 2 < sizex else int(row[x0])

    x = step
    while x < sizex - step * 3:
        c3 = int(row[x0 + x + step * 3])  # base3[x] = base[x + step*3]
        row[x0 + x] -= _cubic(c0, c1, c2, c3)
        c0, c1, c2 = c1, c2, c3
        x += step * 2

    # Finish remaining positions at boundary (use c2 for c3)
    while x < sizex:
        row[x0 + x] -= _cubic(c0, c1, c2, c2)
        c0, c1 = c1, c2
        x += step * 2

    # Update step with cubic interpolation
    # For even position at x, we use 4 odd neighbors:
    # g0 = x - step*3, g1 = x - step, g2 = x + step, g3 = x + step*3
    g0 = int(row[x0 + step])
    g1 = int(row[x0 + step])
    g2 = int(row[x0 + step * 3]) if step * 3 < sizex else int(row[x0 + step])

    x = step * 2
    while x < sizex - step * 3:
        g3 = int(row[x0 + x + step * 3])  # base3[x] = base[x + step*3]
        row[x0 + x] += _trunc_div(_cubic(g0, g1, g2, g3), 2)
        g0, g1, g2 = g1, g2, g3
        x += step * 2

    # Finish remaining positions at boundary
    while x < sizex:
        row[x0 + x] += _trunc_div(_cubic(g0, g1, g2, g2), 2)
        g0, g1 = g1, g2
        x += step * 2


def _lift_vertical(
    image: np.ndarray,
    x0: int,
    y0: int,
    sizex: int,
    sizey: int,
    step: int,
    filter_type: Filter,
) -> None:
    """Apply vertical lifting (predict and update) to all columns."""
    # Predict step: odd rows -= f(even rows)
    for y in range(step, sizey, step * 2):
        base = image[y0 + y]
        c1base = image[y0 + y - step]
        c2base = image[y0 + y + step] if y + step < sizey else c1base

        if filter_type == Filter.CUBIC:
            c0base = image[y0 + y - step * 3] if y >= step * 3 else c1base
            c3base = image[y0 + y + step * 3] if y + step * 3 < sizey else c2base
            for x in range(0, sizex, step):
                base[x0 + x] -= _cubic(
                    int(c0base[x0 + x]),
                    int(c1base[x0 + x]),
                    int(c2base[x0 + x]),
                    int(c3base[x0 + x]),
                )
        else:
            for x in range(0, sizex, step):
                # Use truncation division to match C++ behavior
                base[x0 + x] -= _trunc_div(int(c1base[x0 + x]) + int(c2base[x0 + x]), 2)

    # Update step: even rows += f(odd rows)
    for y in range(step * 2, sizey, step * 2):
        base = image[y0 + y]
        g1base = image[y0 + y - step]
        g2base = image[y0 + y + step] if y + step < sizey else g1base

        if filter_type == Filter.CUBIC:
            g0base = image[y0 + y - step * 3] if y >= step * 3 else g1base
            g3base = image[y0 + y + step * 3] if y + step * 3 < sizey else g2base
            for x in range(0, sizex, step):
                # Use truncation division to match C++ behavior
                base[x0 + x] += _trunc_div(
                    _cubic(
                        int(g0base[x0 + x]),
                        int(g1base[x0 + x]),
                        int(g2base[x0 + x]),
                        int(g3base[x0 + x]),
                    ),
                    2,
                )
        else:
            for x in range(0, sizex, step):
                # Use truncation division to match C++ behavior
                base[x0 + x] += _trunc_div(int(g1base[x0 + x]) + int(g2base[x0 + x]), 4)


def unlift(  # cm:d5e6f7 — unlift(): inverse wavelet transform (wavelet → spatial domain, in-place)
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_step: int,
    filter_type: Filter,
) -> None:
    """
    Inverse wavelet transform using the lifting scheme.

    Transforms wavelet coefficients back to spatial domain in-place.
    This is the inverse of lift() - it undoes the update then the predict
    steps, working from coarsest to finest resolution.

    Args:
        image: 2D numpy array with wavelet coefficients (modified in-place).
        x0, y0: Top-left corner of region to transform.
        x1, y1: Bottom-right corner (exclusive).
        min_step: Minimum step size to process down to (usually 1).
        filter_type: Filter.LINEAR (5/3) or Filter.CUBIC (9/7).

    Note:
        The operations are the exact inverse of lift():
        - Vertical undo-update, then undo-predict
        - Horizontal undo-update, then undo-predict
        - Halve the step and repeat
    """
    sizex = x1 - x0
    sizey = y1 - y0

    # Find the maximum step (coarsest level)
    step = min_step
    while step * 2 < sizex or step * 2 < sizey:
        step *= 2

    # Work from coarsest to finest
    while step >= min_step:
        # Vertical unlifting (reverse order: undo-update then undo-predict)
        if step < sizey:
            _unlift_vertical(image, x0, y0, sizex, sizey, step, filter_type)

        # Horizontal unlifting
        if step < sizex:
            _unlift_horizontal(image, x0, y0, sizex, sizey, step, filter_type)

        step //= 2


def _unlift_horizontal(
    image: np.ndarray,
    x0: int,
    y0: int,
    sizex: int,
    sizey: int,
    step: int,
    filter_type: Filter,
) -> None:
    """Apply horizontal unlifting (undo-update then undo-predict) to all rows."""
    for y in range(0, sizey, step):
        row = image[y0 + y]

        if filter_type == Filter.CUBIC:
            _unlift_horizontal_cubic(row, x0, sizex, step)
        else:
            _unlift_horizontal_linear(row, x0, sizex, step)


def _unlift_horizontal_linear(row: np.ndarray, x0: int, sizex: int, step: int) -> None:
    """Horizontal unlifting with 5/3 linear filter."""
    # Undo update step: even -= (left + right) / 4
    # Use truncation division to match C++ behavior
    x = step * 2
    while x < sizex - step:
        row[x0 + x] -= _trunc_div(int(row[x0 + x - step]) + int(row[x0 + x + step]), 4)
        x += step * 2
    # Boundary
    if x < sizex:
        row[x0 + x] -= _trunc_div(int(row[x0 + x - step]), 2)

    # Undo predict step: odd += (left + right) / 2
    # Use truncation division to match C++ behavior
    x = step
    while x < sizex - step:
        row[x0 + x] += _trunc_div(int(row[x0 + x - step]) + int(row[x0 + x + step]), 2)
        x += step * 2
    # Boundary
    if x < sizex:
        row[x0 + x] += int(row[x0 + x - step])


def _unlift_horizontal_cubic(row: np.ndarray, x0: int, sizex: int, step: int) -> None:
    """Horizontal unlifting with 9/7 cubic filter."""
    # Undo update step - uses odd neighbors to adjust even positions
    # Use truncation division to match C++ behavior
    g0 = int(row[x0 + step])
    g1 = int(row[x0 + step])
    g2 = int(row[x0 + step * 3]) if step * 3 < sizex else int(row[x0 + step])

    x = step * 2
    while x < sizex - step * 3:
        g3 = int(row[x0 + x + step * 3])  # base3[x] = base[x + step*3]
        row[x0 + x] -= _trunc_div(_cubic(g0, g1, g2, g3), 2)
        g0, g1, g2 = g1, g2, g3
        x += step * 2

    while x < sizex:
        row[x0 + x] -= _trunc_div(_cubic(g0, g1, g2, g2), 2)
        g0, g1 = g1, g2
        x += step * 2

    # Undo predict step - uses even neighbors to adjust odd positions
    c0 = int(row[x0])
    c1 = int(row[x0])
    c2 = int(row[x0 + step * 2]) if step * 2 < sizex else int(row[x0])

    x = step
    while x < sizex - step * 3:
        c3 = int(row[x0 + x + step * 3])  # base3[x] = base[x + step*3]
        row[x0 + x] += _cubic(c0, c1, c2, c3)
        c0, c1, c2 = c1, c2, c3
        x += step * 2

    while x < sizex:
        row[x0 + x] += _cubic(c0, c1, c2, c2)
        c0, c1 = c1, c2
        x += step * 2


def _unlift_vertical(
    image: np.ndarray,
    x0: int,
    y0: int,
    sizex: int,
    sizey: int,
    step: int,
    filter_type: Filter,
) -> None:
    """Apply vertical unlifting (undo-update then undo-predict) to all columns."""
    # Undo update step: even rows -= f(odd rows)
    # Use truncation division to match C++ behavior
    for y in range(step * 2, sizey, step * 2):
        base = image[y0 + y]
        g1base = image[y0 + y - step]
        g2base = image[y0 + y + step] if y + step < sizey else g1base

        if filter_type == Filter.CUBIC:
            g0base = image[y0 + y - step * 3] if y >= step * 3 else g1base
            g3base = image[y0 + y + step * 3] if y + step * 3 < sizey else g2base
            for x in range(0, sizex, step):
                base[x0 + x] -= _trunc_div(
                    _cubic(
                        int(g0base[x0 + x]),
                        int(g1base[x0 + x]),
                        int(g2base[x0 + x]),
                        int(g3base[x0 + x]),
                    ),
                    2,
                )
        else:
            for x in range(0, sizex, step):
                base[x0 + x] -= _trunc_div(int(g1base[x0 + x]) + int(g2base[x0 + x]), 4)

    # Undo predict step: odd rows += f(even rows)
    # Use truncation division to match C++ behavior
    for y in range(step, sizey, step * 2):
        base = image[y0 + y]
        c1base = image[y0 + y - step]
        c2base = image[y0 + y + step] if y + step < sizey else c1base

        if filter_type == Filter.CUBIC:
            c0base = image[y0 + y - step * 3] if y >= step * 3 else c1base
            c3base = image[y0 + y + step * 3] if y + step * 3 < sizey else c2base
            for x in range(0, sizex, step):
                base[x0 + x] += _cubic(
                    int(c0base[x0 + x]),
                    int(c1base[x0 + x]),
                    int(c2base[x0 + x]),
                    int(c3base[x0 + x]),
                )
        else:
            for x in range(0, sizex, step):
                base[x0 + x] += _trunc_div(int(c1base[x0 + x]) + int(c2base[x0 + x]), 2)


def lift_full(image: np.ndarray, filter_type: Filter) -> None:
    """
    Apply forward wavelet transform to the entire image.

    Convenience function that calls lift() with full image bounds.

    Args:
        image: 2D numpy array (modified in-place).
        filter_type: Filter.LINEAR or Filter.CUBIC.
    """
    lift(image, 0, 0, image.shape[1], image.shape[0], 1, filter_type)


def unlift_full(image: np.ndarray, filter_type: Filter) -> None:
    """
    Apply inverse wavelet transform to the entire image.

    Convenience function that calls unlift() with full image bounds.

    Args:
        image: 2D numpy array (modified in-place).
        filter_type: Filter.LINEAR or Filter.CUBIC.
    """
    unlift(image, 0, 0, image.shape[1], image.shape[0], 1, filter_type)
