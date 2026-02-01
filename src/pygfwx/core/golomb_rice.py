"""
GFWX Golomb-Rice Coding.

This module implements the Golomb-Rice entropy coding used in GFWX.
Golomb-Rice codes are variable-length codes efficient for geometric
distributions, which is typical for wavelet coefficients.

The coding uses a parameter 'pot' (power-of-two) to adapt to the data:
- Quotient = x >> pot (encoded in unary as zeros + terminating 1)
- Remainder = x & ((1 << pot) - 1) (encoded in binary)

For large values (quotient >= 12), an escape code is used and the
remainder is recursively coded with a larger pot.
"""

from pygfwx.core.bitstream import BitReader, BitWriter

# =============================================================================
# DECODE FUNCTIONS
# =============================================================================


def unsigned_decode(pot: int, stream: BitReader) -> int:
    """
    Decode an unsigned value using Golomb-Rice coding.

    The value is encoded as:
    - x // (2^pot) in unary (zeros followed by a 1, max 12 zeros)
    - x % (2^pot) in binary (pot bits)

    If quotient >= 12, an escape code triggers recursive decoding
    with pot increased by 4.

    Args:
        pot: The power-of-two parameter (log2 of divisor).
        stream: The bit stream to read from.

    Returns:
        The decoded unsigned integer value.

    Example:
        >>> # Decode a value with pot=4
        >>> value = unsigned_decode(4, stream)
    """
    # Read unary zeros (quotient), max 12
    x = stream.get_zeros(12)

    # Limit actual pot to prevent overflow
    p = min(pot, 24)

    if pot < 108 and x == 12:
        # Escape code: quotient >= 12, recursively decode larger value
        # The 108 limit prevents infinite recursion on malformed data
        return (12 << p) + unsigned_decode(min(pot + 4, 108), stream)

    if p > 0:
        # Read remainder bits
        return (x << p) + stream.get_bits(p)
    else:
        # pot == 0 means no remainder bits
        return x


def interleaved_decode(pot: int, stream: BitReader) -> int:
    """
    Decode a signed value using interleaved coding.

    Signed values are mapped to unsigned using interleaving:
    - 0 -> 0
    - +1 -> 1, -1 -> 2
    - +2 -> 3, -2 -> 4
    - etc.

    This is efficient for distributions centered at zero.

    Args:
        pot: The power-of-two parameter.
        stream: The bit stream to read from.

    Returns:
        The decoded signed integer value.

    Example:
        >>> # Decode a signed value with pot=3
        >>> value = interleaved_decode(3, stream)
    """
    x = unsigned_decode(pot, stream)
    # Undo interleaving: odd -> positive, even -> negative
    if x & 1:
        return (x >> 1) + 1  # Odd: 1->1, 3->2, 5->3, ...
    else:
        return -(x >> 1)  # Even: 0->0, 2->-1, 4->-2, ...


def signed_decode(pot: int, stream: BitReader) -> int:
    """
    Decode a signed value using sign bit coding.

    The magnitude is coded first, then a sign bit follows
    if the magnitude is non-zero.

    Args:
        pot: The power-of-two parameter.
        stream: The bit stream to read from.

    Returns:
        The decoded signed integer value.

    Example:
        >>> # Decode a signed value with pot=4
        >>> value = signed_decode(4, stream)
    """
    x = unsigned_decode(pot, stream)
    if x == 0:
        return 0
    # Read sign bit: 1 = positive, 0 = negative
    return x if stream.get_bits(1) else -x


# Convenience functions for common pot values


def unsigned_decode_4(stream: BitReader) -> int:
    """Decode unsigned value with pot=4."""
    return unsigned_decode(4, stream)


def interleaved_decode_4(stream: BitReader) -> int:
    """Decode interleaved signed value with pot=4."""
    return interleaved_decode(4, stream)


def signed_decode_4(stream: BitReader) -> int:
    """Decode signed value with pot=4."""
    return signed_decode(4, stream)


# =============================================================================
# ENCODE FUNCTIONS
# =============================================================================


def unsigned_encode(pot: int, value: int, stream: BitWriter) -> None:
    """
    Encode an unsigned value using Golomb-Rice coding.

    Args:
        pot: The power-of-two parameter (log2 of divisor).
        value: The unsigned integer value to encode.
        stream: The bit stream to write to.
    """
    p = min(pot, 24)
    quotient = value >> p

    if pot < 108 and quotient >= 12:
        # Escape code: write 12 zeros (no terminating 1), then recurse
        stream.put_bits(0, 12)
        unsigned_encode(min(pot + 4, 108), value - (12 << p), stream)
    else:
        # Write quotient in unary (zeros + terminating 1)
        stream.put_bits(1, quotient + 1)
        # Write remainder
        if p > 0:
            stream.put_bits(value & ((1 << p) - 1), p)


def interleaved_encode(pot: int, value: int, stream: BitWriter) -> None:
    """
    Encode a signed value using interleaved coding.

    Maps signed values to unsigned:
    - 0 -> 0
    - +1 -> 1, -1 -> 2
    - +2 -> 3, -2 -> 4

    Args:
        pot: The power-of-two parameter.
        value: The signed integer value to encode.
        stream: The bit stream to write to.
    """
    if value > 0:
        unsigned_encode(pot, (value << 1) - 1, stream)
    else:
        unsigned_encode(pot, (-value) << 1, stream)


def signed_encode(pot: int, value: int, stream: BitWriter) -> None:
    """
    Encode a signed value using sign bit coding.

    Encodes magnitude first, then sign bit if non-zero.

    Args:
        pot: The power-of-two parameter.
        value: The signed integer value to encode.
        stream: The bit stream to write to.
    """
    if value == 0:
        unsigned_encode(pot, 0, stream)
    elif value > 0:
        unsigned_encode(pot, value, stream)
        stream.put_bits(1, 1)  # Positive sign
    else:
        unsigned_encode(pot, -value, stream)
        stream.put_bits(0, 1)  # Negative sign
