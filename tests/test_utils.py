"""
Shared test utilities for PyGFWX.

This module provides common helper functions used across test modules.
"""

import numpy as np


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if two arrays are exactly equal."""
    return np.array_equal(a, b)


def arrays_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """Check if two arrays are approximately equal."""
    return np.allclose(a, b, atol=atol)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate maximum absolute difference between two arrays."""
    return float(np.abs(a.astype(float) - b.astype(float)).max())


def diff_count(a: np.ndarray, b: np.ndarray) -> int:
    """Count number of differing elements."""
    return int(np.sum(a != b))


def diff_summary(a: np.ndarray, b: np.ndarray, name: str = "arrays") -> str:
    """Generate a summary of differences between two arrays."""
    if arrays_equal(a, b):
        return f"{name}: identical"

    diff = a.astype(float) - b.astype(float)
    return (
        f"{name}: "
        f"max_diff={np.abs(diff).max():.4f}, "
        f"mean_diff={np.abs(diff).mean():.4f}, "
        f"diff_count={diff_count(a, b)}/{a.size}"
    )


def hex_dump(data: bytes, bytes_per_line: int = 16, max_lines: int = 20) -> str:
    """Generate a hex dump of binary data."""
    lines = []
    for i in range(0, min(len(data), bytes_per_line * max_lines), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{i:08X}  {hex_part:<{bytes_per_line * 3}}  {ascii_part}")

    if len(data) > bytes_per_line * max_lines:
        lines.append(f"... ({len(data) - bytes_per_line * max_lines} more bytes)")

    return "\n".join(lines)


def bits_to_str(value: int, num_bits: int) -> str:
    """Format integer as binary string."""
    return format(value, f"0{num_bits}b")


def find_first_diff(a: bytes, b: bytes) -> int | None:
    """Find the first differing byte between two byte sequences.

    Returns:
        Index of first differing byte, or None if identical.
    """
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min_len
    return None


def compare_bitstreams(py_bits: bytes, sdk_bits: bytes, context: int = 4) -> str:
    """Compare two bitstreams and report differences."""
    diff_pos = find_first_diff(py_bits, sdk_bits)

    if diff_pos is None:
        return "Bitstreams match exactly!"

    lines = [
        f"First difference at byte {diff_pos}:",
        f"  Python length: {len(py_bits)} bytes",
        f"  SDK length: {len(sdk_bits)} bytes",
    ]

    if diff_pos < len(py_bits) and diff_pos < len(sdk_bits):
        lines.append(f"  Python byte: 0x{py_bits[diff_pos]:02X} ({bits_to_str(py_bits[diff_pos], 8)})")
        lines.append(f"  SDK byte:    0x{sdk_bits[diff_pos]:02X} ({bits_to_str(sdk_bits[diff_pos], 8)})")

        # Context
        start = max(0, diff_pos - context)
        end = min(min(len(py_bits), len(sdk_bits)), diff_pos + context + 1)
        lines.append(f"\nContext (bytes {start}-{end - 1}):")
        lines.append(f"  Python: {py_bits[start:end].hex(' ')}")
        lines.append(f"  SDK:    {sdk_bits[start:end].hex(' ')}")

    return "\n".join(lines)
