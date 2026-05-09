"""
Debugging and analysis utilities.

This module provides tools for:
- Hex dump visualization of GFWX files (hexdump.py)
- Wavelet coefficient and progressive decode visualization (visualize.py)
- Bitstream analysis

Usage::

    from pygfwx.debug.hexdump import print_hexdump
    from pygfwx.debug.visualize import plot_wavelet_decomposition
"""

from pygfwx.debug.hexdump import hexdump, print_hexdump

__all__ = ["hexdump", "print_hexdump"]

