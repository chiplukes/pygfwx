"""
Annotated hex dump of GFWX files.

Provides a human-readable, annotated hex dump that labels each region of
a .gfwx file: magic, header fields, metadata, transform program, and
compressed block data.

Usage::

    from pygfwx.debug.hexdump import hexdump, print_hexdump

    # Print annotated dump to stdout
    print_hexdump(Path("image.gfwx"))

    # Get a list of annotated lines
    lines = hexdump(data)
"""

from __future__ import annotations

import struct
from pathlib import Path

# ── Region annotation ─────────────────────────────────────────────────────────


def _regions(data: bytes) -> list[tuple[int, int, str]]:
    """
    Identify annotated regions in a GFWX byte stream.

    Returns a list of (start_offset, end_offset, label) tuples in order.
    Regions cover the full file without gaps.
    """
    if len(data) < 4:
        return [(0, len(data), "data")]

    regions: list[tuple[int, int, str]] = []

    # ── Header (32 bytes fixed) ───────────────────────────────────────────────
    if len(data) >= 4:
        regions.append((0, 4, "magic: 'GFWX'"))
    if len(data) >= 8:
        regions.append((4, 8, "version"))
    if len(data) >= 12:
        regions.append((8, 12, "sizex (width)"))
    if len(data) >= 16:
        regions.append((12, 16, "sizey (height)"))
    if len(data) >= 20:
        regions.append((16, 18, "layers - 1"))
        regions.append((18, 20, "channels - 1"))
    if len(data) >= 32:
        regions.append((20, 21, "bit_depth - 1"))
        # Bits 21: 1-bit signed + 10-bit quality + 8-bit chroma_scale + 5-bit block_size = packed
        regions.append((21, 24, "signed|quality|chroma_scale|block_size (packed bits)"))
        regions.append((24, 25, "filter"))
        regions.append((25, 26, "quantization"))
        regions.append((26, 27, "encoder"))
        regions.append((27, 28, "intent"))
        regions.append((28, 32, "metadata_size (words)"))

    if len(data) < 32:
        return regions

    # ── Metadata (variable) ───────────────────────────────────────────────────
    metadata_words = struct.unpack_from("<I", data, 28)[0]
    metadata_bytes = metadata_words * 4
    meta_start = 32
    meta_end = meta_start + metadata_bytes
    if metadata_bytes > 0 and meta_end <= len(data):
        regions.append((meta_start, meta_end, f"metadata ({metadata_words} words)"))
    pos = meta_end if meta_end <= len(data) else 32

    # ── Transform program (word-aligned, variable) ────────────────────────────
    if pos < len(data):
        transform_end = _find_transform_end(data, pos)
        if transform_end > pos:
            regions.append((pos, transform_end, "transform program"))
        pos = transform_end

    # ── Compressed block data (remainder) ────────────────────────────────────
    if pos < len(data):
        regions.append((pos, len(data), "compressed block data"))

    return regions


def _find_transform_end(data: bytes, start: int) -> int:
    """
    Find end offset of the transform program (word-aligned).

    The transform program is a sequence of signed-elias-coded integers read
    until a -1 is encountered as a dest-channel.  We use a simplified heuristic:
    scan forward one word at a time and look for the characteristic end pattern
    (a word-aligned boundary after the first -1 Elias code, which encodes as
    the single bit pattern "1" → 1-bit, so the first byte cannot be 0x00).

    For simplicity we cap the search at 32 bytes (more than enough for any
    known program), and return the next word-aligned position after the scan.
    """
    # The transform program is at most 32 bytes (8 words) for any standard program.
    # We scan word by word looking for the word that has the trailing flush.
    # Simpler approach: just look at the first word.  If data[start] has bit 0 set,
    # it encodes a signed -1 end-marker (identity transform) in the first word.
    max_scan = min(start + 32, len(data))  # noqa: F841
    # Round up to next word boundary from start
    pos = start
    if pos % 4 != 0:
        pos = (pos // 4 + 1) * 4
    end = min(pos + 32, len(data))
    # Round end up to word boundary
    if end % 4 != 0:
        end = (end // 4 + 1) * 4
    end = min(end, len(data))

    # Determine if this looks like a real transform program by reading the first byte
    if pos < len(data):
        first_byte = data[pos]
        if first_byte & 0x01:
            # Bit 0 set: first signed Elias code is -1 (identity marker) → 1 word
            return min(pos + 4, len(data))
        else:
            # Longer program; use a conservative 8-word estimate
            return min(pos + 32, len(data))
    return pos


# ── Hex dump formatting ───────────────────────────────────────────────────────


def hexdump(data: bytes, *, width: int = 16) -> list[str]:
    """
    Return an annotated hex dump of GFWX data as a list of strings.

    Each region of the file is preceded by a label line.  Within each region
    bytes are printed in rows of `width` bytes showing both hex and ASCII.

    Args:
        data: GFWX compressed data bytes.
        width: Bytes per row (default 16).

    Returns:
        List of formatted strings (one per line of output).
    """
    lines: list[str] = []
    regions = _regions(data)

    for start, end, label in regions:
        # Region header
        lines.append(f"  ┌─ {label}  [0x{start:04X}..0x{end - 1:04X}] ({end - start} bytes)")

        # Hex rows
        chunk = data[start:end]
        for row_start in range(0, len(chunk), width):
            row = chunk[row_start : row_start + width]
            offset = start + row_start
            hex_part = " ".join(f"{b:02X}" for b in row)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
            padding = "   " * (width - len(row))
            lines.append(f"  │  {offset:06X}  {hex_part}{padding}  │{ascii_part}│")

        lines.append("  └" + "─" * 72)

    return lines


def print_hexdump(source: bytes | Path | str, *, width: int = 16) -> None:
    """
    Print an annotated hex dump of a GFWX file or bytes to stdout.

    Args:
        source: Either raw bytes, or a path (str or Path) to a .gfwx file.
        width: Bytes per row (default 16).

    Example::

        print_hexdump(Path("image.gfwx"))
        print_hexdump(compressed_bytes)
    """
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        data = bytes(source)

    total = len(data)
    print(f"GFWX hexdump — {total} bytes (0x{total:X})")
    print()
    for line in hexdump(data, width=width):
        print(line)
