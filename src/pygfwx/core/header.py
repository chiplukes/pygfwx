"""
GFWX Header Parsing.

This module handles parsing and writing of GFWX file headers.
The header contains all metadata needed to decode the image.
"""

from dataclasses import dataclass
from enum import IntEnum

from pygfwx.core.bitstream import BitReader, BitWriter


class Filter(IntEnum):
    """Wavelet filter type."""

    LINEAR = 0  # 5/3 wavelet (better for lossless)
    CUBIC = 1  # 9/7 wavelet (better for lossy)


class Encoder(IntEnum):
    """Encoder mode."""

    TURBO = 0  # Fastest, lowest compression (deprecated in v1)
    FAST = 1  # Fast, medium compression
    CONTEXTUAL = 2  # Slowest, best compression
    HIGH_BITRATE = 3  # Best quality at high bitrates


class Intent(IntEnum):
    """Color intent/format."""

    GENERIC = 0
    MONO = 1
    BAYER_RGGB = 2
    BAYER_BGGR = 3
    BAYER_GRBG = 4
    BAYER_GBRG = 5
    BAYER_GENERIC = 6
    RGB = 7
    RGBA = 8
    RGBA_PREMULT = 9
    BGR = 10
    BGRA = 11
    BGRA_PREMULT = 12
    CMYK = 13


# Magic number for GFWX files
# The SDK writes 'G' | ('F' << 8) | ('W' << 16) | ('X' << 24) as a 32-bit value
# Stored in file as little-endian: bytes 'G', 'F', 'W', 'X'
# When read as little-endian uint32: 0x58574647
GFWX_MAGIC = ord("G") | (ord("F") << 8) | (ord("W") << 16) | (ord("X") << 24)  # 0x58574647

# Maximum quality (lossless)
QUALITY_MAX = 1024


@dataclass
class GFWXHeader:
    """GFWX file header."""

    version: int
    sizex: int
    sizey: int
    layers: int
    channels: int
    bit_depth: int
    is_signed: bool
    quality: int
    chroma_scale: int
    block_size: int
    filter: Filter
    quantization: int
    encoder: Encoder
    intent: Intent
    metadata_size: int  # Size in 32-bit words

    @property
    def is_lossless(self) -> bool:
        """Return True if quality is maximum (lossless)."""
        return self.quality == QUALITY_MAX


class HeaderParseError(Exception):
    """Raised when header parsing fails."""

    pass


def parse_header(data: bytes) -> tuple[GFWXHeader, int]:
    """
    Parse a GFWX header from compressed data.

    Args:
        data: The compressed data bytes.

    Returns:
        A tuple of (header, header_size_bytes).
        header_size_bytes is the total size including metadata.

    Raises:
        HeaderParseError: If the header is invalid or malformed.
    """
    if len(data) < 32:
        raise HeaderParseError("Data too short for GFWX header")

    reader = BitReader(data)

    # Magic number
    magic = reader.get_bits(32)
    if magic != GFWX_MAGIC:
        raise HeaderParseError(f"Invalid magic number: 0x{magic:08X}, expected 0x{GFWX_MAGIC:08X}")

    # Version
    version = reader.get_bits(32)
    if version != 1:
        raise HeaderParseError(f"Unsupported version: {version}")

    # Dimensions
    sizex = reader.get_bits(32)
    sizey = reader.get_bits(32)

    # Layers and channels (stored as value - 1)
    layers = reader.get_bits(16) + 1
    channels = reader.get_bits(16) + 1

    # Bit depth (stored as value - 1)
    bit_depth = reader.get_bits(8) + 1

    # Signed flag (1 bit) and quality (10 bits, stored as value - 1)
    is_signed = bool(reader.get_bits(1))
    quality = reader.get_bits(10) + 1

    # Chroma scale (stored as value - 1)
    chroma_scale = reader.get_bits(8) + 1

    # Block size (stored as value - 2) and filter
    block_size = reader.get_bits(5) + 2
    filter_val = reader.get_bits(8)

    # Quantization, encoder, intent
    quantization = reader.get_bits(8)
    encoder_val = reader.get_bits(8)
    intent_val = reader.get_bits(8)

    # Metadata size in 32-bit words
    metadata_size = reader.get_bits(32)

    # Convert to enums
    try:
        filter_type = Filter(filter_val)
    except ValueError:
        filter_type = Filter.LINEAR  # Default to linear for unknown

    try:
        encoder_type = Encoder(encoder_val)
    except ValueError:
        encoder_type = Encoder.FAST  # Default to fast for unknown

    try:
        intent_type = Intent(intent_val)
    except ValueError:
        intent_type = Intent.GENERIC  # Default to generic for unknown

    header = GFWXHeader(
        version=version,
        sizex=sizex,
        sizey=sizey,
        layers=layers,
        channels=channels,
        bit_depth=bit_depth,
        is_signed=is_signed,
        quality=quality,
        chroma_scale=chroma_scale,
        block_size=block_size,
        filter=filter_type,
        quantization=quantization,
        encoder=encoder_type,
        intent=intent_type,
        metadata_size=metadata_size,
    )

    # Calculate header size: fixed header + metadata
    # The reader has consumed the fixed header bits
    # Header bits: 32 + 32 + 32 + 32 + 16 + 16 + 8 + 1 + 10 + 8 + 5 + 8 + 8 + 8 + 8 + 32 = 256 bits = 32 bytes
    fixed_header_bytes = 32
    metadata_bytes = metadata_size * 4
    total_header_bytes = fixed_header_bytes + metadata_bytes

    return header, total_header_bytes


def read_metadata(data: bytes, header: GFWXHeader) -> bytes:
    """
    Read metadata bytes from the compressed data.

    Args:
        data: The compressed data bytes.
        header: The parsed header.

    Returns:
        The metadata bytes (may be empty if no metadata).
    """
    if header.metadata_size == 0:
        return b""

    # Metadata starts at byte 32 (after fixed header)
    metadata_start = 32
    metadata_end = metadata_start + header.metadata_size * 4

    if metadata_end > len(data):
        raise HeaderParseError("Data too short for metadata")

    return data[metadata_start:metadata_end]


def write_header(header: GFWXHeader, metadata: bytes = b"") -> bytes:
    """
    Write a GFWX header to bytes.

    The header format matches the SDK exactly:
    - Magic number (32 bits)
    - Version (32 bits)
    - sizex (32 bits)
    - sizey (32 bits)
    - layers - 1 (16 bits)
    - channels - 1 (16 bits)
    - bit_depth - 1 (8 bits)
    - is_signed (1 bit)
    - quality - 1 (10 bits)
    - chroma_scale - 1 (8 bits)
    - block_size - 2 (5 bits)
    - filter (8 bits)
    - quantization (8 bits)
    - encoder (8 bits)
    - intent (8 bits)
    - metadata_size in words (32 bits)
    - metadata (variable)

    Total fixed header: 256 bits = 32 bytes

    Args:
        header: The GFWXHeader to write.
        metadata: Optional metadata bytes (must be multiple of 4).

    Returns:
        The serialized header bytes.

    Raises:
        ValueError: If metadata length is not a multiple of 4.
    """
    if len(metadata) % 4 != 0:
        raise ValueError("Metadata length must be a multiple of 4 bytes")

    metadata_size_words = len(metadata) // 4

    # Allocate buffer: 8 words for fixed header + metadata
    buffer_words = 8 + metadata_size_words
    writer = BitWriter(buffer_words)

    # Magic number
    writer.put_bits(GFWX_MAGIC, 32)

    # Version (always 1)
    writer.put_bits(1, 32)

    # Dimensions
    writer.put_bits(header.sizex, 32)
    writer.put_bits(header.sizey, 32)

    # Layers and channels (stored as value - 1)
    writer.put_bits(header.layers - 1, 16)
    writer.put_bits(header.channels - 1, 16)

    # Bit depth (stored as value - 1)
    writer.put_bits(header.bit_depth - 1, 8)

    # Signed flag (1 bit)
    writer.put_bits(1 if header.is_signed else 0, 1)

    # Quality (stored as value - 1, 10 bits)
    writer.put_bits(header.quality - 1, 10)

    # Chroma scale (stored as value - 1, 8 bits)
    writer.put_bits(header.chroma_scale - 1, 8)

    # Block size (stored as value - 2, 5 bits)
    writer.put_bits(header.block_size - 2, 5)

    # Filter, quantization, encoder, intent (8 bits each)
    writer.put_bits(int(header.filter), 8)
    writer.put_bits(header.quantization, 8)
    writer.put_bits(int(header.encoder), 8)
    writer.put_bits(int(header.intent), 8)

    # Metadata size in words
    writer.put_bits(metadata_size_words, 32)

    # Get the header bytes
    header_bytes = writer.get_data()

    # Append metadata
    if metadata:
        header_bytes = header_bytes + metadata

    return header_bytes


def create_default_header(
    width: int,
    height: int,
    channels: int = 1,
    layers: int = 1,
    quality: int = QUALITY_MAX,
    bit_depth: int = 8,
    is_signed: bool = False,
    filter_type: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Intent = Intent.GENERIC,
    chroma_scale: int = 1,
    block_size: int = 7,
) -> GFWXHeader:
    """
    Create a GFWXHeader with default values.

    This is a convenience function for encoding.

    Args:
        width: Image width.
        height: Image height.
        channels: Number of channels (1=mono, 3=RGB, 4=RGBA).
        layers: Number of layers (typically 1).
        quality: Quality parameter (1-1024, 1024=lossless).
        bit_depth: Bits per sample (8 or 16).
        is_signed: Whether samples are signed.
        filter_type: Wavelet filter type.
        encoder: Encoder mode.
        intent: Color intent.
        chroma_scale: Chroma subsampling scale.
        block_size: Block size parameter (2-33, typical 7).

    Returns:
        A configured GFWXHeader.
    """
    return GFWXHeader(
        version=1,
        sizex=width,
        sizey=height,
        layers=layers,
        channels=channels,
        bit_depth=bit_depth,
        is_signed=is_signed,
        quality=quality,
        chroma_scale=chroma_scale,
        block_size=block_size,
        filter=filter_type,
        quantization=0,
        encoder=encoder,
        intent=intent,
        metadata_size=0,
    )

