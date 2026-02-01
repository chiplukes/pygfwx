# GFWX Bitstream Format

This document describes the GFWX bitstream format.

## Overview

The GFWX bitstream consists of:
1. Fixed header (28+ bytes)
2. Optional metadata
3. Encoded coefficient data (organized in blocks)

## Header Format

The header is stored in little-endian byte order.

### Fixed Header Fields (28 bytes minimum)

| Offset | Size (bytes) | Field | Description |
|--------|--------------|-------|-------------|
| 0 | 4 | magic | "GFWX" (0x58, 0x46, 0x57, 0x47 in bytes) |
| 4 | 4 | version | Format version (currently 1) |
| 8 | 4 | sizex | Image width in pixels |
| 12 | 4 | sizey | Image height in pixels |
| 16 | 2 | layers-1 | Number of layers minus 1 |
| 18 | 2 | channels-1 | Number of channels minus 1 |
| 20 | 1 | bitDepth-1 | Bits per sample minus 1 (7 = 8-bit) |
| 21 | 2 | isSigned + quality | 1 bit signed flag + 10 bits quality-1 |
| 23 | 1 | chromaScale-1 | Chroma quality divisor minus 1 |
| 24 | 1 | blockSize + filter | 5 bits blockSize-2 + 8 bits filter |
| 25 | 1 | quantization | Quantization type (always 0 = scalar) |
| 26 | 1 | encoder | Encoder type (0=Turbo, 1=Fast, 2=Contextual) |
| 27 | 1 | intent | Color intent |
| 28 | 4 | metaDataSize | Size of metadata in 32-bit words |
| 32+ | var | metadata | Optional metadata bytes |

### Bit-Packed Fields Detail

The `isSigned + quality` field (bytes 21-22):
```
Bit 0:      isSigned (0 = unsigned, 1 = signed)
Bits 1-10:  quality - 1 (range 0-1023, represents quality 1-1024)
```

The `blockSize + filter` field (byte 24):
```
Bits 0-4:   blockSize - 2 (actual block size = 2^(value+2))
Bits 5-12:  filter type
```

## Constants

### Magic Number
```python
GFWX_MAGIC = b'GFWX'  # or 0x58465747 as 32-bit little-endian
```

### Quality
```python
QUALITY_MAX = 1024      # Lossless
QUALITY_DEFAULT = 512   # Good balance
```

### Filter Types
```python
FILTER_LINEAR = 0   # 5/3 wavelet (better for lossless)
FILTER_CUBIC = 1    # 9/7 wavelet (better for lossy)
```

### Encoder Types
```python
ENCODER_TURBO = 0       # Fastest, lowest compression
ENCODER_FAST = 1        # Fast, medium compression
ENCODER_CONTEXTUAL = 2  # Slowest, best compression
```

### Intent Types
```python
INTENT_GENERIC = 0      # No color transform
INTENT_MONO = 1         # Grayscale
INTENT_BAYER_RGGB = 2   # Bayer pattern RGGB
INTENT_BAYER_BGGR = 3   # Bayer pattern BGGR
INTENT_BAYER_GRBG = 4   # Bayer pattern GRBG
INTENT_BAYER_GBRG = 5   # Bayer pattern GBRG
INTENT_BAYER_GENERIC = 6
INTENT_RGB = 7          # RGB color
INTENT_RGBA = 8         # RGBA with alpha
INTENT_RGBA_PM = 9      # RGBA premultiplied
INTENT_BGR = 10         # BGR color
INTENT_BGRA = 11        # BGRA with alpha
INTENT_BGRA_PM = 12     # BGRA premultiplied
INTENT_CMYK = 13        # CMYK print colors
```

## Block Structure

After the header, data is organized into blocks:

```
[Header]
[Metadata (optional)]
[Block 0 data]
[Block 1 data]
...
[Block N data]
```

Each block contains encoded wavelet coefficients for a spatial region.

## Coefficient Encoding Order

Within each block, coefficients are encoded in pyramid order:
1. DC coefficient (top-level LL band)
2. Top level detail bands (HL, LH, HH)
3. Next level detail bands
4. ... continuing to finest level

## Decoding Process

1. Parse header to get image dimensions and encoding parameters
2. Allocate image buffer based on dimensions, channels, bit depth
3. Read and decode coefficient blocks
4. Apply inverse quantization
5. Apply inverse wavelet transform
6. Apply inverse color transform (if applicable)

## Notes

- All multi-byte values are little-endian
- The format is designed for efficient streaming/progressive decode
- Metadata can contain arbitrary application-specific data
