# PyGFWX Development Roadmap

A Python implementation of the GFWX (Good, Fast Wavelet Codec) for educational purposes, validated against the reference C++ SDK.

**Purpose:** Educational wavelet codec implementation in Python
**Status:** Core codec complete (385+ tests passing)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why GFWX](#why-gfwx)
3. [Project Structure](#project-structure)
4. [Implementation Status](#implementation-status)
5. [Technical Reference](#technical-reference)
6. [Future Enhancements](#future-enhancements)

---

## Project Overview

### What is GFWX?

GFWX (Good, Fast Wavelet Codec) is a wavelet-based image compression format created by Graham Fyffe at USC Institute for Creative Technologies. Key characteristics:

- **~1000 lines of C++** - dramatically simpler than alternatives like CineForm
- **Single header file** - `gfwx.h` contains the entire implementation
- **No dependencies** - uses only standard C++11 libraries
- **BSD 3-Clause License** - permissive open source

### GFWX Capabilities

| Feature | Support |
|---------|---------|
| Lossy/Lossless | Both |
| Bit depths | 8-16 bit |
| Channels | Up to 65536 |
| Layers/Frames | Up to 65536 |
| Bayer support | Yes |
| Progressive decode | Yes (with downsampling) |
| Chroma downsampling | Yes |
| Color transforms | Programmable (UYV, A710) |
| Filters | Linear (5/3), Cubic (9/7) |
| Entropy coding | Golomb-Rice + zero runs |

---

## Why GFWX

### Advantages for Educational Implementation

1. **Single-file SDK**: The entire codec is one ~1000 line header file
2. **No build complexity**: Include header, done
3. **Clear algorithm flow**: Linear code path, no callback/plugin architecture
4. **Flexible formats**: Native mono support, arbitrary channels
5. **Modern patterns**: Clean C++11, templates, RAII

### Algorithm Summary

| Component | Technique |
|-----------|-----------|
| Wavelet | 5/3 (Linear) or 9/7 (Cubic) lifting |
| Transform | In-place lifting scheme |
| Quantization | Simple scalar (quality/1024) |
| Entropy | Golomb-Rice + zero-run coding |
| Bitstream | Simple header + raw coefficient data |

---

## Project Structure

```
pygfwx/
├── src/pygfwx/                 # Main Python package
│   ├── core/                   # Core codec implementation
│   │   ├── bitstream.py        # Bit-level I/O
│   │   ├── header.py           # GFWX header parsing/writing
│   │   ├── lifting.py          # Wavelet lifting (5/3, 9/7)
│   │   ├── quantization.py     # Scalar quantization
│   │   ├── golomb_rice.py      # Golomb-Rice coding
│   │   ├── context.py          # Adaptive context modeling
│   │   ├── encoder.py          # Coefficient encoding
│   │   ├── decoder.py          # Coefficient decoding
│   │   ├── codec.py            # High-level encode/decode API
│   │   └── transforms.py       # Color transforms (UYV, A710)
│   └── utils/                  # Utility functions
│       └── image_io.py         # Image loading/saving/generation
│
├── cross_codec/                # SDK wrapper and validation
│   └── gfwx_sdk.py             # Python wrapper for GFWX SDK (ctypes)
│
├── tests/                      # Pytest test suite (385+ tests)
│
├── examples/                   # Example scripts
│   ├── basic_usage.py          # Simple encode/decode
│   ├── compression_walkthrough.py  # Educational step-by-step
│   └── progressive_demo.py     # Progressive decoding demo
│
├── gfwx-sdk/                   # Reference C++ SDK
│   └── gfwx.h                  # The entire SDK (single header)
│
└── notes/                      # Technical documentation
    ├── compression_overview.md
    ├── bitstream.md
    ├── lifting_scheme.md
    ├── golomb_rice.md
    └── context_modeling.md
```

---

## Implementation Status

### Completed Features

- Full decode pipeline (header, entropy, dequant, inverse wavelet)
- Full encode pipeline (wavelet, quantize, entropy, bitstream)
- Lossless and lossy compression
- 8-bit and 16-bit support
- Mono, RGB, and multi-channel images
- Linear (5/3) and Cubic (9/7) wavelets
- Progressive decoding with downsampling
- Multi-layer support
- Bayer/RAW image support
- Color transforms (UYV, A710)
- Metadata support
- SDK cross-validation tests

### Test Coverage

- 385+ tests covering all codec components
- Roundtrip validation for various image types
- Cross-validation against reference C++ SDK

---

## Technical Reference

### GFWX Header Format

```
Offset  Size   Field
------  -----  --------------
0       4      Magic ("GFWX")
4       4      Version
8       4      Width
12      4      Height
16      2      Layers - 1
18      2      Channels - 1
20      1      BitDepth - 1
21      ~1.1   isSigned + quality
22      ~1     ChromaScale - 1
23      ~1.6   BlockSize + Filter
...
```

See [bitstream.md](bitstream.md) for complete format details.

### Quality Levels

| Quality | Description |
|---------|-------------|
| 1 | Maximum compression |
| 64 | Good balance |
| 256 | High quality |
| 512 | Very high quality |
| 1024 | Lossless |

### Filter Types

| Value | Name | Best For |
|-------|------|----------|
| 0 | Linear (5/3) | Lossless |
| 1 | Cubic (9/7) | Lossy |

### Encoder Types

| Value | Name | Speed | Compression |
|-------|------|-------|-------------|
| 0 | Turbo | Fastest | Lowest |
| 1 | Fast | Fast | Medium |
| 2 | Contextual | Slowest | Best |

---

## Future Enhancements

Potential areas for expansion:

1. **Streaming/Row-Based Processing** - Memory-efficient encoding
2. **Quality Metrics** - PSNR, SSIM calculation
3. **Interactive Visualization** - Web-based wavelet explorer
4. **Performance Optimization** - NumPy vectorization, optional Numba
5. **Command-Line Tool** - Full-featured CLI with batch processing

---

## References

1. **GFWX Website**: https://www.gfwx.org/
2. **GFWX Source**: https://github.com/kalcutter/gfwx
3. **Sweldens (1995)**: "The Lifting Scheme: A New Philosophy in Biorthogonal Wavelet Constructions"
4. **Rice & Plaunt (1971)**: "Adaptive Variable-Length Coding for Efficient Compression"
