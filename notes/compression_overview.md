# GFWX Wavelet Compression Overview

This document provides an overview of the GFWX compression algorithm.

## Pipeline Overview

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Color Transform    │  (Optional: UYV, A710, custom)
│  RGB → YUV          │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Wavelet Lifting    │  (Multi-level decomposition)
│  Forward Transform  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Quantization       │  (Scalar, quality-based)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Entropy Coding     │  (Golomb-Rice + zero runs)
│  Context Modeling   │
└─────────────────────┘
    │
    ▼
Compressed Bitstream
```

## Key Components

### 1. Color Transform

GFWX supports programmable color transforms:
- **UYV**: Standard YUV-like transform
- **A710**: High-quality color transform
- **Custom**: User-defined channel combinations

The transform is stored in the bitstream header and applied/reversed during encode/decode.

### 2. Wavelet Lifting Scheme

GFWX uses the lifting scheme (Sweldens, 1995) for wavelet transforms:

**Advantages of lifting:**
- In-place computation (memory efficient)
- Perfectly invertible (for 5/3 filter)
- Flexible filter design

**Filter options:**
- **Linear (5/3)**: Integer-only, perfectly invertible, best for lossless
- **Cubic (9/7)**: Better frequency separation, best for lossy

The transform is applied in 2D (horizontal then vertical) at multiple
resolution levels to create a wavelet pyramid.

See [wavelet_transform.md](wavelet_transform.md) for comprehensive details including:
- Mathematical foundations of wavelets
- Lifting scheme explained step-by-step
- Filter implementations (5/3 and 9/7)
- Multi-level decomposition
- Boundary handling

Also see [lifting_scheme.md](lifting_scheme.md) for implementation-focused notes.

### 3. Quantization

Simple scalar quantization with quality parameter:

```
Forward:  quantized = coefficient * quality / maxQ
Inverse:  coefficient = quantized * maxQ / quality
```

Where:
- `quality`: 1-1024 (1024 = lossless)
- `maxQ`: 1024 * 8 (boost factor)

Quality doubles at each pyramid level (coarser levels = higher quality).

### 4. Entropy Coding

GFWX uses Golomb-Rice coding with:
- **Power-of-two parameters**: Efficient bit manipulation
- **Zero-run encoding**: Exploits sparse coefficients
- **Context-adaptive selection**: Based on local statistics

See [golomb_rice.md](golomb_rice.md) for details.

## Comparison with Other Codecs

| Feature | GFWX | JPEG 2000 | CineForm |
|---------|------|-----------|----------|
| Wavelet | Lifting | DWT | Custom (2-6) |
| Entropy | Golomb-Rice | Arithmetic | VLC/Huffman |
| Complexity | ~1K lines | ~100K+ lines | ~50K lines |
| Progressive | Yes | Yes | Limited |
| Lossless | Yes | Yes | Yes |

## Quality Levels

| Quality | Compression | Use Case |
|---------|-------------|----------|
| 1-64 | Very high | Thumbnails, previews |
| 64-256 | High | Web images |
| 256-512 | Medium | General photography |
| 512-1000 | Low | Archival |
| 1024 | Lossless | Master files |

## Block Structure

GFWX organizes coefficients into blocks for parallel processing:
- Default block size: 2^(blockSize+2) = typically 64-256
- Blocks can be encoded/decoded independently
- Enables OpenMP parallelization

## References

- [GFWX Website](https://www.gfwx.org/)
- [GFWX Whitepaper](https://www.gfwx.org/gfwx.pdf)
- Sweldens, W. (1995). "The Lifting Scheme: A New Philosophy in Biorthogonal Wavelet Constructions"
