# GFWX SDK API Reference

Technical documentation for the GFWX C++ SDK.

## Header Structure

The GFWX header is written at the start of every compressed stream:

```cpp
struct Header {
    int version;       // Currently 1
    int sizex;         // Image width
    int sizey;         // Image height  
    int layers;        // Number of layers (1-65536)
    int channels;      // Number of channels (1-65536)
    int bitDepth;      // Bits per sample (8 or 16)
    int isSigned;      // 1 if signed integers, 0 if unsigned
    int quality;       // Quality 1-1024 (1024 = lossless)
    int chromaScale;   // Chroma downsampling (0=none, 1=2x, 2=4x)
    int blockSize;     // Tile size: 16, 32, 64, 128, or 256
    int filter;        // 0=Linear (5/3), 1=Cubic (9/7)
    int quantization;  // 0=Scalar (only supported type)
    int encoder;       // 1=Fast, 2=Contextual, 3=HighBitrate
    int intent;        // 0=Generic, 1=RGB, 2=RGBA, 3=BGR, 4=BGRA, 5=Bayer
    int transform[4];  // Color transform coefficients (row-major 2x2)
};
```

## Constants

### Filters
| Name | Value | Description |
|------|-------|-------------|
| FilterLinear | 0 | 5/3 wavelet (lossless-capable) |
| FilterCubic | 1 | 9/7 wavelet (better lossy quality) |

### Encoders
| Name | Value | Description | Notes |
|------|-------|-------------|-------|
| EncoderTurbo | 0 | Fastest | **Deprecated** - returns ErrorUnsupported in v1 |
| EncoderFast | 1 | Fast mode | Default choice |
| EncoderContextual | 2 | Context modeling | Better compression |
| EncoderHighBitrate | 3 | High bitrate | Best quality at high bitrates |

### Intents
| Name | Value | Color Order | Transform |
|------|-------|-------------|-----------|
| IntentGeneric | 0 | As-is | None |
| IntentRGB | 1 | RGB | UYV |
| IntentRGBA | 2 | RGBA | UYV (first 3) |
| IntentBGR | 3 | BGR | UYV |
| IntentBGRA | 4 | BGRA | UYV (first 3) |
| IntentBayerRGGB | 5-8 | Bayer pattern | Special |

### Error Codes
| Name | Value | Description |
|------|-------|-------------|
| ResultOk | 0 | Success |
| ErrorOverflow | -1 | Buffer overflow |
| ErrorMalformed | -2 | Invalid data |
| ErrorTypeMismatch | -3 | Wrong data type |
| ErrorUnsupported | -4 | Unsupported feature |

## API Functions

### compress<T>
```cpp
int compress(T * buffer, int bufferSize, Header & header, 
             T const * imageData, int imageStride,
             T * auxData, int auxStride)
```

Compresses an image to the output buffer.

**Parameters:**
- `buffer`: Output buffer for compressed data
- `bufferSize`: Size of output buffer in bytes
- `header`: Header struct with configuration (modified with version)
- `imageData`: Input image pixel data (row-major, channel-interleaved)
- `imageStride`: Bytes per row in imageData
- `auxData`: Auxiliary workspace buffer (same size as image)
- `auxStride`: Bytes per row in auxData

**Returns:** Compressed size in bytes, or negative error code

### decompress<T>
```cpp
int decompress(T * imageData, int imageStride,
               T * buffer, int bufferSize,
               Header & header, int downsampling,
               T * auxData, int auxStride)
```

Decompresses data to an image buffer.

**Parameters:**
- `imageData`: Output image buffer
- `imageStride`: Bytes per row in output
- `buffer`: Compressed data
- `bufferSize`: Size of compressed data
- `header`: Output header (filled by function)
- `downsampling`: Reduction factor (0=full, 1=half, 2=quarter, ...)
- `auxData`: Auxiliary workspace
- `auxStride`: Bytes per row in workspace

**Returns:** 0 on success, negative error code on failure

### getSize
```cpp
int getSize(Header const & header, int downsampling)
```

Calculate image size after decoding with optional downsampling.

## Quality Settings

| Quality | Behavior |
|---------|----------|
| 1024 | Lossless |
| 512-1023 | Near-lossless |
| 256-511 | High quality lossy |
| 128-255 | Medium quality |
| 64-127 | Low quality |
| 1-63 | Very low quality |

Quality represents the quantization divisor: `quantized = coefficient * quality / 1024`

## Color Transforms

The transform matrix converts RGB to a decorrelated space:

**UYV Transform (default for RGB):**
```
Y = (R + 2*G + B) / 4
U = R - G
V = B - G
```

**A710 Transform:**
```
Y = (3*R + 3*G + 2*B) / 8  
U = R - G
V = (R + G) / 2 - B
```

## Block-Based Processing

Images are divided into blocks (tiles) for parallel encoding:
- Block sizes: 16, 32, 64, 128, 256 pixels
- Blocks are encoded independently for parallelism
- Default block size: 128

## SDK Version Notes

### Version 1 (Current)
- EncoderTurbo (0) is NOT supported
- Minimum encoder is EncoderFast (1)
- Maximum image dimension: 2^30 pixels

## Building the SDK

The SDK is a header-only C++ library. For Python ctypes integration, a C wrapper is built as a shared library.

### Windows
```powershell
cd gfwx-sdk/build
.\build_windows.ps1
```

### Linux
```bash
cd gfwx-sdk/build
./build_linux.sh
```

## Python Wrapper

The Python ctypes wrapper is in `cross_codec/gfwx_sdk.py`:

```python
from cross_codec.gfwx_sdk import GFWXWrapper, Filter, Encoder

# Create wrapper
sdk = GFWXWrapper()

# Encode
compressed = sdk.encode(
    image,           # numpy array (H, W) or (H, W, C)
    quality=1024,    # lossless
    filter=Filter.LINEAR,
    encoder=Encoder.FAST
)

# Decode
decoded = sdk.decode(compressed)

# Read header
header = sdk.read_header(compressed)
```
