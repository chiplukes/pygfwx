"""
Python wrapper for the GFWX SDK via ctypes.

This module provides a Python interface to the GFWX C++ SDK,
enabling compression and decompression of images using the reference implementation.
"""

import ctypes
import platform
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

# =============================================================================
# Constants
# =============================================================================


class Filter(IntEnum):
    """Wavelet filter type."""

    LINEAR = 0  # 5/3 wavelet, better for lossless
    CUBIC = 1  # 9/7 wavelet, better for lossy


class Encoder(IntEnum):
    """Encoder mode (speed vs compression tradeoff)."""

    TURBO = 0  # Fastest, lowest compression
    FAST = 1  # Fast, medium compression
    CONTEXTUAL = 2  # Slowest, best compression


class Intent(IntEnum):
    """Color intent / image type."""

    GENERIC = 0  # No color transform
    MONO = 1  # Grayscale
    BAYER_RGGB = 2  # Bayer pattern RGGB
    BAYER_BGGR = 3  # Bayer pattern BGGR
    BAYER_GRBG = 4  # Bayer pattern GRBG
    BAYER_GBRG = 5  # Bayer pattern GBRG
    BAYER_GENERIC = 6  # Generic Bayer
    RGB = 7  # RGB color
    RGBA = 8  # RGBA with alpha
    RGBA_PREMULT = 9  # RGBA premultiplied
    BGR = 10  # BGR color
    BGRA = 11  # BGRA with alpha
    BGRA_PREMULT = 12  # BGRA premultiplied
    CMYK = 13  # CMYK print colors


class Result(IntEnum):
    """Result codes from SDK functions."""

    OK = 0
    ERROR_OVERFLOW = -1
    ERROR_MALFORMED = -2
    ERROR_TYPE_MISMATCH = -3
    ERROR_UNSUPPORTED = -4


# Quality constants
QUALITY_MAX = 1024  # Lossless quality
BLOCK_DEFAULT = 7


# =============================================================================
# Header Structure
# =============================================================================


class GFWXHeaderC(ctypes.Structure):
    """C-compatible header structure matching the SDK wrapper."""

    _fields_ = [
        ("sizex", ctypes.c_int32),
        ("sizey", ctypes.c_int32),
        ("layers", ctypes.c_int32),
        ("channels", ctypes.c_int32),
        ("bitDepth", ctypes.c_int32),
        ("quality", ctypes.c_int32),
        ("chromaScale", ctypes.c_int32),
        ("blockSize", ctypes.c_int32),
        ("filter", ctypes.c_int32),
        ("quantization", ctypes.c_int32),
        ("encoder", ctypes.c_int32),
        ("intent", ctypes.c_int32),
        ("version", ctypes.c_int32),
        ("isSigned", ctypes.c_int32),
    ]


@dataclass
class GFWXHeader:
    """Python-friendly GFWX header."""

    sizex: int
    sizey: int
    layers: int = 1
    channels: int = 1
    bit_depth: int = 8
    quality: int = QUALITY_MAX
    chroma_scale: int = 1
    block_size: int = BLOCK_DEFAULT
    filter: Filter = Filter.LINEAR
    quantization: int = 0
    encoder: Encoder = Encoder.CONTEXTUAL
    intent: Intent = Intent.GENERIC
    version: int = 1
    is_signed: bool = False

    def to_c(self) -> GFWXHeaderC:
        """Convert to C structure."""
        return GFWXHeaderC(
            sizex=self.sizex,
            sizey=self.sizey,
            layers=self.layers,
            channels=self.channels,
            bitDepth=self.bit_depth,
            quality=self.quality,
            chromaScale=self.chroma_scale,
            blockSize=self.block_size,
            filter=int(self.filter),
            quantization=self.quantization,
            encoder=int(self.encoder),
            intent=int(self.intent),
            version=self.version,
            isSigned=1 if self.is_signed else 0,
        )

    @classmethod
    def from_c(cls, c_header: GFWXHeaderC) -> "GFWXHeader":
        """Create from C structure."""
        return cls(
            sizex=c_header.sizex,
            sizey=c_header.sizey,
            layers=c_header.layers,
            channels=c_header.channels,
            bit_depth=c_header.bitDepth,
            quality=c_header.quality,
            chroma_scale=c_header.chromaScale,
            block_size=c_header.blockSize,
            filter=Filter(c_header.filter),
            quantization=c_header.quantization,
            encoder=Encoder(c_header.encoder),
            intent=Intent(c_header.intent),
            version=c_header.version,
            is_signed=c_header.isSigned != 0,
        )


# =============================================================================
# SDK Wrapper
# =============================================================================


def _find_library() -> Path | None:
    """Find the GFWX shared library."""
    # Determine library name based on platform
    system = platform.system()
    if system == "Windows":
        lib_names = ["gfwx.dll"]
    elif system == "Darwin":
        lib_names = ["libgfwx.dylib", "gfwx.dylib"]
    else:
        lib_names = ["libgfwx.so", "gfwx.so"]

    # Search paths relative to this file
    base_path = Path(__file__).parent.parent / "gfwx-sdk" / "build"
    search_dirs = [
        base_path / "Release",
        base_path / "Debug",
        base_path / "out" / "Release",
        base_path / "out" / "Debug",
        base_path / "out",
        base_path,
    ]

    for search_dir in search_dirs:
        for lib_name in lib_names:
            lib_path = search_dir / lib_name
            if lib_path.exists():
                return lib_path

    return None


class GFWXWrapper:
    """Python wrapper for the GFWX SDK."""

    def __init__(self, library_path: Path | None = None):
        """Initialize the wrapper.

        Args:
            library_path: Path to the GFWX shared library.
                          If None, will search standard locations.

        Raises:
            FileNotFoundError: If the library cannot be found.
            OSError: If the library cannot be loaded.
        """
        if library_path is None:
            library_path = _find_library()

        if library_path is None:
            raise FileNotFoundError(
                "GFWX library not found. Please build it first:\n"
                "  Windows: cd gfwx-sdk/build && .\\build_windows.ps1\n"
                "  Linux:   cd gfwx-sdk/build && ./build_linux.sh"
            )

        self._lib_path = library_path
        self._lib = ctypes.CDLL(str(library_path))
        self._setup_functions()

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        # Compress functions
        for suffix, dtype in [
            ("u8", ctypes.c_uint8),
            ("u16", ctypes.c_uint16),
            ("i8", ctypes.c_int8),
            ("i16", ctypes.c_int16),
        ]:
            func = getattr(self._lib, f"gfwx_compress_{suffix}")
            func.argtypes = [
                ctypes.POINTER(dtype),  # imageData
                ctypes.POINTER(GFWXHeaderC),  # header
                ctypes.POINTER(ctypes.c_uint8),  # buffer
                ctypes.c_size_t,  # bufferSize
                ctypes.POINTER(ctypes.c_int32),  # transform
            ]
            func.restype = ctypes.c_int64

        # Decompress functions
        for suffix, dtype in [
            ("u8", ctypes.c_uint8),
            ("u16", ctypes.c_uint16),
            ("i8", ctypes.c_int8),
            ("i16", ctypes.c_int16),
        ]:
            func = getattr(self._lib, f"gfwx_decompress_{suffix}")
            func.argtypes = [
                ctypes.POINTER(dtype),  # imageData
                ctypes.POINTER(GFWXHeaderC),  # header
                ctypes.POINTER(ctypes.c_uint8),  # data
                ctypes.c_size_t,  # dataSize
                ctypes.c_int32,  # downsampling
                ctypes.c_int32,  # test
            ]
            func.restype = ctypes.c_int64

        # Read header
        self._lib.gfwx_read_header.argtypes = [
            ctypes.POINTER(GFWXHeaderC),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self._lib.gfwx_read_header.restype = ctypes.c_int64

        # Buffer size
        self._lib.gfwx_buffer_size.argtypes = [ctypes.POINTER(GFWXHeaderC)]
        self._lib.gfwx_buffer_size.restype = ctypes.c_size_t

        # Transform functions
        self._lib.gfwx_transform_uyv.argtypes = []
        self._lib.gfwx_transform_uyv.restype = ctypes.POINTER(ctypes.c_int32)

        self._lib.gfwx_transform_a710_rgb.argtypes = []
        self._lib.gfwx_transform_a710_rgb.restype = ctypes.POINTER(ctypes.c_int32)

    def _get_dtype_suffix(self, dtype: np.dtype) -> str:
        """Get the function suffix for a numpy dtype."""
        if dtype == np.uint8:
            return "u8"
        elif dtype == np.uint16:
            return "u16"
        elif dtype == np.int8:
            return "i8"
        elif dtype == np.int16:
            return "i16"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def encode(
        self,
        image: np.ndarray,
        quality: int = QUALITY_MAX,
        filter: Filter = Filter.LINEAR,
        encoder: Encoder = Encoder.CONTEXTUAL,
        intent: Intent | None = None,
        chroma_scale: int = 1,
        use_transform: bool = True,
    ) -> bytes:
        """Compress an image using GFWX.

        Args:
            image: Input image as numpy array.
                   Shape: (H, W) for mono, (H, W, C) for multi-channel.
                   Dtype: uint8, uint16, int8, or int16.
            quality: Quality level (1-1024). 1024 = lossless.
            filter: Wavelet filter (LINEAR or CUBIC).
            encoder: Encoder mode (TURBO, FAST, or CONTEXTUAL).
            intent: Color intent. If None, auto-detected from shape.
            chroma_scale: Chroma quality divisor (1 = same as luma).
            use_transform: Whether to use color transform for RGB/RGBA.

        Returns:
            Compressed data as bytes.

        Raises:
            ValueError: If image format is not supported.
            RuntimeError: If compression fails.
        """
        # Ensure contiguous array
        image = np.ascontiguousarray(image)

        # Determine dimensions and channels
        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        elif image.ndim == 3:
            height, width, channels = image.shape
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        # Auto-detect intent
        if intent is None:
            if channels == 1:
                intent = Intent.MONO
            elif channels == 3:
                intent = Intent.RGB
            elif channels == 4:
                intent = Intent.RGBA
            else:
                intent = Intent.GENERIC

        # Create header
        header = GFWXHeader(
            sizex=width,
            sizey=height,
            layers=1,
            channels=channels,
            bit_depth=image.dtype.itemsize * 8,
            quality=quality,
            chroma_scale=chroma_scale,
            block_size=BLOCK_DEFAULT,
            filter=filter,
            quantization=0,
            encoder=encoder,
            intent=intent,
            is_signed=np.issubdtype(image.dtype, np.signedinteger),
        )

        # Get function suffix
        suffix = self._get_dtype_suffix(image.dtype)

        # Allocate output buffer (generous estimate)
        buffer_size = image.nbytes * 2 + 1024
        buffer = np.zeros(buffer_size, dtype=np.uint8)

        # Get transform
        transform = None
        if use_transform and channels >= 3 and intent in (Intent.RGB, Intent.RGBA, Intent.BGR, Intent.BGRA):
            transform = self._lib.gfwx_transform_uyv()

        # Compress
        c_header = header.to_c()
        compress_func = getattr(self._lib, f"gfwx_compress_{suffix}")
        result = compress_func(
            image.ctypes.data_as(
                ctypes.POINTER(
                    ctypes.c_uint8
                    if suffix == "u8"
                    else ctypes.c_uint16
                    if suffix == "u16"
                    else ctypes.c_int8
                    if suffix == "i8"
                    else ctypes.c_int16
                )
            ),
            ctypes.byref(c_header),
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            buffer_size,
            transform,
        )

        if result < 0:
            error_names = {-1: "overflow", -2: "malformed", -3: "type mismatch", -4: "unsupported"}
            raise RuntimeError(f"Compression failed: {error_names.get(result, f'error {result}')}")

        return bytes(buffer[:result])

    def encode_multi_layer(
        self,
        image: np.ndarray,
        layers: int,
        channels: int,
        quality: int = QUALITY_MAX,
        filter: Filter = Filter.LINEAR,
        encoder: Encoder = Encoder.CONTEXTUAL,
        intent: Intent | None = None,
        chroma_scale: int = 1,
        use_transform: bool = True,
    ) -> bytes:
        """Compress a multi-layer image using GFWX.

        Args:
            image: Input image as numpy array.
                   Shape: (H, W) for mono or (H, W, layers*channels) for multi.
                   Data should be interleaved: [L0C0, L0C1, ..., L1C0, L1C1, ...]
            layers: Number of layers (e.g., 2 for stereo).
            channels: Channels per layer (e.g., 3 for RGB).
            quality: Quality level (1-1024). 1024 = lossless.
            filter: Wavelet filter (LINEAR or CUBIC).
            encoder: Encoder mode (TURBO, FAST, or CONTEXTUAL).
            intent: Color intent. If None, auto-detected from channels.
            chroma_scale: Chroma quality divisor (1 = same as luma).
            use_transform: Whether to use color transform for RGB/RGBA.

        Returns:
            Compressed data as bytes.

        Raises:
            ValueError: If image format is not supported.
            RuntimeError: If compression fails.
        """
        # Ensure contiguous array
        image = np.ascontiguousarray(image)

        # Validate dimensions
        if image.ndim == 2:
            height, width = image.shape
            if layers * channels != 1:
                raise ValueError(
                    f"2D image requires layers*channels=1, got {layers}*{channels}"
                )
        elif image.ndim == 3:
            height, width, total = image.shape
            if total != layers * channels:
                raise ValueError(
                    f"Image has {total} channels but layers*channels={layers * channels}"
                )
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        # Auto-detect intent
        if intent is None:
            if channels == 1:
                intent = Intent.MONO
            elif channels == 3:
                intent = Intent.RGB
            elif channels == 4:
                intent = Intent.RGBA
            else:
                intent = Intent.GENERIC

        # Create header
        header = GFWXHeader(
            sizex=width,
            sizey=height,
            layers=layers,
            channels=channels,
            bit_depth=image.dtype.itemsize * 8,
            quality=quality,
            chroma_scale=chroma_scale,
            block_size=BLOCK_DEFAULT,
            filter=filter,
            quantization=0,
            encoder=encoder,
            intent=intent,
            is_signed=np.issubdtype(image.dtype, np.signedinteger),
        )

        # Get function suffix
        suffix = self._get_dtype_suffix(image.dtype)

        # Allocate output buffer
        buffer_size = image.nbytes * 2 + 1024
        buffer = np.zeros(buffer_size, dtype=np.uint8)

        # Get transform - only for RGB-like with >= 3 channels
        transform = None
        if use_transform and channels >= 3 and intent in (Intent.RGB, Intent.RGBA, Intent.BGR, Intent.BGRA):
            transform = self._lib.gfwx_transform_uyv()

        # Compress
        c_header = header.to_c()
        compress_func = getattr(self._lib, f"gfwx_compress_{suffix}")
        result = compress_func(
            image.ctypes.data_as(
                ctypes.POINTER(
                    ctypes.c_uint8
                    if suffix == "u8"
                    else ctypes.c_uint16
                    if suffix == "u16"
                    else ctypes.c_int8
                    if suffix == "i8"
                    else ctypes.c_int16
                )
            ),
            ctypes.byref(c_header),
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            buffer_size,
            transform,
        )

        if result < 0:
            error_names = {-1: "overflow", -2: "malformed", -3: "type mismatch", -4: "unsupported"}
            raise RuntimeError(f"Compression failed: {error_names.get(result, f'error {result}')}")

        return bytes(buffer[:result])

    def decode(
        self,
        data: bytes,
        downsampling: int = 0,
    ) -> np.ndarray:
        """Decompress GFWX data to an image.

        Args:
            data: Compressed GFWX data.
            downsampling: Downsampling level (0=full, 1=half, 2=quarter, etc.)

        Returns:
            Decompressed image as numpy array.
            For multi-layer images, shape is (H, W, layers*channels).

        Raises:
            RuntimeError: If decompression fails.
        """
        # Read header first
        header = self.read_header(data)

        # Determine output dimensions
        out_width = (header.sizex + (1 << downsampling) - 1) >> downsampling
        out_height = (header.sizey + (1 << downsampling) - 1) >> downsampling

        # Determine dtype
        if header.is_signed:
            dtype = np.int8 if header.bit_depth <= 8 else np.int16
        else:
            dtype = np.uint8 if header.bit_depth <= 8 else np.uint16

        # Allocate output - include all layers
        total_channels = header.layers * header.channels
        if total_channels == 1:
            output = np.zeros((out_height, out_width), dtype=dtype)
        else:
            output = np.zeros((out_height, out_width, total_channels), dtype=dtype)

        output = np.ascontiguousarray(output)

        # Get function suffix
        suffix = self._get_dtype_suffix(dtype)

        # Create data buffer
        data_array = np.frombuffer(data, dtype=np.uint8)

        # Decompress
        c_header = GFWXHeaderC()
        decompress_func = getattr(self._lib, f"gfwx_decompress_{suffix}")
        result = decompress_func(
            output.ctypes.data_as(
                ctypes.POINTER(
                    ctypes.c_uint8
                    if suffix == "u8"
                    else ctypes.c_uint16
                    if suffix == "u16"
                    else ctypes.c_int8
                    if suffix == "i8"
                    else ctypes.c_int16
                )
            ),
            ctypes.byref(c_header),
            data_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(data),
            downsampling,
            0,  # test = false
        )

        if result < 0:
            error_names = {-1: "overflow", -2: "malformed", -3: "type mismatch", -4: "unsupported"}
            raise RuntimeError(f"Decompression failed: {error_names.get(result, f'error {result}')}")

        return output

    def read_header(self, data: bytes) -> GFWXHeader:
        """Read header from compressed data without decompressing.

        Args:
            data: Compressed GFWX data (at least 28 bytes).

        Returns:
            GFWXHeader with file information.

        Raises:
            ValueError: If data is too short.
            RuntimeError: If header is malformed.
        """
        if len(data) < 28:
            raise ValueError("Data too short for GFWX header (need at least 28 bytes)")

        data_array = np.frombuffer(data, dtype=np.uint8)
        c_header = GFWXHeaderC()

        result = self._lib.gfwx_read_header(
            ctypes.byref(c_header),
            data_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(data),
        )

        if result < 0:
            raise RuntimeError(f"Failed to read header: error {result}")

        return GFWXHeader.from_c(c_header)

    @property
    def library_path(self) -> Path:
        """Path to the loaded library."""
        return self._lib_path


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_wrapper: GFWXWrapper | None = None


def _get_wrapper() -> GFWXWrapper:
    """Get or create the default wrapper instance."""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = GFWXWrapper()
    return _default_wrapper


def is_sdk_available() -> bool:
    """Check if the GFWX SDK is available."""
    return _find_library() is not None


def encode(
    image: np.ndarray,
    quality: int = QUALITY_MAX,
    filter: Filter = Filter.LINEAR,
    encoder: Encoder = Encoder.CONTEXTUAL,
    intent: Intent | None = None,
    chroma_scale: int = 1,
    use_transform: bool = True,
) -> bytes:
    """Compress an image using GFWX. See GFWXWrapper.encode for details."""
    return _get_wrapper().encode(image, quality, filter, encoder, intent, chroma_scale, use_transform)


def decode(data: bytes, downsampling: int = 0) -> np.ndarray:
    """Decompress GFWX data to an image. See GFWXWrapper.decode for details."""
    return _get_wrapper().decode(data, downsampling)


def read_header(data: bytes) -> GFWXHeader:
    """Read header from compressed data. See GFWXWrapper.read_header for details."""
    return _get_wrapper().read_header(data)
