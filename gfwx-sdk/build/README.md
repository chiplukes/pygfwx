# GFWX SDK Build Directory

This directory contains the build infrastructure for the GFWX C++ SDK wrapper.

## Files

- `CMakeLists.txt` - CMake build configuration
- `gfwx_wrapper.cpp` - C wrapper exposing GFWX functions for ctypes
- `build_windows.ps1` - Windows build script
- `build_linux.sh` - Linux/macOS build script

## Building

### Windows (PowerShell)

```powershell
# Build Release (default)
.\build_windows.ps1

# Build Debug
.\build_windows.ps1 -Configuration Debug

# Clean and rebuild
.\build_windows.ps1 -Clean
```

Requirements:
- Visual Studio 2019 or 2022 with C++ workload
- CMake (included with Visual Studio, or install separately)

### Linux / macOS

```bash
# Make script executable (first time only)
chmod +x build_linux.sh

# Build Release (default)
./build_linux.sh

# Build Debug
./build_linux.sh Debug

# Clean and rebuild
./build_linux.sh Release --clean
```

Requirements:
- CMake (`apt install cmake` or `brew install cmake`)
- GCC or Clang (`apt install build-essential` or `xcode-select --install`)

## Output

After building, the shared library will be placed in:
- Windows: `Release/gfwx.dll` or `Debug/gfwx.dll`
- Linux: `Release/libgfwx.so` or `Debug/libgfwx.so`
- macOS: `Release/libgfwx.dylib` or `Debug/libgfwx.dylib`

## Wrapper API

The wrapper exports C functions that can be called from Python via ctypes:

### Compression
- `gfwx_compress_u8()` - Compress 8-bit unsigned images
- `gfwx_compress_u16()` - Compress 16-bit unsigned images
- `gfwx_compress_i8()` - Compress 8-bit signed images
- `gfwx_compress_i16()` - Compress 16-bit signed images

### Decompression
- `gfwx_decompress_u8()` - Decompress to 8-bit unsigned
- `gfwx_decompress_u16()` - Decompress to 16-bit unsigned
- `gfwx_decompress_i8()` - Decompress to 8-bit signed
- `gfwx_decompress_i16()` - Decompress to 16-bit signed

### Utilities
- `gfwx_read_header()` - Read header without decompression
- `gfwx_buffer_size()` - Calculate required buffer size
- `gfwx_transform_uyv()` - Get UYV color transform
- `gfwx_transform_a710_rgb()` - Get A710 transform for RGB
- `gfwx_quality_max()`, `gfwx_filter_linear()`, etc. - Constants
