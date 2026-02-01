#!/bin/bash
# Build script for GFWX SDK wrapper on Linux/macOS
# Automatically finds CMake and compiler

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${1:-Release}"
CLEAN="${2:-}"

echo "=== GFWX SDK Build Script (Linux/macOS) ==="

# Check for CMake
if command -v cmake &> /dev/null; then
    CMAKE="cmake"
    echo "Found CMake: $(which cmake)"
else
    echo "ERROR: CMake not found!"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt install cmake"
    echo "  Fedora: sudo dnf install cmake"
    echo "  macOS: brew install cmake"
    exit 1
fi

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    echo "Found g++: $(which g++)"
elif command -v clang++ &> /dev/null; then
    echo "Found clang++: $(which clang++)"
else
    echo "ERROR: No C++ compiler found!"
    echo "Please install a C++ compiler:"
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    echo "  Fedora: sudo dnf install gcc-c++"
    echo "  macOS: xcode-select --install"
    exit 1
fi

# Set up build directory
BUILD_DIR="$SCRIPT_DIR/out"
OUTPUT_DIR="$SCRIPT_DIR/$BUILD_TYPE"

if [ "$CLEAN" = "--clean" ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Configure
echo ""
echo "Configuring..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use Ninja if available, otherwise Unix Makefiles
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

$CMAKE -S "$SCRIPT_DIR" -B "$BUILD_DIR" -G "$GENERATOR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build
echo ""
echo "Building ($BUILD_TYPE)..."
$CMAKE --build "$BUILD_DIR" --config "$BUILD_TYPE"

# Copy output
echo ""
echo "Copying output..."
mkdir -p "$OUTPUT_DIR"

# Find the library (different extension on Linux vs macOS)
if [ "$(uname)" = "Darwin" ]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi

LIB_FILE=$(find "$BUILD_DIR" -name "libgfwx.$LIB_EXT" -o -name "gfwx.$LIB_EXT" 2>/dev/null | head -1)
if [ -n "$LIB_FILE" ]; then
    cp "$LIB_FILE" "$OUTPUT_DIR/"
    echo "Output: $OUTPUT_DIR/$(basename "$LIB_FILE")"
else
    echo "WARNING: Library not found in build output"
fi

echo ""
echo "=== Build Complete ==="
