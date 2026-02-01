"""
Debug script to compare Python encoder vs SDK encoder outputs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_codec.gfwx_sdk import encode as sdk_encode, decode as sdk_decode, is_sdk_available
from cross_codec.gfwx_sdk import Encoder as SDKEncoder
from src.pygfwx.core.codec import encode as py_encode, decode as py_decode
from src.pygfwx.core.header import parse_header, GFWXHeader, Encoder


def compare_small_image():
    """Compare encoding of a small test image."""
    print("=" * 60)
    print("ENCODER COMPARISON: Small Test Image")
    print("=" * 60)

    # Create a simple 64x64 gradient
    image = np.zeros((64, 64), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            image[y, x] = (x + y) * 2

    print(f"\nInput: {image.shape}, dtype={image.dtype}")
    print(f"Raw size: {image.nbytes} bytes")
    print(f"Sample values [0,0]={image[0,0]}, [32,32]={image[32,32]}, [63,63]={image[63,63]}")

    qualities = [64, 256, 512, 1024]

    for quality in qualities:
        print(f"\n--- Quality {quality} ---")

        # Python encode - use FAST for fair comparison
        py_data = py_encode(image, quality=quality, encoder=Encoder.FAST)
        py_header, _ = parse_header(py_data)
        print(f"Python FAST: {len(py_data):6d} bytes")

        # SDK encode with FAST (same encoder)
        if is_sdk_available():
            sdk_data = sdk_encode(image, quality=quality, encoder=SDKEncoder.FAST)
            sdk_header, _ = parse_header(sdk_data)
            print(f"SDK FAST:    {len(sdk_data):6d} bytes")
            print(f"FAST Ratio:  {len(py_data) / len(sdk_data):.2f}x")

            # Also test with CONTEXTUAL
            py_ctx_data = py_encode(image, quality=quality, encoder=Encoder.CONTEXTUAL)
            sdk_ctx_data = sdk_encode(image, quality=quality, encoder=SDKEncoder.CONTEXTUAL)
            print(f"Python CTX:  {len(py_ctx_data):6d} bytes")
            print(f"SDK CTX:     {len(sdk_ctx_data):6d} bytes")
            print(f"CTX Ratio:   {len(py_ctx_data) / len(sdk_ctx_data):.2f}x")

            # Decode both and compare (using FAST versions)
            py_decoded = py_decode(py_data)
            sdk_decoded = sdk_decode(sdk_data)

            if np.array_equal(image, py_decoded):
                print("Python FAST decode: EXACT match")
            else:
                py_diff = np.abs(image.astype(float) - py_decoded.astype(float))
                print(f"Python FAST decode: max diff={py_diff.max():.1f}, mean={py_diff.mean():.2f}")

            if np.array_equal(image, sdk_decoded):
                print("SDK FAST decode: EXACT match")
            else:
                sdk_diff = np.abs(image.astype(float) - sdk_decoded.astype(float))
                print(f"SDK FAST decode: max diff={sdk_diff.max():.1f}, mean={sdk_diff.mean():.2f}")
        else:
            print("SDK not available")


def dump_bitstream_structure():
    """Dump the structure of encoded bitstreams."""
    print("\n" + "=" * 60)
    print("BITSTREAM STRUCTURE COMPARISON")
    print("=" * 60)

    # Create small test image
    image = np.zeros((16, 16), dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            image[y, x] = (x + y) * 8

    quality = 256

    py_data = py_encode(image, quality=quality)
    py_header, _ = parse_header(py_data)

    print(f"\nPython encoded: {len(py_data)} bytes")
    print(f"Header: size={py_header.sizex}x{py_header.sizey}, q={py_header.quality}, block_size={py_header.block_size}")

    # Dump first 64 bytes as hex
    print("\nFirst 64 bytes (hex):")
    for i in range(0, min(64, len(py_data)), 16):
        hex_str = " ".join(f"{b:02x}" for b in py_data[i:i+16])
        print(f"  {i:04d}: {hex_str}")

    if is_sdk_available():
        sdk_data = sdk_encode(image, quality=quality)
        sdk_header, _ = parse_header(sdk_data)

        print(f"\nSDK encoded: {len(sdk_data)} bytes")
        print(f"Header: size={sdk_header.sizex}x{sdk_header.sizey}, q={sdk_header.quality}, block_size={sdk_header.block_size}")

        print("\nFirst 64 bytes (hex):")
        for i in range(0, min(64, len(sdk_data)), 16):
            hex_str = " ".join(f"{b:02x}" for b in sdk_data[i:i+16])
            print(f"  {i:04d}: {hex_str}")


def analyze_coefficient_encoding():
    """Analyze coefficient values before and after encoding."""
    print("\n" + "=" * 60)
    print("COEFFICIENT ANALYSIS")
    print("=" * 60)

    from src.pygfwx.core.lifting import lift
    from src.pygfwx.core.header import Filter, QUALITY_MAX
    from src.pygfwx.core.quantization import quantize

    # Small test image
    image = np.zeros((16, 16), dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            image[y, x] = (x + y) * 8

    quality = 256
    boost = 8

    # Convert to int32 and apply boost
    coeffs = image.astype(np.int32) * boost
    print(f"\nAfter boost (x{boost}):")
    print(f"  Range: {coeffs.min()} to {coeffs.max()}")

    # Apply wavelet transform
    lift(coeffs, 0, 0, 16, 16, 1, Filter.LINEAR)
    print(f"\nAfter wavelet lift:")
    print(f"  Range: {coeffs.min()} to {coeffs.max()}")
    print(f"  DC coefficient: {coeffs[0, 0]}")
    print(f"  Sample high-freq [0,1]={coeffs[0,1]}, [1,0]={coeffs[1,0]}, [1,1]={coeffs[1,1]}")

    # Apply quantization
    max_q = QUALITY_MAX * boost
    quantize(coeffs, 0, 0, 16, 16, 1, quality, 0, max_q)
    print(f"\nAfter quantization (q={quality}, max_q={max_q}):")
    print(f"  Range: {coeffs.min()} to {coeffs.max()}")
    print(f"  DC coefficient: {coeffs[0, 0]}")
    print(f"  Non-zero count: {np.sum(coeffs != 0)} / {coeffs.size}")
    print(f"  Unique values: {len(np.unique(coeffs))}")


if __name__ == "__main__":
    if not is_sdk_available():
        print("WARNING: GFWX SDK not available. Only Python encoder will be tested.")
        print("Build the SDK to enable comparison.\n")

    compare_small_image()
    dump_bitstream_structure()
    analyze_coefficient_encoding()
