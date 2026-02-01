"""
Basic GFWX Usage Example.

Demonstrates simple encode/decode operations using the pure Python codec.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Demonstrate basic GFWX encode/decode."""
    print("PyGFWX Basic Usage Example")
    print("=" * 40)

    from pygfwx import encode, decode, QUALITY_MAX

    # Create a simple test image
    print("\n1. Creating test image...")
    width, height = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw a gradient
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = x  # R increases left to right
            image[y, x, 1] = y  # G increases top to bottom
            image[y, x, 2] = 128  # B constant

    print(f"   Image shape: {image.shape}")
    print(f"   Raw size: {image.nbytes} bytes")

    # Encode at different quality levels
    print("\n2. Encoding at different quality levels...")
    qualities = [256, 512, QUALITY_MAX]

    for quality in qualities:
        # Encode
        compressed = encode(image, quality=quality)

        # Decode
        decoded = decode(compressed)

        # Compare
        if quality == QUALITY_MAX:
            is_lossless = np.array_equal(image, decoded)
            status = "Lossless" if is_lossless else "Not lossless"
        else:
            mse = np.mean((image.astype(float) - decoded.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
            status = f"PSNR: {psnr:.2f} dB"

        ratio = image.nbytes / len(compressed)
        print(f"   Quality {quality:4d}: {len(compressed):6d} bytes ({ratio:.1f}x compression) - {status}")

    # Demonstrate grayscale
    print("\n3. Grayscale encoding...")
    gray = image[:, :, 0]  # Use R channel as grayscale
    compressed_gray = encode(gray, quality=512)
    _ = decode(compressed_gray)  # Verify decode works
    print(f"   Grayscale shape: {gray.shape}")
    print(f"   Compressed: {len(compressed_gray)} bytes")
    print(f"   Compression ratio: {gray.nbytes / len(compressed_gray):.1f}x")

    print("\n" + "=" * 40)
    print("Example complete!")


if __name__ == "__main__":
    main()
