"""
Progressive Decoding Demo.

Demonstrates GFWX's progressive decoding capability, showing how
partial data can be decoded to get increasingly detailed images.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets"


def main():
    """Demonstrate progressive decoding."""
    print("PyGFWX Progressive Decoding Demo")
    print("=" * 50)

    from pygfwx import encode, decode

    # Check for cat image
    cat_path = ASSETS_DIR / "cat_demo.png"
    if not cat_path.exists():
        print(f"\nCat image not found at {cat_path}")
        print("Please add a test image to the assets folder.")
        return

    # Load test image
    print("\n1. Loading test image...")
    img = Image.open(cat_path)
    original = np.array(img)
    print(f"   Image: {original.shape}")

    # Encode with high quality
    print("\n2. Encoding at quality 512...")
    compressed = encode(original, quality=512)
    total_size = len(compressed)
    print(f"   Compressed size: {total_size} bytes")

    # Decode at different truncation points
    print("\n3. Progressive decoding at different data amounts...")
    print("-" * 50)

    truncation_points = [0.1, 0.25, 0.5, 0.75, 1.0]

    for fraction in truncation_points:
        truncated_size = int(total_size * fraction)
        truncated_data = compressed[:truncated_size]

        try:
            # Try to decode truncated data
            decoded = decode(truncated_data)

            # Calculate PSNR
            mse = np.mean((original.astype(float) - decoded.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

            print(f"   {fraction * 100:5.1f}% ({truncated_size:6d} bytes): PSNR = {psnr:.2f} dB")

            # Save progressive result
            output_path = ASSETS_DIR / f"progressive_{int(fraction * 100)}.png"
            Image.fromarray(decoded).save(output_path)
        except Exception as e:
            print(f"   {fraction * 100:5.1f}% ({truncated_size:6d} bytes): Decode failed - {e}")

    print("-" * 50)

    # Demonstrate downsampled decoding
    print("\n4. Downsampled decoding (for quick previews)...")
    print("-" * 50)

    for downsample in range(4):
        factor = 2**downsample
        decoded = decode(compressed, downsampling=downsample)

        print(f"   Downsample {downsample}: {decoded.shape[1]}x{decoded.shape[0]} (1/{factor} scale)")

        # Save downsampled result
        output_path = ASSETS_DIR / f"downsampled_{factor}x.png"
        Image.fromarray(decoded).save(output_path)

    print("-" * 50)

    print(f"\nProgressive images saved to {ASSETS_DIR}/")
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()
