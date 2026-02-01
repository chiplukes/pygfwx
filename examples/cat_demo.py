"""
Generate cat image demo for README.

Encodes the cat demo image with GFWX at various quality levels
and generates comparison metrics.

Image Credit:
    Orange tabby cat sitting on fallen leaves
    Author: Hisashi from Japan (derivative work: Caspian blue)
    License: CC BY-SA 2.0
    Source: https://commons.wikimedia.org/wiki/File:Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Paths
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "cat_demo_outputs"
CAT_IMAGE_PATH = ASSETS_DIR / "cat_demo.png"

# Add parent src directory to path for pygfwx import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add parent directory for cross_codec SDK wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def save_image(data: np.ndarray, path: Path) -> None:
    """Save numpy array as image."""
    img = Image.fromarray(data)
    img.save(path)


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim_simple(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate a simplified SSIM (structural similarity)."""
    # Simple correlation-based similarity
    orig_flat = original.astype(np.float64).flatten()
    comp_flat = compressed.astype(np.float64).flatten()

    # Normalize
    orig_norm = (orig_flat - orig_flat.mean()) / (orig_flat.std() + 1e-10)
    comp_norm = (comp_flat - comp_flat.mean()) / (comp_flat.std() + 1e-10)

    # Correlation
    correlation = np.mean(orig_norm * comp_norm)
    return max(0.0, min(1.0, correlation))


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def run_python_codec(original: np.ndarray, qualities: list, output_dir: Path) -> None:
    """Run compression demo using pure Python codec."""
    from pygfwx import encode, decode, QUALITY_MAX

    print("\n[Python Codec]")
    print("-" * 60)
    print(f"{'Quality':<10} {'Size':<12} {'Ratio':<10} {'PSNR':<12}")
    print("-" * 60)

    original_size = original.nbytes

    for quality in qualities:
        # Encode
        compressed = encode(original, quality=quality)
        comp_size = len(compressed)
        ratio = original_size / comp_size

        # Decode
        decoded = decode(compressed)

        # Calculate metrics
        if quality == QUALITY_MAX:
            psnr = "inf (lossless)"
        else:
            psnr_val = calculate_psnr(original, decoded)
            psnr = f"{psnr_val:.2f} dB"

        print(f"{quality:<10} {format_size(comp_size):<12} {ratio:.1f}x{'':<6} {psnr:<12}")

        # Save decoded image for visual comparison
        output_path = output_dir / f"python_q{quality}.png"
        save_image(decoded, output_path)

    print("-" * 60)


def run_sdk_codec(original: np.ndarray, qualities: list, output_dir: Path) -> bool:
    """Run compression demo using C SDK. Returns True if successful."""
    try:
        from cross_codec.gfwx_sdk import encode, decode, is_sdk_available

        if not is_sdk_available():
            print("\n[SDK Codec] Not available - build the DLL in gfwx-sdk/build/")
            return False
    except ImportError as e:
        print(f"\n[SDK Codec] Import failed: {e}")
        return False

    print("\n[SDK Codec]")
    print("-" * 60)
    print(f"{'Quality':<10} {'Size':<12} {'Ratio':<10} {'PSNR':<12}")
    print("-" * 60)

    original_size = original.nbytes

    for quality in qualities:
        # Encode
        compressed = encode(original, quality=quality)
        comp_size = len(compressed)
        ratio = original_size / comp_size

        # Decode
        decoded = decode(compressed)

        # Calculate metrics
        if quality == 1024:
            psnr = "inf (lossless)"
        else:
            psnr_val = calculate_psnr(original, decoded)
            psnr = f"{psnr_val:.2f} dB"

        print(f"{quality:<10} {format_size(comp_size):<12} {ratio:.1f}x{'':<6} {psnr:<12}")

        # Save decoded image for visual comparison
        output_path = output_dir / f"sdk_q{quality}.png"
        save_image(decoded, output_path)

    print("-" * 60)
    return True


def main():
    """Generate cat image demo."""
    parser = argparse.ArgumentParser(description="GFWX Cat Image Demo")
    parser.add_argument(
        "--sdk", action="store_true",
        help="Also run SDK codec for comparison (requires built DLL)"
    )
    parser.add_argument(
        "--sdk-only", action="store_true",
        help="Only run SDK codec (skip Python codec)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PyGFWX Cat Image Demo Generator")
    print("=" * 60)

    # Load the cat image
    if not CAT_IMAGE_PATH.exists():
        print(f"Error: Cat demo image not found at {CAT_IMAGE_PATH}")
        return

    original = load_image(CAT_IMAGE_PATH)
    print(f"\nImage: {CAT_IMAGE_PATH.name}")
    print(f"Size: {original.shape} ({original.dtype})")
    print(f"File size: {format_size(CAT_IMAGE_PATH.stat().st_size)}")
    print(f"Raw size: {format_size(original.nbytes)}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Quality levels to test
    from pygfwx import QUALITY_MAX
    qualities = [64, 256, 512, QUALITY_MAX]

    # Run Python codec
    if not args.sdk_only:
        run_python_codec(original, qualities, OUTPUT_DIR)

    # Run SDK codec if requested
    if args.sdk or args.sdk_only:
        run_sdk_codec(original, qualities, OUTPUT_DIR)

    print(f"\nCompressed images saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
