"""
GFWX Compression Pipeline Walkthrough.

This educational script demonstrates how the GFWX codec compresses images
step by step. Each stage shows intermediate results and explains the concepts.

Pipeline stages:
1. Original Image - Load and examine input
2. Color Transform - Convert RGB to YUV-like representation
3. Wavelet Transform - Decompose into frequency bands
4. Quantization - Reduce precision for compression
5. Entropy Coding - Golomb-Rice variable-length coding
6. Reconstruction - Decode and compare to original

Run with: uv run examples/compression_walkthrough.py
"""

import sys
from pathlib import Path

import numpy as np

# Add parent src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Output directory for visualization images
OUTPUT_DIR = Path(__file__).parent / "walkthrough_outputs"


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(subtitle: str) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * 50)
    print(f"  {subtitle}")
    print("-" * 50)


def ensure_matplotlib():
    """Import matplotlib or provide instructions."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("\nMatplotlib not available for visualizations.")
        print("Install with: uv pip install matplotlib")
        return None


# =============================================================================
# STEP 1: ORIGINAL IMAGE
# =============================================================================


def create_test_image(size: int = 64) -> np.ndarray:
    """
    Create a test image with patterns designed to show wavelet behavior.

    Layout (64x64 default) - 4 quadrants:

    +---------------------------+---------------------------+
    | TOP-LEFT: Smooth gradient | TOP-RIGHT: Horizontal     |
    | (horizontal)              | high-freq sine wave       |
    | Low freq in both dirs     | High freq horizontal      |
    | -> HL, LH, HH all small   | -> HL band shows this     |
    +---------------------------+---------------------------+
    | BOTTOM-LEFT: Smooth       | BOTTOM-RIGHT: Vertical    |
    | gradient (vertical)       | high-freq sine wave       |
    | Low freq in both dirs     | High freq vertical        |
    | -> HL, LH, HH all small   | -> LH band shows this     |
    +---------------------------+---------------------------+

    Returns:
        numpy array of shape (size, size), dtype=int16, values 0-255
    """
    print_header("STEP 1: Create Test Image")

    print("""
    TEST IMAGE DESIGN
    =================

    We create a 64x64 test image with FOUR quadrants:

    +---------------------------+---------------------------+
    | TOP-LEFT:                 | TOP-RIGHT:                |
    | Horizontal gradient       | Horizontal sine wave      |
    | (low freq, smooth)        | (high freq, 4px period)   |
    | -> Small HL, LH, HH       | -> Shows in HL band       |
    +---------------------------+---------------------------+
    | BOTTOM-LEFT:              | BOTTOM-RIGHT:             |
    | Vertical gradient         | Vertical sine wave        |
    | (low freq, smooth)        | (high freq, 4px period)   |
    | -> Small HL, LH, HH       | -> Shows in LH band       |
    +---------------------------+---------------------------+

    This design ensures ALL wavelet bands have interesting content:
    - HL band: Detects horizontal high-frequency (top-right quadrant)
    - LH band: Detects vertical high-frequency (bottom-right quadrant)
    - HH band: Detects diagonal (corners where patterns meet)
    """)

    image = np.zeros((size, size), dtype=np.int16)
    half = size // 2

    # Uniform gray background
    image[:, :] = 128

    # TOP-LEFT quadrant: smooth horizontal gradient (low frequency)
    for x in range(half):
        image[:half, x] = int(x * 255 / (half - 1))

    # BOTTOM-LEFT quadrant: smooth vertical gradient (low frequency)
    for y in range(half):
        image[half + y, :half] = int(y * 255 / (half - 1))

    # TOP-RIGHT quadrant: horizontal high-frequency sine wave (centered)
    hf_start = half + size // 8  # Start 1/8 into right half
    hf_end = size - size // 8  # End 1/8 before right edge
    for x in range(hf_start, hf_end):
        phase = (x - hf_start) * 2 * np.pi / 4  # 4-pixel period
        value = 128 + int(64 * np.sin(phase))
        image[:half, x] = value

    # BOTTOM-RIGHT quadrant: vertical high-frequency sine wave (centered)
    vf_start = half + size // 8  # Start 1/8 into bottom half
    vf_end = size - size // 8  # End 1/8 before bottom edge
    for y in range(vf_start, vf_end):
        phase = (y - vf_start) * 2 * np.pi / 4  # 4-pixel period
        value = 128 + int(64 * np.sin(phase))
        image[y, half:] = value

    print("Image dimensions:", image.shape)
    print("Value range:", image.min(), "to", image.max())

    # Sample values from each quadrant
    print("\nSample from top-left (horizontal gradient), row 16, cols 0-7:")
    print("  ", image[16, :8].tolist())
    print("\nSample from top-right (horizontal sine), row 16, cols 40-47:")
    print("  ", image[16, 40:48].tolist())
    print("\nSample from bottom-right (vertical sine), rows 40-47, col 48:")
    print("  ", image[40:48, 48].tolist())

    # Visualize
    plt = ensure_matplotlib()
    if plt:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Full image
        im = axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Test Image (64x64)\n4 Quadrants with different patterns", fontsize=12)
        axes[0].set_xlabel("Column (x)")
        axes[0].set_ylabel("Row (y)")
        axes[0].axvline(x=31.5, color="red", linestyle="--", linewidth=1)
        axes[0].axhline(y=31.5, color="red", linestyle="--", linewidth=1)
        plt.colorbar(im, ax=axes[0], label="Pixel Value")

        # Horizontal profile (row 16 from top half)
        axes[1].plot(image[16, :], "b-", linewidth=1.5)
        axes[1].axvline(x=32, color="red", linestyle="--", label="Half boundary")
        axes[1].set_xlabel("Column (x)")
        axes[1].set_ylabel("Pixel Value")
        axes[1].set_title("Row 16 Profile (top half)\nGradient -> Sine wave")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Vertical profile (col 48 from right half)
        axes[2].plot(image[:, 48], "g-", linewidth=1.5)
        axes[2].axvline(x=32, color="red", linestyle="--", label="Half boundary")
        axes[2].set_xlabel("Row (y)")
        axes[2].set_ylabel("Pixel Value")
        axes[2].set_title("Column 48 Profile (right half)\nSine wave -> Sine wave")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step1_original_image.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step1_original_image.png'}")

    return image.astype(np.uint8)


# =============================================================================
# STEP 2: COLOR TRANSFORM (for RGB images)
# =============================================================================


def demonstrate_color_transform() -> None:
    """Explain the UYV color transform used by GFWX."""
    print_header("STEP 2: Color Transform")

    print("""
    UYV COLOR TRANSFORM
    ===================

    For RGB images, GFWX uses a reversible color transform similar to YUV
    that separates luminance (brightness) from chrominance (color).

    The UYV transform:
    +---------------------------------------------------------------------+
    | Forward (RGB -> UYV):                                               |
    |   R' = R - G                 (Red-Green difference, chroma)         |
    |   B' = B - G                 (Blue-Green difference, chroma)        |
    |   Y  = G + (R' + B') / 4     (Luminance approximation)              |
    |                                                                     |
    | Inverse (UYV -> RGB):                                               |
    |   G  = Y - (R' + B') / 4                                            |
    |   R  = R' + G                                                       |
    |   B  = B' + G                                                       |
    +---------------------------------------------------------------------+

    WHY USE A COLOR TRANSFORM?
    --------------------------
    1. Human vision is more sensitive to brightness than color
       -> We can quantize chroma channels more aggressively

    2. R, G, B channels are correlated (nearby pixels often similar)
       -> Differences (R-G, B-G) have smaller values, compress better

    3. The transform is reversible (lossless) with integer math
       -> Perfect reconstruction when quality=1024

    For our grayscale walkthrough, we skip this step.
    The wavelet transform works directly on the single channel.
    """)


# =============================================================================
# STEP 3: WAVELET TRANSFORM
# =============================================================================


def demonstrate_1d_wavelet(image: np.ndarray) -> None:
    """Show how 1D wavelet lifting works on a row."""
    print_header("STEP 3a: 1D Wavelet Transform (Lifting Scheme)")

    print("""
    THE LIFTING WAVELET TRANSFORM
    =============================

    GFWX uses a "lifting scheme" wavelet, which works in-place:
    no temporary buffers needed!

    Two filter types are available:
    - LINEAR (5/3): Integer-exact, for lossless compression
    - CUBIC (9/7): Better frequency separation, for lossy compression

    THE LINEAR (5/3) LIFTING STEPS:
    +---------------------------------------------------------------------+
    | 1. PREDICT (odd samples):                                           |
    |    odd[i] -= (even[i-1] + even[i+1]) / 2                            |
    |    The odd sample becomes the "prediction error" - how different    |
    |    it is from its neighbors' average.                               |
    |                                                                     |
    | 2. UPDATE (even samples):                                           |
    |    even[i] += (odd[i-1] + odd[i+1]) / 4                             |
    |    Adjusts even samples to preserve the average.                    |
    +---------------------------------------------------------------------+

    After lifting:
    - Even positions contain LOWPASS (approximation) coefficients
    - Odd positions contain HIGHPASS (detail) coefficients

    EXAMPLE: Processing the row [8, 10, 12, 14, 16, 18, 20, 22]
    (a smooth gradient - low frequency)

    Before:    8   10   12   14   16   18   20   22
              [e]  [o]  [e]  [o]  [e]  [o]  [e]  [o]  (even/odd positions)

    Predict:  10 -= (8+12)/2 = 10-10 = 0    (perfect prediction!)
              14 -= (12+16)/2 = 14-14 = 0
              18 -= (16+20)/2 = 18-18 = 0
              22 -= (20+20)/2 = 22-20 = 2    (boundary effect)

    After predict: 8, 0, 12, 0, 16, 0, 20, 2

    Update:   12 += (0+0)/4 = 12+0 = 12
              16 += (0+0)/4 = 16+0 = 16
              20 += (0+2)/4 = 20+0 = 20

    Final:    8, 0, 12, 0, 16, 0, 20, 2
              ^     ^      ^      ^        <- Even (lowpass): 8,12,16,20
                 ^     ^      ^     ^      <- Odd (highpass): 0,0,0,2

    The highpass coefficients are nearly zero because the gradient
    is smooth! This is why gradients compress so well.
    """)

    # Demonstrate on Row 16 (gradient + sine)
    row = image[16, :].copy().astype(np.int32)
    print("\nActual Row 16 from test image (first 16 values):")
    print(f"  {row[:16].tolist()}")

    # Manual 1D lifting demonstration
    row_copy = row[:16].copy()
    print("\nManual lifting on first 16 values:")

    # Predict step (odd positions)
    print("\n  PREDICT step (odd samples become differences):")
    for i in range(1, 15, 2):  # Skip boundary
        old = row_copy[i]
        left = row_copy[i - 1]
        right = row_copy[i + 1]
        row_copy[i] -= (left + right) // 2
        print(f"    pos[{i}]: {old} -= ({left}+{right})/2 = {row_copy[i]}")

    print(f"\n  After predict: {row_copy.tolist()}")

    # Update step (even positions)
    print("\n  UPDATE step (even samples adjusted):")
    for i in range(2, 14, 2):
        old = row_copy[i]
        left = row_copy[i - 1]
        right = row_copy[i + 1]
        row_copy[i] += (left + right) // 4
        print(f"    pos[{i}]: {old} += ({left}+{right})/4 = {row_copy[i]}")

    print(f"\n  Final: {row_copy.tolist()}")

    # Separate into lowpass/highpass
    lowpass = row_copy[::2]  # Even positions
    highpass = row_copy[1::2]  # Odd positions
    print(f"\n  Lowpass (even positions): {lowpass.tolist()}")
    print(f"  Highpass (odd positions): {highpass.tolist()}")

    # Stats
    near_zero = np.sum(np.abs(highpass) < 3)
    print(f"\n  Highpass values near zero (|x| < 3): {near_zero}/{len(highpass)}")

    plt = ensure_matplotlib()
    if plt:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Original row
        axes[0].plot(row[:32], "b-", linewidth=1.5, marker="o", markersize=3)
        axes[0].set_title("Original Row 16 (first 32 values)\nGradient region")
        axes[0].set_xlabel("Position")
        axes[0].set_ylabel("Value")
        axes[0].grid(True, alpha=0.3)

        # Lowpass
        axes[1].plot(lowpass, "g-", linewidth=1.5, marker="s", markersize=4)
        axes[1].set_title("Lowpass (approximation)\nSmooth version at half resolution")
        axes[1].set_xlabel("Position (half)")
        axes[1].set_ylabel("Value")
        axes[1].grid(True, alpha=0.3)

        # Highpass
        axes[2].plot(highpass, "r-", linewidth=1.5, marker="^", markersize=4)
        axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[2].set_title("Highpass (detail)\nSmall for smooth regions!")
        axes[2].set_xlabel("Position (half)")
        axes[2].set_ylabel("Value")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("1D Wavelet Transform: Splitting into Lowpass + Highpass", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step3a_1d_wavelet.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step3a_1d_wavelet.png'}")


def demonstrate_2d_wavelet(image: np.ndarray) -> tuple:
    """Show the 2D wavelet decomposition into 4 subbands."""
    print_header("STEP 3b: 2D Wavelet Transform")

    print("""
    2D WAVELET DECOMPOSITION
    ========================

    The 2D transform applies 1D wavelets in two directions:
    1. First, apply horizontal transform to each row
    2. Then, apply vertical transform to each column

    This creates FOUR subbands:

    +------------+------------+
    |            |            |
    |  LL (Low-  |  HL (Low   |
    |  Low) -    |  vert,     |
    |  Approx.   |  High      |
    |            |  horiz)    |
    +------------+------------+
    |            |            |
    |  LH (High  |  HH (High- |
    |  vert,     |  High) -   |
    |  Low       |  Diagonal  |
    |  horiz)    |  edges     |
    |            |            |
    +------------+------------+

    Band naming convention:
    - LL: Low vertical, Low horizontal  -> smooth approximation
    - HL: Low vertical, High horizontal -> vertical edges
    - LH: High vertical, Low horizontal -> horizontal edges
    - HH: High vertical, High horizontal -> diagonal/texture

    FOR OUR TEST IMAGE:
    - Top-left gradient: mostly in LL (smooth)
    - Top-right sine: large values in HL (horizontal frequency)
    - Bottom-right checkerboard: large values in HH (diagonal frequency)
    """)

    # Import and use our actual lifting code
    from pygfwx.core.header import Filter
    from pygfwx.core.lifting import _lift_horizontal, _lift_vertical

    # Create working copy
    result = image.astype(np.int32).copy()
    height, width = result.shape

    # Apply forward transform (ONE level only)
    # The lift() function does ALL levels automatically, so we call the internal
    # functions directly to do just one level of decomposition.
    _lift_horizontal(result, 0, 0, width, height, 1, Filter.LINEAR)
    _lift_vertical(result, 0, 0, width, height, 1, Filter.LINEAR)

    # Extract subbands (interleaved layout)
    # After lifting, even indices = lowpass, odd indices = highpass
    ll = result[::2, ::2]  # Even rows, even cols
    hl = result[::2, 1::2]  # Even rows, odd cols (horizontal high)
    lh = result[1::2, ::2]  # Odd rows, even cols (vertical high)
    hh = result[1::2, 1::2]  # Odd rows, odd cols (diagonal)

    print(f"\nInput image shape: {image.shape}")
    print(f"Each subband shape: {ll.shape} (half in each dimension)")

    print("\n--- LL Band (Approximation) ---")
    print(f"Values: min={ll.min()}, max={ll.max()}")
    print(f"First 4x4:\n{ll[:4, :4]}")

    print("\n--- HL Band (Horizontal edges) ---")
    print(f"Values: min={hl.min()}, max={hl.max()}")
    # Show region with sine wave (top-right of original = top-right of HL)
    print(f"Top-right 4x4 (from sine region):\n{hl[:4, -4:]}")

    print("\n--- LH Band (Vertical edges) ---")
    print(f"Values: min={lh.min()}, max={lh.max()}")

    print("\n--- HH Band (Diagonal details) ---")
    print(f"Values: min={hh.min()}, max={hh.max()}")
    # Show checkerboard region (bottom-right)
    print(f"Bottom-right 4x4 (from checkerboard):\n{hh[-4:, -4:]}")

    # Statistics
    print("\n" + "=" * 50)
    print("KEY INSIGHT: Why Wavelets Enable Compression")
    print("=" * 50)

    total_coeffs = ll.size + hl.size + lh.size + hh.size
    hl_near_zero = np.sum(np.abs(hl) < 5)
    lh_near_zero = np.sum(np.abs(lh) < 5)
    hh_near_zero = np.sum(np.abs(hh) < 5)
    high_freq_coeffs = hl.size + lh.size + hh.size
    near_zero = hl_near_zero + lh_near_zero + hh_near_zero

    print(f"Total coefficients: {total_coeffs}")
    print(f"High-frequency coefficients: {high_freq_coeffs}")
    print(f"Near zero (|value| < 5): {near_zero} ({100 * near_zero / high_freq_coeffs:.1f}%)")
    print("\n-> Most high-frequency content is small!")
    print("-> Small values compress extremely well with entropy coding.")

    plt = ensure_matplotlib()
    if plt:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title(f"Original Image {image.shape}")
        axes[0, 0].axis("off")

        # LL band (rescale for display)
        ll_display = (ll - ll.min()) / (ll.max() - ll.min() + 1e-8) * 255
        axes[0, 1].imshow(ll_display, cmap="gray")
        axes[0, 1].set_title(f"LL Band (Approximation)\nmin={ll.min()}, max={ll.max()}")
        axes[0, 1].axis("off")

        # HL band (symmetric colormap)
        vmax = max(abs(hl.min()), abs(hl.max()), 1)
        axes[0, 2].imshow(hl, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, 2].set_title("HL Band (Horizontal detail)\nSine wave visible top-right!")
        axes[0, 2].axis("off")

        # LH band
        vmax = max(abs(lh.min()), abs(lh.max()), 1)
        axes[1, 0].imshow(lh, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, 0].set_title("LH Band (Vertical detail)\nEdges at quadrant boundaries")
        axes[1, 0].axis("off")

        # HH band
        vmax = max(abs(hh.min()), abs(hh.max()), 1)
        axes[1, 1].imshow(hh, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, 1].set_title("HH Band (Diagonal detail)\nCheckerboard visible bottom-right!")
        axes[1, 1].axis("off")

        # Histogram of all high-freq
        all_hf = np.concatenate([hl.flatten(), lh.flatten(), hh.flatten()])
        axes[1, 2].hist(all_hf, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[1, 2].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[1, 2].set_title("High-Freq Coefficient Distribution\nMost values near zero!")
        axes[1, 2].set_xlabel("Coefficient Value")
        axes[1, 2].set_ylabel("Count")

        plt.suptitle("2D Wavelet Decomposition: Separating Frequencies", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step3b_2d_wavelet.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step3b_2d_wavelet.png'}")

    return ll, hl, lh, hh


def demonstrate_multilevel(image: np.ndarray) -> None:
    """Show multi-level wavelet pyramid."""
    print_header("STEP 3c: Multi-Level Wavelet Pyramid")

    print("""
    MULTI-LEVEL DECOMPOSITION
    =========================

    GFWX typically uses multiple levels of wavelet decomposition.
    After the first level, we recursively apply the transform to the LL band:

    Level 1:  Full image -> LL1, HL1, LH1, HH1
    Level 2:  LL1        -> LL2, HL2, LH2, HH2
    Level 3:  LL2        -> LL3, HL3, LH3, HH3

    Visually (for 64x64 input, 3 levels):

    +-------+-------+---------------+
    |LL3|HL3|       |               |
    |---+---|  HL2  |               |
    |LH3|HH3|       |     HL1       |
    +-------+-------+               |
    |       |       |               |
    |  LH2  |  HH2  |               |
    |       |       |               |
    +-------+-------+---------------+
    |               |               |
    |               |               |
    |     LH1       |     HH1       |
    |               |               |
    |               |               |
    +---------------+---------------+

    The LL band gets progressively smaller, while detail bands at each
    level capture different scales of features:
    - Level 1: Fine details (2-4 pixel features)
    - Level 2: Medium details (4-8 pixel features)
    - Level 3: Coarse details (8-16 pixel features)
    """)

    from pygfwx.core.header import Filter
    from pygfwx.core.lifting import _lift_horizontal, _lift_vertical

    # Apply multi-level transform MANUALLY, extracting bands AT EACH LEVEL
    # before proceeding to the next. This gives us clean LL bands at each level.
    result = image.astype(np.int32).copy()
    height, width = result.shape

    # Level 1: step=1 (operates on all pixels)
    _lift_horizontal(result, 0, 0, width, height, 1, Filter.LINEAR)
    _lift_vertical(result, 0, 0, width, height, 1, Filter.LINEAR)
    
    # Extract Level 1 bands BEFORE applying level 2
    ll1 = result[0::2, 0::2].copy()  # 32x32 - this is the clean LL1
    hl1 = result[0::2, 1::2].copy()  # 32x32
    lh1 = result[1::2, 0::2].copy()  # 32x32
    hh1 = result[1::2, 1::2].copy()  # 32x32
    
    # Level 2: step=2 (operates on LL1 positions)
    _lift_horizontal(result, 0, 0, width, height, 2, Filter.LINEAR)
    _lift_vertical(result, 0, 0, width, height, 2, Filter.LINEAR)
    
    # Extract Level 2 bands BEFORE applying level 3
    # LL2 is at positions [0::4, 0::4], etc.
    ll2 = result[0::4, 0::4].copy()  # 16x16 - clean LL2
    hl2 = result[0::4, 2::4].copy()  # 16x16
    lh2 = result[2::4, 0::4].copy()  # 16x16
    hh2 = result[2::4, 2::4].copy()  # 16x16
    
    # Level 3: step=4 (operates on LL2 positions)
    _lift_horizontal(result, 0, 0, width, height, 4, Filter.LINEAR)
    _lift_vertical(result, 0, 0, width, height, 4, Filter.LINEAR)
    
    # Extract Level 3 bands
    ll3 = result[0::8, 0::8].copy()  # 8x8
    hl3 = result[0::8, 4::8].copy()  # 8x8
    lh3 = result[4::8, 0::8].copy()  # 8x8
    hh3 = result[4::8, 4::8].copy()  # 8x8

    print(f"After 3-level decomposition on {image.shape} image:")
    print("  Level 1: LL 32x32, detail bands 32x32 each")
    print("  Level 2: LL 16x16, detail bands 16x16 each")
    print("  Level 3: LL 8x8, detail bands 8x8 each")

    levels_data = [
        (ll1, hl1, lh1, hh1),  # Level 1
        (ll2, hl2, lh2, hh2),  # Level 2
        (ll3, hl3, lh3, hh3),  # Level 3
    ]

    for i, (ll, hl, lh, hh) in enumerate(levels_data):
        level = i + 1
        total = hl.size + lh.size + hh.size
        near_zero = np.sum(np.abs(hl) < 3) + np.sum(np.abs(lh) < 3) + np.sum(np.abs(hh) < 3)
        print(f"\n  Level {level}: LL {ll.shape}, detail bands {hl.shape}")
        print(f"           High-freq near zero: {near_zero}/{total} ({100 * near_zero / total:.0f}%)")

    plt = ensure_matplotlib()
    if plt:
        # Create a 3x4 grid: rows are levels, columns are LL, HL, LH, HH
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        band_names = ["LL (Approximation)", "HL (Horizontal)", "LH (Vertical)", "HH (Diagonal)"]
        
        for level_idx, (ll, hl, lh, hh) in enumerate(levels_data):
            bands = [ll, hl, lh, hh]
            
            for col_idx, (band, name) in enumerate(zip(bands, band_names)):
                ax = axes[level_idx, col_idx]
                
                if col_idx == 0:  # LL band - grayscale, normalized
                    band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
                    ax.imshow(band_norm, cmap="gray")
                    ax.set_title(f"Level {level_idx + 1} {name}\n{band.shape[0]}x{band.shape[1]}")
                else:  # Detail bands - symmetric colormap around zero
                    vmax = max(abs(band.min()), abs(band.max()), 1)
                    ax.imshow(band, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    near_zero_pct = 100 * np.sum(np.abs(band) <= 2) / band.size
                    ax.set_title(f"Level {level_idx + 1} {name}\n{near_zero_pct:.0f}% near-zero")
                
                ax.axis("off")
        
        plt.suptitle("3-Level Wavelet Pyramid - All Bands\n"
                     "LL bands show progressively coarser approximations; detail bands capture edges at each scale",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step3c_multilevel.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step3c_multilevel.png'}")


# =============================================================================
# STEP 4: QUANTIZATION
# =============================================================================


def demonstrate_quantization(hl: np.ndarray) -> np.ndarray:
    """Explain and demonstrate quantization."""
    print_header("STEP 4: Quantization")

    print("""
    QUANTIZATION: Trading Precision for Compression
    ================================================

    Quantization reduces precision by dividing coefficients:

        quantized = coefficient // divisor

    Example with divisor = 8:
        - Input 100 -> 100 // 8 = 12
        - Input 7   -> 7 // 8 = 0   (small values vanish!)
        - Input -50 -> -50 // 8 = -6

    WHY QUANTIZATION HELPS COMPRESSION:
    +---------------------------------------------------------------------+
    | 1. Small coefficients become zero                                   |
    |    -> More runs of zeros -> better entropy coding                   |
    |                                                                     |
    | 2. Fewer unique values                                              |
    |    -> Shorter codes on average                                      |
    |                                                                     |
    | 3. Controlled quality loss                                          |
    |    -> Larger divisor = more loss = smaller file                     |
    +---------------------------------------------------------------------+

    GFWX QUALITY PARAMETER:
    - quality=1024: Lossless (no quantization)
    - quality=512: High quality (mild quantization)
    - quality=256: Medium quality (moderate quantization)
    - quality=64: Low quality (aggressive quantization)

    Divisor is typically: 1024 / quality * scale_factor
    """)

    # Demonstrate with actual HL band
    print_subheader("Example: Quantizing HL Band")

    original = hl.copy()
    divisor = 8

    print("\nHL band before quantization (first 4x4):")
    print(original[:4, :4])

    quantized = original // divisor

    print(f"\nHL band after quantization (divisor={divisor}):")
    print(quantized[:4, :4])

    # Statistics
    orig_zeros = np.sum(original == 0)
    quant_zeros = np.sum(quantized == 0)
    orig_unique = len(np.unique(original))
    quant_unique = len(np.unique(quantized))

    print("\nCompression Statistics:")
    print(f"  Original zeros: {orig_zeros}/{original.size} ({100 * orig_zeros / original.size:.1f}%)")
    print(f"  Quantized zeros: {quant_zeros}/{quantized.size} ({100 * quant_zeros / quantized.size:.1f}%)")
    print(f"  Original unique values: {orig_unique}")
    print(f"  Quantized unique values: {quant_unique}")
    print(f"\n  -> Quantization created {quant_zeros - orig_zeros} new zeros!")
    print(f"  -> Reduced from {orig_unique} to {quant_unique} unique values!")

    plt = ensure_matplotlib()
    if plt:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original
        vmax = max(abs(original.min()), abs(original.max()), 1)
        axes[0].imshow(original, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"Original HL Band\n{orig_unique} unique values")
        axes[0].axis("off")

        # Quantized
        vmax_q = max(abs(quantized.min()), abs(quantized.max()), 1)
        axes[1].imshow(quantized, cmap="RdBu_r", vmin=-vmax_q, vmax=vmax_q)
        axes[1].set_title(f"Quantized (/{divisor})\n{quant_unique} unique values")
        axes[1].axis("off")

        # Histograms
        axes[2].hist(original.flatten(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[2].axvline(x=0, color="red", linestyle="--")
        axes[2].set_title("Original Distribution")
        axes[2].set_xlabel("Coefficient Value")

        axes[3].hist(quantized.flatten(), bins=30, color="green", edgecolor="black", alpha=0.7)
        axes[3].axvline(x=0, color="red", linestyle="--")
        axes[3].set_title("Quantized Distribution\nMany more zeros!")
        axes[3].set_xlabel("Coefficient Value")

        plt.suptitle(f"Quantization Effect (divisor={divisor})", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step4_quantization.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step4_quantization.png'}")

    return quantized


# =============================================================================
# STEP 5: ENTROPY CODING (GOLOMB-RICE)
# =============================================================================


def demonstrate_golomb_rice(quantized: np.ndarray) -> None:
    """Explain Golomb-Rice entropy coding."""
    print_header("STEP 5: Golomb-Rice Entropy Coding")

    print("""
    GOLOMB-RICE CODING
    ==================

    Golomb-Rice is a variable-length code efficient for geometric distributions
    (lots of small values, few large values) - perfect for wavelet coefficients!

    The code uses a parameter 'k' (power-of-two divisor):

    +---------------------------------------------------------------------+
    | To encode value x with parameter k:                                 |
    |                                                                     |
    | 1. quotient = x >> k        (divide by 2^k)                         |
    | 2. remainder = x & ((1<<k)-1)  (modulo 2^k)                         |
    |                                                                     |
    | 3. Output quotient in UNARY: 'quotient' zeros followed by a 1       |
    | 4. Output remainder in BINARY: 'k' bits                             |
    +---------------------------------------------------------------------+

    EXAMPLE with k=2 (divisor=4):

        Value 5:
          quotient = 5 >> 2 = 1
          remainder = 5 & 3 = 1
          Encoding: 0 1 01 = "0101" (4 bits)
                    ^ ^ ^^
                    |   |  remainder (2 bits)
                    unary for quotient=1

        Value 0:
          quotient = 0, remainder = 0
          Encoding: 1 00 = "100" (3 bits)
                    ^ ^^
                    |  remainder
                    no zeros before 1

        Value 13:
          quotient = 13 >> 2 = 3
          remainder = 13 & 3 = 1
          Encoding: 000 1 01 = "0001001" (7 bits)

    WHY IT'S EFFICIENT:
    - Small values (common) -> short codes
    - Large values (rare) -> longer codes
    - Zero (very common in quantized coefficients) -> very short code

    GFWX also uses:
    - Run-length coding for sequences of zeros
    - Adaptive context to adjust 'k' parameter
    - Signed coding: magnitude + sign bit
    """)

    # Manual encoding example
    print_subheader("Step-by-Step Encoding Example")

    # Take some coefficients
    sample = quantized.flatten()[:20]
    print(f"\nFirst 20 quantized coefficients: {sample.tolist()}")

    print("\nEncoding each value with k=2 (divisor=4):")
    k = 2
    total_bits = 0

    for i, val in enumerate(sample[:10]):
        abs_val = abs(val)
        q = abs_val >> k
        r = abs_val & ((1 << k) - 1)

        # Unary for quotient
        unary = "0" * q + "1"
        # Binary for remainder
        binary = format(r, f"0{k}b")
        # Sign bit if non-zero
        sign = "" if val == 0 else ("1" if val > 0 else "0")

        code = unary + binary + sign
        total_bits += len(code)

        print(f"  [{i:2d}] {val:4d} -> q={q}, r={r}, code={code} ({len(code)} bits)")

    print(f"\nTotal for first 10: {total_bits} bits")
    print(f"Naive encoding: {10 * 16} bits (16-bit integers)")
    print(f"Compression: {10 * 16 / total_bits:.1f}x")

    # Show actual encoding using our library
    print_subheader("Full Band Encoding Statistics")

    from pygfwx.core.bitstream import BitWriter
    from pygfwx.core.golomb_rice import signed_encode

    # Encode entire band - allocate enough space (worst case: ~32 bits per coefficient)
    max_words = (quantized.size * 32 + 31) // 32
    writer = BitWriter(max_words)
    for val in quantized.flatten():
        signed_encode(4, int(val), writer)  # k=4 for typical encoding

    encoded_bytes = writer.get_data()
    original_bits = quantized.size * 16  # 16-bit coefficients
    compressed_bits = len(encoded_bytes) * 8

    print(f"\nFull HL band: {quantized.size} coefficients")
    print(f"  Original: {original_bits} bits ({original_bits // 8} bytes)")
    print(f"  Compressed: {compressed_bits} bits ({len(encoded_bytes)} bytes)")
    print(f"  Bits per coefficient: {compressed_bits / quantized.size:.2f}")
    print(f"  Compression ratio: {original_bits / compressed_bits:.1f}x")


# =============================================================================
# STEP 6: FULL ENCODE/DECODE DEMONSTRATION
# =============================================================================


def demonstrate_full_pipeline(image: np.ndarray) -> None:
    """Demonstrate full encode/decode using the pure Python codec."""
    print_header("STEP 6: Full Encode/Decode Pipeline")

    from pygfwx import encode, decode, QUALITY_MAX

    print("""
    COMPLETE COMPRESSION PIPELINE
    =============================

    Now we put it all together using the pure Python GFWX codec:

        Original Image
             |
             v
        [Color Transform] (RGB -> UYV for color images)
             |
             v
        [Wavelet Transform] (Multi-level decomposition)
             |
             v
        [Quantization] (Reduce precision based on quality)
             |
             v
        [Entropy Coding] (Golomb-Rice with adaptive context)
             |
             v
        Compressed Bitstream

    Decoding reverses each step:

        Compressed Bitstream
             |
             v
        [Entropy Decode] (Golomb-Rice)
             |
             v
        [Dequantize] (Multiply by quantizer)
             |
             v
        [Inverse Wavelet] (Reconstruct from subbands)
             |
             v
        [Inverse Color] (UYV -> RGB)
             |
             v
        Reconstructed Image
    """)

    # Use the same test image from Step 1
    print(f"\nUsing test image from Step 1: {image.shape}, {image.dtype}")
    print(f"Raw size: {image.nbytes} bytes")

    # Encode at different quality levels
    print("\nCompression Results:")
    print("-" * 60)
    qualities = [64, 256, 512, QUALITY_MAX]

    results = []
    for quality in qualities:
        compressed = encode(image, quality=quality)
        decoded = decode(compressed)

        # Error metrics
        if np.array_equal(image, decoded):
            psnr = float("inf")
            status = "LOSSLESS"
        else:
            mse = np.mean((image.astype(float) - decoded.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
            status = f"PSNR={psnr:.1f}dB"

        ratio = image.nbytes / len(compressed)
        results.append((quality, len(compressed), ratio, psnr, decoded))

        print(f"  Quality {quality:4d}: {len(compressed):5d} bytes, {ratio:5.1f}x compression, {status}")

    plt = ensure_matplotlib()
    if plt:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title(f"Original\n{image.nbytes} bytes")
        axes[0, 0].axis("off")

        # Decoded at different qualities
        for i, (quality, size, ratio, psnr, decoded) in enumerate(results[:4]):
            row = (i + 1) // 3
            col = (i + 1) % 3
            axes[row, col].imshow(decoded, cmap="gray", vmin=0, vmax=255)
            if psnr == float("inf"):
                title = f"Quality {quality} (Lossless)\n{size} bytes ({ratio:.1f}x)"
            else:
                title = f"Quality {quality}\n{size} bytes ({ratio:.1f}x)\nPSNR={psnr:.1f}dB"
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

        # Quality vs Size plot
        if len(results) > 3:
            axes[1, 2].plot([r[0] for r in results], [r[1] for r in results], "bo-", linewidth=2, markersize=8)
            axes[1, 2].set_xlabel("Quality Parameter")
            axes[1, 2].set_ylabel("Compressed Size (bytes)")
            axes[1, 2].set_title("Quality vs File Size")
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_xlim(0, 1100)

        plt.suptitle("GFWX Compression at Different Quality Levels", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "step6_full_pipeline.png", dpi=150)
        plt.close()
        print(f"\n[OK] Saved: {OUTPUT_DIR / 'step6_full_pipeline.png'}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run the complete walkthrough."""
    print("\n" + "=" * 70)
    print("       GFWX COMPRESSION PIPELINE WALKTHROUGH")
    print("       An Educational Journey Through Wavelet Compression")
    print("=" * 70)

    # Setup output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Step 1: Create test image
    image = create_test_image()

    # Step 2: Color transform explanation
    demonstrate_color_transform()

    # Step 3: Wavelet transform
    demonstrate_1d_wavelet(image)
    ll, hl, lh, hh = demonstrate_2d_wavelet(image)
    demonstrate_multilevel(image)

    # Step 4: Quantization
    quantized_hl = demonstrate_quantization(hl)

    # Step 5: Entropy coding
    demonstrate_golomb_rice(quantized_hl)

    # Step 6: Full pipeline (using same image from Step 1)
    demonstrate_full_pipeline(image)

    # Summary
    print_header("SUMMARY")
    print("""
    The GFWX compression pipeline achieves high compression through:

    1. COLOR TRANSFORM: Separates brightness from color (for RGB)
       -> Human vision tolerates more chroma loss

    2. WAVELET TRANSFORM: Decomposes image into frequency bands
       -> Most energy in low frequencies (LL band)
       -> High-frequency bands mostly contain small values

    3. QUANTIZATION: Reduces precision of coefficients
       -> Creates many zeros in high-frequency bands
       -> Controlled quality/size tradeoff

    4. ENTROPY CODING: Golomb-Rice with run-length encoding
       -> Short codes for common values (zeros, small numbers)
       -> Long codes for rare large values
       -> Adaptive context improves predictions

    The result: 10-100x compression with minimal visible quality loss!

    Generated visualizations:
    """)

    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    - {f.name}")

    print(f"\n    View these in: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("       WALKTHROUGH COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
