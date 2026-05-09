"""
Wavelet coefficient and image visualizations for GFWX educational purposes.

This module provides matplotlib-based visualizations to help understand:
- The wavelet decomposition produced by GFWX lifting
- Coefficient subband structure at each resolution level
- Encoded GFWX image data decoded at various progressive levels

All visualization functions are optional and require matplotlib.  They are
intended for Jupyter notebooks or interactive exploration, not production use.

Usage::

    import numpy as np
    from pygfwx.debug.visualize import plot_subbands, plot_wavelet_decomposition

    # Visualize a 2-level wavelet decomposition of a grayscale image
    image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    plot_wavelet_decomposition(image, levels=2)

    # Visualize GFWX compressed file decoded at progressive levels
    from pygfwx.debug.visualize import plot_progressive_decode
    plot_progressive_decode(Path("image.gfwx"))
"""

from __future__ import annotations

import numpy as np


def _require_matplotlib():
    """Raise a clear ImportError if matplotlib is not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from e


# ── Wavelet coefficient visualisation ────────────────────────────────────────


def plot_subbands(
    coefficients: np.ndarray,
    *,
    levels: int = 1,
    title: str = "Wavelet subbands",
    log_scale: bool = True,
    channel: int = 0,
) -> None:
    """
    Display wavelet coefficient subbands for a single channel.

    Renders the standard wavelet tiling (LL top-left, LH/HL/HH around it)
    for the given number of decomposition levels.

    Args:
        coefficients: Coefficient array, shape (H, W) or (C, H, W) or (H, W, C).
        levels: Number of decomposition levels to annotate (default 1).
        title: Plot title.
        log_scale: If True, display log(1 + |coeff|) for better visibility of
            high-frequency detail (default True).
        channel: Which channel to display for multi-channel arrays (default 0).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require_matplotlib()
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # Normalise to (H, W)
    arr = np.asarray(coefficients, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] <= 4:  # (C, H, W)
            arr = arr[channel]
        else:  # (H, W, C)
            arr = arr[:, :, channel]

    display = np.log1p(np.abs(arr)) if log_scale else np.abs(arr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display, cmap="gray", interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.axis("off")

    # Annotate subband boundaries
    h, w = arr.shape
    for level in range(1, levels + 1):
        lh = h >> level
        lw = w >> level
        # LL boundary
        rect = patches.Rectangle((0, 0), lw, lh, linewidth=1, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect)
        # Label subbands at this level
        for label, cx, cy in [
            ("LL", lw / 2, lh / 2),
            ("LH", lw * 1.5, lh / 2),
            ("HL", lw / 2, lh * 1.5),
            ("HH", lw * 1.5, lh * 1.5),
        ]:
            ax.text(cx, cy, label, color="yellow", ha="center", va="center", fontsize=8)

    scale_note = "log(1+|c|)" if log_scale else "|c|"
    ax.set_xlabel(scale_note)
    plt.tight_layout()
    plt.show()


def plot_wavelet_decomposition(
    image: np.ndarray,
    *,
    levels: int = 3,
    filter_type: str = "linear",
    channel: int = 0,
) -> None:
    """
    Apply the GFWX lifting transform to an image and plot the coefficients.

    This is the primary educational visualization: it shows what happens at
    each stage of the wavelet decomposition.

    Args:
        image: Input image as uint8 or uint16 numpy array.
            - Shape (H, W) for grayscale.
            - Shape (H, W, C) for multi-channel; `channel` selects which to show.
        levels: Number of decomposition levels to apply (default 3).
        filter_type: "linear" (5/3) or "cubic" (9/7) (default "linear").
        channel: Which channel to visualise for multi-channel images (default 0).

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If `filter_type` is not "linear" or "cubic".
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    from pygfwx.core.header import Filter
    from pygfwx.core.lifting import lift

    if filter_type == "linear":
        ftype = Filter.LINEAR
    elif filter_type == "cubic":
        ftype = Filter.CUBIC
    else:
        raise ValueError(f"filter_type must be 'linear' or 'cubic', got {filter_type!r}")

    # Extract single channel
    arr = np.asarray(image, dtype=np.int32)
    if arr.ndim == 3:
        arr = arr[:, :, channel]

    h, w = arr.shape
    coeffs = arr.copy()

    # Apply multi-level wavelet decomposition
    for lvl in range(levels):
        lh = h >> lvl
        lw = w >> lvl
        subband = coeffs[:lh, :lw].copy()
        lift(subband, ftype)
        coeffs[:lh, :lw] = subband

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Wavelet decomposition ({filter_type}, {levels} levels, ch={channel})")

    axes[0].imshow(arr, cmap="gray", interpolation="nearest")
    axes[0].set_title("Original")
    axes[0].axis("off")

    display = np.log1p(np.abs(coeffs.astype(np.float64)))
    axes[1].imshow(display, cmap="gray", interpolation="nearest")
    axes[1].set_title(f"Coefficients  (log scale, {levels} levels)")
    axes[1].axis("off")

    # Draw subband grid lines
    ax = axes[1]
    for lvl in range(1, levels + 1):
        lh = h >> lvl
        lw = w >> lvl
        ax.axhline(lh - 0.5, color="cyan", linewidth=0.7)
        ax.axvline(lw - 0.5, color="cyan", linewidth=0.7)

    plt.tight_layout()
    plt.show()


# ── Progressive decode visualisation ─────────────────────────────────────────


def plot_progressive_decode(
    source: bytes | str,
    *,
    steps: int = 4,
) -> None:
    """
    Decode a GFWX stream at multiple quality levels and plot the results.

    This visualises how GFWX's progressive decoding works: it shows the image
    as decoded with increasing amounts of data (from a small prefix to the full
    file).

    Args:
        source: Raw GFWX bytes, or a path (str or Path) to a .gfwx file.
        steps: Number of progressive steps to visualise (default 4).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    from pathlib import Path

    _require_matplotlib()
    import matplotlib.pyplot as plt

    from pygfwx import decode, get_header

    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    else:
        data = bytes(source)

    header = get_header(data)
    total = len(data)

    # Sample at `steps` sizes from 20% of the file to 100%
    sizes = [max(64, int(total * (i + 1) / steps)) for i in range(steps)]
    sizes[-1] = total  # Always include the full file

    ncols = len(sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"Progressive decode — {header.sizex}×{header.sizey} {header.channels}ch {header.bit_depth}-bit"
    )

    for ax, size in zip(axes, sizes, strict=False):
        try:
            img = decode(data[:size])
            pct = 100 * size // total
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=2**header.bit_depth - 1)
            else:
                # Clip to uint8 range for display
                display = np.clip(img, 0, 255).astype(np.uint8)
                ax.imshow(display, interpolation="nearest")
            ax.set_title(f"{pct}%  ({size:,}B)")
        except Exception as exc:
            ax.set_title(f"{100 * size // total}% (error)")
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes, ha="center", va="center", fontsize=6)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ── Histogram ────────────────────────────────────────────────────────────────


def plot_channel_histograms(
    image: np.ndarray,
    *,
    title: str = "Channel histograms",
    bins: int = 64,
) -> None:
    """
    Plot pixel value histograms for each channel.

    Args:
        image: Image array, shape (H, W) or (H, W, C).
        title: Plot title.
        bins: Number of histogram bins (default 64).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    arr = np.asarray(image)
    if arr.ndim == 2:
        channels = [arr]
        labels = ["gray"]
    else:
        channels = [arr[:, :, c] for c in range(arr.shape[2])]
        labels = ["R", "G", "B", "A"][: len(channels)] if len(channels) <= 4 else [f"ch{c}" for c in range(len(channels))]

    colors = ["red", "green", "blue", "gray", "purple", "orange"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for ch, label, color in zip(channels, labels, colors, strict=False):
        ax.hist(ch.ravel(), bins=bins, alpha=0.5, label=label, color=color, density=True)
    ax.set_title(title)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()
