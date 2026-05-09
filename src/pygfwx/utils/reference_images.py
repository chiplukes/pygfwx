"""
Reference image generator for PyGFWX.

This module provides a centralized source for test images used throughout the project.
All tests, examples, and validation code should use these generators to maintain consistency.

Primary Reference Image:
    The main reference image combines high and low frequency content in both
    horizontal and vertical directions, with each channel being distinct.
    This single image is suitable for most testing scenarios.
"""

import numpy as np


def create_reference_image(size: int = 64, channels: int = 1, bit_depth: int = 8) -> np.ndarray:  # cm:a7b8c9d — create_reference_image(): canonical test image (4-quadrant: smooth/stripes/checker/mixed)
    """Create the primary reference image for testing.

    This image is designed to exercise all aspects of wavelet compression:
    - Low frequency (smooth gradients) in both H and V directions
    - High frequency (edges, patterns) in both H and V directions
    - Each channel has distinct content to catch channel-mixing bugs
    - Reproducible (deterministic generation)

    The image is divided into quadrants:
    - Top-left: Low frequency (smooth gradient)
    - Top-right: High frequency horizontal (vertical stripes)
    - Bottom-left: High frequency vertical (horizontal stripes)
    - Bottom-right: Mixed high frequency (checkerboard + diagonal)

    For multi-channel images, each channel is phase-shifted and scaled differently.

    Args:
        size: Image dimension (creates size x size image). Should be power of 2.
        channels: Number of channels (1=mono, 3=RGB, 4=RGBA)
        bit_depth: Bits per sample (8 or 16)

    Returns:
        numpy array of shape (size, size) for mono or (size, size, channels) for multi-channel
    """
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    max_val = (1 << bit_depth) - 1
    half = size // 2

    def make_channel(channel_idx: int) -> np.ndarray:
        """Generate a single channel with channel-specific variations."""
        img = np.zeros((size, size), dtype=np.float64)

        # Phase and scale offsets per channel to make each distinct
        phase = channel_idx * 0.33  # Phase offset (0, 0.33, 0.66, 0.99)
        scale = 1.0 - channel_idx * 0.1  # Slight scale variation

        # Quadrant 1 (top-left): Smooth gradient (low frequency)
        # Diagonal gradient from corner
        for y in range(half):
            for x in range(half):
                # Smooth diagonal gradient with channel-specific direction
                if channel_idx % 2 == 0:
                    img[y, x] = ((x + y) / (half * 2)) * scale
                else:
                    img[y, x] = ((half - 1 - x + y) / (half * 2)) * scale

        # Quadrant 2 (top-right): Vertical stripes (high H frequency)
        stripe_width = max(2, size // 16)
        for y in range(half):
            for x in range(half, size):
                stripe_idx = ((x - half) // stripe_width + channel_idx) % 2
                img[y, x] = stripe_idx * scale

        # Quadrant 3 (bottom-left): Horizontal stripes (high V frequency)
        for y in range(half, size):
            for x in range(half):
                stripe_idx = ((y - half) // stripe_width + channel_idx) % 2
                img[y, x] = stripe_idx * scale

        # Quadrant 4 (bottom-right): Checkerboard + smooth overlay (mixed frequencies)
        checker_size = max(2, size // 32)
        for y in range(half, size):
            for x in range(half, size):
                # Checkerboard base
                cx = (x - half) // checker_size
                cy = (y - half) // checker_size
                checker = ((cx + cy + channel_idx) % 2) * 0.5

                # Add smooth diagonal overlay
                smooth = ((x - half) + (y - half)) / (half * 2) * 0.5

                img[y, x] = (checker + smooth * phase) * scale

        # Normalize to dtype range and apply channel-specific offset
        img = img * max_val * 0.8 + max_val * 0.1  # Keep away from extremes
        img = np.clip(img, 0, max_val)

        return img.astype(dtype)

    if channels == 1:
        return make_channel(0)
    else:
        channel_arrays = [make_channel(i) for i in range(channels)]
        return np.stack(channel_arrays, axis=-1)


def create_uniform_image(size: int = 64, value: int = 128, channels: int = 1, bit_depth: int = 8) -> np.ndarray:  # cm:e0f1a2b — create_uniform_image(): constant-value image (DC and quantization edge cases)
    """Create a uniform (constant value) image.

    Useful for testing DC handling and quantization edge cases.

    Args:
        size: Image dimension
        value: Pixel value (will be scaled for 16-bit)
        channels: Number of channels
        bit_depth: Bits per sample

    Returns:
        Uniform image array
    """
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    if bit_depth == 16:
        value = value * 257  # Scale 8-bit value to 16-bit

    if channels == 1:
        return np.full((size, size), value, dtype=dtype)
    else:
        return np.full((size, size, channels), value, dtype=dtype)


def create_impulse_image(size: int = 64, channels: int = 1, bit_depth: int = 8) -> np.ndarray:  # cm:c3d4e5c — create_impulse_image(): single bright pixel (impulse response / boundary tests)
    """Create an image with a single bright pixel at center.

    Useful for testing impulse response and boundary handling.

    Args:
        size: Image dimension
        channels: Number of channels
        bit_depth: Bits per sample

    Returns:
        Image with single bright pixel at center
    """
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    max_val = (1 << bit_depth) - 1
    center = size // 2

    if channels == 1:
        img = np.zeros((size, size), dtype=dtype)
        img[center, center] = max_val
    else:
        img = np.zeros((size, size, channels), dtype=dtype)
        img[center, center, :] = max_val

    return img


# =============================================================================
# Convenience accessors for common configurations
# =============================================================================


def get_reference_mono(size: int = 64) -> np.ndarray:
    """Get the primary reference image as 8-bit mono."""
    return create_reference_image(size=size, channels=1, bit_depth=8)


def get_reference_rgb(size: int = 64) -> np.ndarray:
    """Get the primary reference image as 8-bit RGB."""
    return create_reference_image(size=size, channels=3, bit_depth=8)


def get_reference_rgba(size: int = 64) -> np.ndarray:
    """Get the primary reference image as 8-bit RGBA."""
    return create_reference_image(size=size, channels=4, bit_depth=8)


def get_reference_mono_16bit(size: int = 64) -> np.ndarray:
    """Get the primary reference image as 16-bit mono."""
    return create_reference_image(size=size, channels=1, bit_depth=16)


def get_reference_rgb_16bit(size: int = 64) -> np.ndarray:
    """Get the primary reference image as 16-bit RGB."""
    return create_reference_image(size=size, channels=3, bit_depth=16)
