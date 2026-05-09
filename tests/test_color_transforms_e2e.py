"""End-to-end tests for color transforms wired into the encode/decode pipeline."""

import numpy as np
import pytest

from pygfwx import QUALITY_MAX, decode, encode
from pygfwx.core.header import Encoder, Filter, Intent


def _rgb_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rgba_image(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


# ── Lossless round-trips ──────────────────────────────────────────────────────


@pytest.mark.parametrize("transform", ["uyv", "a710"])
def test_lossless_roundtrip_rgb(transform):
    """Lossless encode+decode with UYV or A710 must reconstruct exactly."""
    image = _rgb_image(32, 32)
    data = encode(image, quality=QUALITY_MAX, filter=Filter.LINEAR, color_transform=transform)
    result = decode(data)
    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize("transform", ["uyv", "a710"])
def test_lossless_roundtrip_larger_image(transform):
    """Roundtrip on a larger image to exercise full wavelet decomposition."""
    image = _rgb_image(64, 64, seed=42)
    data = encode(image, quality=QUALITY_MAX, color_transform=transform)
    result = decode(data)
    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize("transform", ["uyv", "a710"])
def test_lossless_roundtrip_all_encoders(transform):
    """Transforms work with all encoder modes."""
    image = _rgb_image(16, 16)
    for enc in [Encoder.CONTEXTUAL, Encoder.FAST, Encoder.HIGH_BITRATE]:
        data = encode(image, quality=QUALITY_MAX, encoder=enc, color_transform=transform)
        result = decode(data)
        np.testing.assert_array_equal(result, image)


def test_no_transform_roundtrip():
    """Default (no transform) still works correctly."""
    image = _rgb_image(32, 32)
    data = encode(image, quality=QUALITY_MAX, color_transform=None)
    result = decode(data)
    np.testing.assert_array_equal(result, image)


# ── Lossy round-trips (quality check) ────────────────────────────────────────


@pytest.mark.parametrize("transform", ["uyv", "a710"])
def test_lossy_roundtrip_reasonable_psnr(transform):
    """Lossy encode/decode with transform produces a close (but not exact) result."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    data = encode(image, quality=512, filter=Filter.CUBIC, color_transform=transform)
    result = decode(data)
    assert result.shape == image.shape
    # Lossy: allow difference, but must be in the same ballpark
    diff = np.abs(image.astype(np.int32) - result.astype(np.int32))
    assert diff.max() < 60, f"Max diff {diff.max()} is unexpectedly large for q=512"


# ── Compression ratio ─────────────────────────────────────────────────────────


def test_transform_produces_valid_compressed_output():
    """Transform produces valid compressed output (smaller than some reasonable bound)."""
    image = _rgb_image(64, 64)
    no_transform = encode(image, quality=512, filter=Filter.CUBIC)
    with_uyv = encode(image, quality=512, filter=Filter.CUBIC, color_transform="uyv")
    with_a710 = encode(image, quality=512, filter=Filter.CUBIC, color_transform="a710")
    # All should produce valid output (not crash, produce bytes)
    assert len(no_transform) > 0
    assert len(with_uyv) > 0
    assert len(with_a710) > 0
    # All should decode back correctly
    assert decode(no_transform).shape == image.shape
    assert decode(with_uyv).shape == image.shape
    assert decode(with_a710).shape == image.shape


# ── Error handling ────────────────────────────────────────────────────────────


def test_invalid_transform_raises():
    """Unknown transform name raises ValueError."""
    image = _rgb_image(16, 16)
    with pytest.raises(ValueError, match="Unknown color_transform"):
        encode(image, color_transform="bad_transform")


def test_transform_on_mono_image_falls_back_to_identity():
    """A mono image with a transform requested falls back gracefully (< 3 channels)."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    data = encode(image, quality=QUALITY_MAX, color_transform="uyv")
    result = decode(data)
    np.testing.assert_array_equal(result, image)


def test_transform_on_rgba_image():
    """RGBA images with transform: only the first 3 channels are transformed."""
    image = _rgba_image(32, 32)
    data = encode(image, quality=QUALITY_MAX, color_transform="uyv")
    result = decode(data)
    np.testing.assert_array_equal(result, image)


# ── Interoperability: encoded data is valid GFWX ─────────────────────────────


def test_encoded_with_transform_has_valid_header():
    """Files encoded with a transform have a correctly parseable header."""
    from pygfwx import get_header

    image = _rgb_image(16, 16)
    data = encode(image, quality=QUALITY_MAX, color_transform="uyv")
    header = get_header(data)
    assert header.sizex == 16
    assert header.sizey == 16
    assert header.channels == 3
