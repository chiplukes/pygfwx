"""
Tests for pygfwx.utils.image_io — load_image, save_image, get_bit_depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygfwx.utils.image_io import get_bit_depth, load_image, save_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gray8(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_rgb8(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_rgba8(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


def _make_gray16(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 65536, (h, w), dtype=np.uint16)


def _make_rgb16(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)


# ---------------------------------------------------------------------------
# get_bit_depth
# ---------------------------------------------------------------------------


def test_get_bit_depth_uint8():
    assert get_bit_depth(np.zeros((4, 4), dtype=np.uint8)) == 8


def test_get_bit_depth_uint16():
    assert get_bit_depth(np.zeros((4, 4), dtype=np.uint16)) == 16


def test_get_bit_depth_invalid():
    with pytest.raises(ValueError, match="Cannot determine bit depth"):
        get_bit_depth(np.zeros((4, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# save_image / load_image round-trips — 8-bit PNG
# ---------------------------------------------------------------------------


def test_roundtrip_gray8_png(tmp_path):
    img = _make_gray8()
    path = tmp_path / "gray8.png"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def test_roundtrip_rgb8_png(tmp_path):
    img = _make_rgb8()
    path = tmp_path / "rgb8.png"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def test_roundtrip_rgba8_png(tmp_path):
    img = _make_rgba8()
    path = tmp_path / "rgba8.png"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


# ---------------------------------------------------------------------------
# save_image / load_image round-trips — 8-bit JPEG (lossy, shape only)
# ---------------------------------------------------------------------------


def test_roundtrip_gray8_jpeg(tmp_path):
    img = _make_gray8()
    path = tmp_path / "gray8.jpg"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape


def test_roundtrip_rgb8_jpeg(tmp_path):
    img = _make_rgb8()
    path = tmp_path / "rgb8.jpg"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape


def test_rgba8_jpeg_strips_alpha(tmp_path):
    """RGBA saved as JPEG should silently drop the alpha channel."""
    img = _make_rgba8()
    path = tmp_path / "rgba8.jpg"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.shape == (*img.shape[:2], 3)


# ---------------------------------------------------------------------------
# save_image / load_image round-trips — 8-bit TIFF
# ---------------------------------------------------------------------------


def test_roundtrip_gray8_tiff(tmp_path):
    img = _make_gray8()
    path = tmp_path / "gray8.tiff"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def test_roundtrip_rgb8_tiff(tmp_path):
    img = _make_rgb8()
    path = tmp_path / "rgb8.tiff"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


# ---------------------------------------------------------------------------
# save_image / load_image round-trips — 16-bit PNG
# ---------------------------------------------------------------------------


def test_roundtrip_gray16_png(tmp_path):
    img = _make_gray16()
    path = tmp_path / "gray16.png"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint16
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


# ---------------------------------------------------------------------------
# save_image / load_image round-trips — 16-bit TIFF
# ---------------------------------------------------------------------------


def test_roundtrip_gray16_tiff(tmp_path):
    img = _make_gray16()
    path = tmp_path / "gray16.tiff"
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint16
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def test_roundtrip_rgb16_tiff(tmp_path):
    """16-bit multi-channel TIFF save is not supported by Pillow; expect ValueError."""
    img = _make_rgb16()
    path = tmp_path / "rgb16.tiff"
    with pytest.raises(ValueError, match="16-bit multi-channel"):
        save_image(img, path)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.png")


def test_save_unsupported_dtype():
    img = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        save_image(img, "/tmp/bad.png")


def test_save_16bit_jpeg_raises():
    img = _make_gray16()
    with pytest.raises(ValueError, match="16-bit images require PNG or TIFF"):
        save_image(img, "/tmp/bad.jpg")


def test_save_unsupported_channels():
    img = np.zeros((4, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported channel count"):
        save_image(img, "/tmp/bad.png")


# ---------------------------------------------------------------------------
# Integration with pygfwx encode/decode
# ---------------------------------------------------------------------------


def test_load_save_roundtrip_with_codec(tmp_path):
    """Verify load/save works with the encode/decode API."""
    import pygfwx

    # 8-bit RGB lossless round-trip via file
    original = _make_rgb8()
    png_path = tmp_path / "test.png"
    save_image(original, png_path)

    loaded = load_image(png_path)
    compressed = pygfwx.encode(loaded)
    decoded = pygfwx.decode(compressed)

    assert np.array_equal(decoded, original)
