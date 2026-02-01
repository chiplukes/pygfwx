"""
Tests for encode -> decode roundtrip using the high-level codec API.

These tests verify that images can be encoded and decoded correctly
using the pure Python implementation.
"""

import numpy as np
import pytest
from pygfwx import encode, decode, Filter, Encoder, QUALITY_MAX
from pygfwx.core.codec import decode_full, get_header


class TestLosslessRoundtrip:
    """Test lossless (quality=QUALITY_MAX) encode/decode roundtrip."""

    def test_small_mono_image(self):
        """2x2 grayscale image roundtrip."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_gradient_mono_image(self):
        """8x8 gradient grayscale image roundtrip."""
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_random_mono_image(self):
        """16x16 random grayscale image roundtrip."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_uniform_mono_image(self):
        """Large uniform grayscale image should compress very well."""
        image = np.full((64, 64), 128, dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)
        # Uniform image should compress to much smaller than original
        assert len(encoded) < image.size // 4

    def test_rgb_image(self):
        """8x8 RGB image roundtrip."""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_rgba_image(self):
        """4x4 RGBA image roundtrip."""
        rng = np.random.default_rng(456)
        image = rng.integers(0, 256, size=(4, 4, 4), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_larger_rgb_image(self):
        """32x32 RGB image roundtrip."""
        rng = np.random.default_rng(789)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_non_power_of_two_dimensions(self):
        """Non-power-of-two image dimensions."""
        rng = np.random.default_rng(111)
        image = rng.integers(0, 256, size=(13, 17), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_non_power_of_two_rgb(self):
        """Non-power-of-two RGB image."""
        rng = np.random.default_rng(222)
        image = rng.integers(0, 256, size=(11, 19, 3), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)


class TestFilters:
    """Test different wavelet filter options."""

    @pytest.mark.parametrize("filter_type", [Filter.LINEAR, Filter.CUBIC])
    def test_filter_lossless_mono(self, filter_type):
        """Both filters should produce lossless results at quality=QUALITY_MAX."""
        rng = np.random.default_rng(333)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX, filter=filter_type)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    @pytest.mark.parametrize("filter_type", [Filter.LINEAR, Filter.CUBIC])
    def test_filter_lossless_rgb(self, filter_type):
        """Both filters should produce lossless results for RGB."""
        rng = np.random.default_rng(444)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX, filter=filter_type)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)


class TestEncoderModes:
    """Test different encoder modes."""

    @pytest.mark.parametrize("encoder_mode", [Encoder.FAST, Encoder.CONTEXTUAL])
    def test_encoder_mode_lossless(self, encoder_mode):
        """Both encoder modes should produce lossless results at QUALITY_MAX."""
        rng = np.random.default_rng(555)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX, encoder=encoder_mode)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)


class TestLossyRoundtrip:
    """Test lossy encode/decode roundtrip."""

    def test_lossy_preserves_dimensions(self):
        """Lossy encoding should preserve image dimensions."""
        rng = np.random.default_rng(666)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        encoded = encode(image, quality=512)
        decoded = decode(encoded)
        assert decoded.shape == image.shape

    def test_lossy_smaller_than_original(self):
        """Lossy encoding should produce smaller output."""
        rng = np.random.default_rng(777)
        image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        lossless = encode(image, quality=QUALITY_MAX)
        lossy = encode(image, quality=256)
        # Lossy should usually be smaller for random data
        assert len(lossy) <= len(lossless)

    def test_lossy_quality_affects_size(self):
        """Lower quality should generally produce smaller files."""
        rng = np.random.default_rng(888)
        image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        high_q = encode(image, quality=512)
        low_q = encode(image, quality=128)
        # Lower quality should be smaller or equal
        assert len(low_q) <= len(high_q)


class TestDecodeMetadata:
    """Test that decode returns correct metadata."""

    def test_decode_returns_header_info(self):
        """get_header should return header information."""
        image = np.zeros((16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX, filter=Filter.LINEAR)
        header = get_header(encoded)
        
        assert header is not None
        assert header.sizex == 16
        assert header.sizey == 16
        assert header.layers == 1

    def test_decode_rgb_header(self):
        """RGB images should report 3 channels."""
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        header = get_header(encoded)
        
        assert header.channels == 3

    def test_decode_full_returns_result(self):
        """decode_full should return DecodeResult with image and header."""
        image = np.zeros((16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        result = decode_full(encoded)
        
        assert result.image is not None
        assert result.header is not None
        np.testing.assert_array_equal(result.image, image)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_size(self):
        """Minimum 2x2 image."""
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_all_zeros(self):
        """All-zero image should roundtrip correctly."""
        image = np.zeros((16, 16), dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_all_ones(self):
        """All-255 image should roundtrip correctly."""
        image = np.full((16, 16), 255, dtype=np.uint8)
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_checkerboard_pattern(self):
        """Checkerboard pattern."""
        image = np.zeros((8, 8), dtype=np.uint8)
        image[::2, ::2] = 255
        image[1::2, 1::2] = 255
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_vertical_stripes(self):
        """Vertical stripe pattern."""
        image = np.zeros((8, 8), dtype=np.uint8)
        image[:, ::2] = 255
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)

    def test_horizontal_stripes(self):
        """Horizontal stripe pattern."""
        image = np.zeros((8, 8), dtype=np.uint8)
        image[::2, :] = 255
        encoded = encode(image, quality=QUALITY_MAX)
        decoded = decode(encoded)
        np.testing.assert_array_equal(decoded, image)


class TestReproducibility:
    """Test that encoding is deterministic."""

    def test_same_input_same_output(self):
        """Same input should produce identical output."""
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        encoded1 = encode(image, quality=QUALITY_MAX)
        encoded2 = encode(image, quality=QUALITY_MAX)
        assert encoded1 == encoded2

    def test_same_input_same_output_rgb(self):
        """Same RGB input should produce identical output."""
        rng = np.random.default_rng(999)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        encoded1 = encode(image, quality=QUALITY_MAX)
        encoded2 = encode(image, quality=QUALITY_MAX)
        assert encoded1 == encoded2
