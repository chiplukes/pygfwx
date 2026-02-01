"""
Tests for the GFWX SDK wrapper.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for cross_codec import
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_codec.gfwx_sdk import (
    Encoder,
    Filter,
    GFWXWrapper,
    is_sdk_available,
)

from pygfwx.utils.reference_images import create_reference_image, create_uniform_image

# Skip all tests if SDK not available
pytestmark = pytest.mark.skipif(not is_sdk_available(), reason="GFWX SDK not available")


@pytest.fixture
def sdk():
    """Create SDK wrapper instance."""
    return GFWXWrapper()


class TestSDKBasics:
    """Basic SDK functionality tests."""

    def test_sdk_loads(self, sdk):
        """Verify SDK loads successfully."""
        assert sdk.library_path.exists()

    def test_is_sdk_available(self):
        """Test SDK availability check."""
        assert is_sdk_available()


class TestSDKEncodeDecode:
    """Encode/decode roundtrip tests."""

    def test_roundtrip_mono_lossless(self, sdk):
        """Test lossless mono image roundtrip."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)

        compressed = sdk.encode(image, quality=1024)
        decoded = sdk.decode(compressed)

        assert decoded.shape == image.shape
        assert decoded.dtype == image.dtype
        assert np.array_equal(image, decoded), "Lossless roundtrip should be exact"

    def test_roundtrip_rgb_lossless(self, sdk):
        """Test lossless RGB image roundtrip."""
        image = create_reference_image(size=64, channels=3, bit_depth=8)

        compressed = sdk.encode(image, quality=1024, use_transform=False)
        decoded = sdk.decode(compressed)

        assert decoded.shape == image.shape
        assert decoded.dtype == image.dtype
        assert np.array_equal(image, decoded), "Lossless roundtrip should be exact"

    def test_roundtrip_mono_lossy(self, sdk):
        """Test lossy mono image roundtrip."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)

        compressed = sdk.encode(image, quality=512)
        decoded = sdk.decode(compressed)

        assert decoded.shape == image.shape
        assert decoded.dtype == image.dtype
        # Lossy should be close but not exact
        max_diff = np.abs(image.astype(np.int16) - decoded.astype(np.int16)).max()
        assert max_diff < 50, f"Lossy diff too large: {max_diff}"

    def test_roundtrip_uniform(self, sdk):
        """Test uniform image roundtrip."""
        image = create_uniform_image(size=64, value=128, channels=1, bit_depth=8)

        compressed = sdk.encode(image, quality=1024)
        decoded = sdk.decode(compressed)

        assert np.array_equal(image, decoded)

    def test_roundtrip_16bit(self, sdk):
        """Test 16-bit image roundtrip."""
        image = create_reference_image(size=64, channels=1, bit_depth=16)

        compressed = sdk.encode(image, quality=1024)
        decoded = sdk.decode(compressed)

        assert decoded.shape == image.shape
        assert decoded.dtype == np.uint16
        assert np.array_equal(image, decoded)


class TestSDKFilters:
    """Test different filter options."""

    def test_filter_linear(self, sdk):
        """Test LINEAR filter."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024, filter=Filter.LINEAR)
        decoded = sdk.decode(compressed)
        assert np.array_equal(image, decoded)

    def test_filter_cubic(self, sdk):
        """Test CUBIC filter."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024, filter=Filter.CUBIC)
        decoded = sdk.decode(compressed)
        assert np.array_equal(image, decoded)


class TestSDKEncoders:
    """Test different encoder modes."""

    def test_encoder_turbo_unsupported(self, sdk):
        """Test TURBO encoder returns unsupported (deprecated in SDK v1)."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        # TURBO encoder is not supported in SDK version 1 - encoder must be >= EncoderFast
        with pytest.raises(RuntimeError, match="unsupported"):
            sdk.encode(image, quality=1024, encoder=Encoder.TURBO)

    def test_encoder_fast(self, sdk):
        """Test FAST encoder."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024, encoder=Encoder.FAST)
        decoded = sdk.decode(compressed)
        assert np.array_equal(image, decoded)

    def test_encoder_contextual(self, sdk):
        """Test CONTEXTUAL encoder."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024, encoder=Encoder.CONTEXTUAL)
        decoded = sdk.decode(compressed)
        assert np.array_equal(image, decoded)


class TestSDKHeader:
    """Test header reading."""

    def test_read_header(self, sdk):
        """Test reading header from compressed data."""
        image = create_reference_image(size=64, channels=3, bit_depth=8)
        compressed = sdk.encode(image, quality=512, filter=Filter.CUBIC)

        header = sdk.read_header(compressed)

        assert header.sizex == 64
        assert header.sizey == 64
        assert header.channels == 3
        assert header.bit_depth == 8
        assert header.quality == 512
        assert header.filter == Filter.CUBIC
        assert not header.is_signed


class TestSDKDownsampling:
    """Test progressive decoding with downsampling."""

    def test_downsample_1(self, sdk):
        """Test 2x downsampling."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024)
        decoded = sdk.decode(compressed, downsampling=1)

        assert decoded.shape == (32, 32)

    def test_downsample_2(self, sdk):
        """Test 4x downsampling."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024)
        decoded = sdk.decode(compressed, downsampling=2)

        assert decoded.shape == (16, 16)
