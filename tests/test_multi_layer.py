"""
Tests for multi-layer GFWX support.

Tests cover:
- MultiLayerImage dataclass operations
- Creating multi-layer images from individual layers
- Splitting multi-layer images back to individual layers
- SDK encode/decode roundtrip for multi-layer data
- PyGFWX decode of SDK-encoded multi-layer data
"""

import numpy as np
import pytest

from pygfwx.core.multi_layer import (
    MultiLayerImage,
    create_multi_layer,
    decode_result_to_multi_layer,
    split_layers,
    validate_multi_layer_header,
)


# ============================================================================
# MultiLayerImage Tests
# ============================================================================


class TestMultiLayerImage:
    """Test MultiLayerImage dataclass."""

    def test_properties(self):
        """Test basic properties."""
        data = np.zeros((100, 200, 6), dtype=np.uint8)
        ml = MultiLayerImage(data=data, layers=2, channels=3)

        assert ml.height == 100
        assert ml.width == 200
        assert ml.layers == 2
        assert ml.channels == 3
        assert ml.total_channels == 6
        assert ml.dtype == np.uint8

    def test_get_layer_rgb(self):
        """Test extracting a layer from RGB stereo."""
        data = np.zeros((10, 20, 6), dtype=np.uint8)
        # Fill left layer (channels 0-2) with 100
        data[:, :, 0:3] = 100
        # Fill right layer (channels 3-5) with 200
        data[:, :, 3:6] = 200

        ml = MultiLayerImage(data=data, layers=2, channels=3)

        left = ml.get_layer(0)
        right = ml.get_layer(1)

        assert left.shape == (10, 20, 3)
        assert right.shape == (10, 20, 3)
        assert np.all(left == 100)
        assert np.all(right == 200)

    def test_get_layer_mono(self):
        """Test extracting a layer from mono stereo."""
        data = np.zeros((10, 20, 2), dtype=np.uint8)
        data[:, :, 0] = 50
        data[:, :, 1] = 150

        ml = MultiLayerImage(data=data, layers=2, channels=1)

        left = ml.get_layer(0)
        right = ml.get_layer(1)

        assert left.shape == (10, 20)
        assert right.shape == (10, 20)
        assert np.all(left == 50)
        assert np.all(right == 150)

    def test_get_layer_out_of_range(self):
        """Test getting a layer out of range."""
        data = np.zeros((10, 20, 6), dtype=np.uint8)
        ml = MultiLayerImage(data=data, layers=2, channels=3)

        with pytest.raises(IndexError):
            ml.get_layer(2)

        with pytest.raises(IndexError):
            ml.get_layer(-1)

    def test_set_layer_rgb(self):
        """Test setting a layer in RGB stereo."""
        data = np.zeros((10, 20, 6), dtype=np.uint8)
        ml = MultiLayerImage(data=data, layers=2, channels=3)

        new_layer = np.full((10, 20, 3), 128, dtype=np.uint8)
        ml.set_layer(1, new_layer)

        assert np.all(ml.data[:, :, 0:3] == 0)
        assert np.all(ml.data[:, :, 3:6] == 128)

    def test_set_layer_mono(self):
        """Test setting a layer in mono stereo."""
        data = np.zeros((10, 20, 2), dtype=np.uint8)
        ml = MultiLayerImage(data=data, layers=2, channels=1)

        new_layer = np.full((10, 20), 200, dtype=np.uint8)
        ml.set_layer(0, new_layer)

        assert np.all(ml.data[:, :, 0] == 200)
        assert np.all(ml.data[:, :, 1] == 0)


# ============================================================================
# create_multi_layer Tests
# ============================================================================


class TestCreateMultiLayer:
    """Test create_multi_layer function."""

    def test_stereo_rgb(self):
        """Test creating stereo RGB from two layers."""
        left = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)

        ml = create_multi_layer(left, right)

        assert ml.layers == 2
        assert ml.channels == 3
        assert ml.data.shape == (10, 20, 6)

        # Verify interleaving
        assert np.array_equal(ml.get_layer(0), left)
        assert np.array_equal(ml.get_layer(1), right)

    def test_stereo_mono(self):
        """Test creating stereo mono from two layers."""
        left = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        right = np.random.randint(0, 255, (10, 20), dtype=np.uint8)

        ml = create_multi_layer(left, right)

        assert ml.layers == 2
        assert ml.channels == 1
        assert ml.data.shape == (10, 20, 2)

    def test_single_layer(self):
        """Test creating from a single layer."""
        img = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
        ml = create_multi_layer(img)

        assert ml.layers == 1
        assert ml.channels == 3
        assert np.array_equal(ml.get_layer(0), img)

    def test_three_layers(self):
        """Test creating from three layers."""
        layers = [
            np.full((5, 10, 3), i * 50, dtype=np.uint8)
            for i in range(3)
        ]
        ml = create_multi_layer(*layers)

        assert ml.layers == 3
        assert ml.channels == 3
        assert ml.data.shape == (5, 10, 9)

    def test_mismatched_shapes_raises(self):
        """Test that mismatched shapes raise an error."""
        a = np.zeros((10, 20, 3), dtype=np.uint8)
        b = np.zeros((15, 20, 3), dtype=np.uint8)  # Different height

        with pytest.raises(ValueError, match="doesn't match"):
            create_multi_layer(a, b)

    def test_no_layers_raises(self):
        """Test that zero layers raises an error."""
        with pytest.raises(ValueError, match="At least one layer"):
            create_multi_layer()

    def test_dtype_conversion(self):
        """Test dtype conversion."""
        a = np.zeros((10, 20, 3), dtype=np.float32)
        ml = create_multi_layer(a, dtype=np.uint8)

        assert ml.dtype == np.uint8


# ============================================================================
# split_layers Tests
# ============================================================================


class TestSplitLayers:
    """Test split_layers function."""

    def test_split_stereo_rgb(self):
        """Test splitting stereo RGB."""
        data = np.random.randint(0, 255, (10, 20, 6), dtype=np.uint8)
        layers = split_layers(data, layers=2, channels=3)

        assert len(layers) == 2
        assert layers[0].shape == (10, 20, 3)
        assert layers[1].shape == (10, 20, 3)
        assert np.array_equal(layers[0], data[:, :, 0:3])
        assert np.array_equal(layers[1], data[:, :, 3:6])

    def test_split_stereo_mono(self):
        """Test splitting stereo mono."""
        data = np.random.randint(0, 255, (10, 20, 2), dtype=np.uint8)
        layers = split_layers(data, layers=2, channels=1)

        assert len(layers) == 2
        assert layers[0].shape == (10, 20)
        assert layers[1].shape == (10, 20)

    def test_split_single_mono(self):
        """Test splitting single mono (2D array)."""
        data = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        layers = split_layers(data, layers=1, channels=1)

        assert len(layers) == 1
        assert np.array_equal(layers[0], data)

    def test_mismatch_channels_raises(self):
        """Test that wrong channel count raises error."""
        data = np.random.randint(0, 255, (10, 20, 6), dtype=np.uint8)

        with pytest.raises(ValueError, match="channels"):
            split_layers(data, layers=2, channels=4)  # 2*4 = 8 != 6


# ============================================================================
# Roundtrip Tests (require SDK)
# ============================================================================


class TestMultiLayerRoundtrip:
    """Test multi-layer encode/decode roundtrip with SDK."""

    def test_stereo_mono_lossless(self, sdk_wrapper):
        """Test stereo mono lossless roundtrip."""
        left = np.random.randint(0, 255, (32, 48), dtype=np.uint8)
        right = np.random.randint(0, 255, (32, 48), dtype=np.uint8)
        ml = create_multi_layer(left, right)

        # Encode
        compressed = sdk_wrapper.encode_multi_layer(
            ml.data,
            layers=2,
            channels=1,
            quality=1024,  # Lossless
            use_transform=False,
        )

        # Verify header
        header = sdk_wrapper.read_header(compressed)
        assert header.layers == 2
        assert header.channels == 1

        # Decode
        decoded = sdk_wrapper.decode(compressed)

        assert decoded.shape == ml.data.shape
        assert np.array_equal(decoded, ml.data)

    def test_stereo_rgb_lossless(self, sdk_wrapper):
        """Test stereo RGB lossless roundtrip."""
        left = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        ml = create_multi_layer(left, right)

        # Encode without transform (simpler validation)
        compressed = sdk_wrapper.encode_multi_layer(
            ml.data,
            layers=2,
            channels=3,
            quality=1024,
            use_transform=False,
        )

        # Verify header
        header = sdk_wrapper.read_header(compressed)
        assert header.layers == 2
        assert header.channels == 3

        # Decode
        decoded = sdk_wrapper.decode(compressed)

        assert decoded.shape == (32, 48, 6)
        assert np.array_equal(decoded, ml.data)

    def test_stereo_lossy(self, sdk_wrapper):
        """Test stereo lossy roundtrip."""
        left = np.full((32, 48, 3), 128, dtype=np.uint8)
        right = np.full((32, 48, 3), 200, dtype=np.uint8)
        ml = create_multi_layer(left, right)

        # Encode lossy
        compressed = sdk_wrapper.encode_multi_layer(
            ml.data,
            layers=2,
            channels=3,
            quality=512,  # Lossy
            use_transform=False,
        )

        # Decode
        decoded = sdk_wrapper.decode(compressed)

        # Check shape and approximate values
        assert decoded.shape == (32, 48, 6)
        # Should be close to original uniform values
        assert np.abs(decoded[:, :, 0:3].mean() - 128) < 10
        assert np.abs(decoded[:, :, 3:6].mean() - 200) < 10


class TestMultiLayerPyGFWXDecode:
    """Test PyGFWX decode of SDK-encoded multi-layer data."""

    def test_pygfwx_decodes_sdk_stereo(self, sdk_wrapper):
        """Test that PyGFWX can decode SDK-encoded stereo."""
        from pygfwx.core.block_decoder import decode_image

        left = np.random.randint(0, 255, (32, 48), dtype=np.uint8)
        right = np.random.randint(0, 255, (32, 48), dtype=np.uint8)
        ml = create_multi_layer(left, right)

        # Encode with SDK
        compressed = sdk_wrapper.encode_multi_layer(
            ml.data,
            layers=2,
            channels=1,
            quality=1024,
            use_transform=False,
        )

        # Decode with PyGFWX
        result = decode_image(compressed)

        assert result.header.layers == 2
        assert result.header.channels == 1
        assert result.image.shape == (32, 48, 2)
        assert np.array_equal(result.image, ml.data)


# ============================================================================
# Header Validation Tests
# ============================================================================


class TestValidateMultiLayerHeader:
    """Test header validation."""

    def test_valid_header(self):
        """Test valid header passes validation."""
        from pygfwx.core.header import GFWXHeader, Filter, Encoder, Intent

        header = GFWXHeader(
            version=1,
            sizex=100,
            sizey=100,
            layers=2,
            channels=3,
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=7,
            filter=Filter.LINEAR,
            quantization=0,
            encoder=Encoder.FAST,
            intent=Intent.RGB,
            metadata_size=0,
        )
        validate_multi_layer_header(header)  # Should not raise

    def test_zero_layers_invalid(self):
        """Test zero layers is invalid."""
        from pygfwx.core.header import GFWXHeader, Filter, Encoder, Intent

        header = GFWXHeader(
            version=1,
            sizex=100,
            sizey=100,
            layers=0,
            channels=3,
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=7,
            filter=Filter.LINEAR,
            quantization=0,
            encoder=Encoder.FAST,
            intent=Intent.RGB,
            metadata_size=0,
        )
        with pytest.raises(ValueError, match="Invalid layers"):
            validate_multi_layer_header(header)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
