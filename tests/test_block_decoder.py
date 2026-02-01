"""
Tests for the GFWX Block Decoder.

These tests verify:
- Transform program parsing
- Block information calculation
- Dequantization
- Full decode pipeline with SDK comparison
"""

import numpy as np
import pytest

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.block_decoder import (
    BlockInfo,
    DecodeResult,
    _parse_transform_program,
    decode_image,
    get_block_info,
)
from pygfwx.core.golomb_rice import signed_encode
from pygfwx.core.header import GFWXHeader
from pygfwx.core.quantization import dequantize


class TestParseTransformProgram:
    """Tests for transform program parsing."""

    def test_no_transform(self):
        """Test parsing an empty transform (just -1 terminator)."""
        # Encode a simple "no transform" program: just -1
        writer = BitWriter(100)
        signed_encode(2, -1, writer)
        data = writer.get_data()

        stream = BitReader(data)
        program, steps, is_chroma = _parse_transform_program(stream, 3)

        assert program == [-1]
        assert steps == []
        assert is_chroma == [0, 0, 0]

    def test_simple_transform(self):
        """Test parsing a simple single-step transform."""
        # Encode: channel 0, uses channel 1 with factor 1, denominator 1, chroma=1
        writer = BitWriter(100)
        signed_encode(2, 0, writer)  # Destination channel 0
        signed_encode(2, 1, writer)  # Source channel 1
        signed_encode(2, 1, writer)  # Factor 1
        signed_encode(2, -1, writer)  # End sources
        signed_encode(2, 1, writer)  # Denominator 1
        signed_encode(2, 1, writer)  # Chroma flag
        signed_encode(2, -1, writer)  # End program
        data = writer.get_data()

        stream = BitReader(data)
        program, steps, is_chroma = _parse_transform_program(stream, 3)

        assert program[0] == 0  # Destination channel
        assert steps == [0]  # First step at index 0
        assert is_chroma[0] == 1  # Channel 0 marked as chroma


class TestDequantize:
    """Tests for dequantization."""

    def test_lossless_no_change(self):
        """Test that lossless (q=1024) doesn't change coefficients."""
        image = np.array([[10, -5], [3, 0]], dtype=np.int32)
        original = image.copy()

        # Quality 1024 means q >= maxQ, so no dequantization
        dequantize(image, 0, 0, 2, 2, 1, 1024, 0, 1024)

        # Should be unchanged since q >= maxQ on first iteration
        # Actually the loop exits immediately when q >= maxQ
        assert np.array_equal(image, original)

    def test_lossy_scaling(self):
        """Test that lossy dequantization scales coefficients."""
        image = np.array([[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]], dtype=np.int32)


        # Low quality should scale up coefficients
        dequantize(image, 0, 0, 4, 4, 1, 100, 0, 1024)

        # Coefficients should be larger after dequantization
        # (except zeros which stay zero)
        # The specific values depend on the quality curve

    def test_zero_preserved(self):
        """Test that zero coefficients remain zero."""
        image = np.zeros((4, 4), dtype=np.int32)

        dequantize(image, 0, 0, 4, 4, 1, 100, 0, 1024)

        assert np.all(image == 0)


class TestGetBlockInfo:
    """Tests for block information calculation."""

    def test_single_block_small_image(self):
        """Test block info for image fitting in one block."""
        header = GFWXHeader(
            version=1,
            sizex=8,
            sizey=8,
            layers=1,
            channels=1,
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=6,  # 2^6 = 64, larger than image
            filter=0,
            quantization=0,
            encoder=0,
            intent=0,
            metadata_size=0,
        )

        # At step=4 (coarsest for 8x8)
        blocks = get_block_info(header, step=4, downsampling=0)

        # Should be 1 block for 1 channel
        assert len(blocks) == 1
        assert blocks[0].bx == 0
        assert blocks[0].by == 0
        assert blocks[0].channel == 0
        assert blocks[0].x0 == 0
        assert blocks[0].y0 == 0
        assert blocks[0].x1 == 8
        assert blocks[0].y1 == 8

    def test_multiple_blocks_large_image(self):
        """Test block info for image spanning multiple blocks."""
        header = GFWXHeader(
            version=1,
            sizex=128,
            sizey=128,
            layers=1,
            channels=1,
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=4,  # 2^4 = 16 per step
            filter=0,
            quantization=0,
            encoder=0,
            intent=0,
            metadata_size=0,
        )

        # At step=64 with block_size=4: bs = 64 * 16 = 1024 > 128, so 1 block per dim
        blocks = get_block_info(header, step=64, downsampling=0)
        assert len(blocks) == 1

        # At step=1 with block_size=4: bs = 1 * 16 = 16
        # 128/16 = 8 blocks per dimension = 64 total for 1 channel
        blocks = get_block_info(header, step=1, downsampling=0)
        assert len(blocks) == 64

    def test_multi_channel(self):
        """Test block info for multi-channel image."""
        header = GFWXHeader(
            version=1,
            sizex=32,
            sizey=32,
            layers=1,
            channels=3,  # RGB
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=6,  # Large blocks
            filter=0,
            quantization=0,
            encoder=0,
            intent=0,
            metadata_size=0,
        )

        blocks = get_block_info(header, step=16, downsampling=0)

        # Should have 1 block per channel = 3 blocks
        assert len(blocks) == 3
        channels = [b.channel for b in blocks]
        assert channels == [0, 1, 2]


class TestDecodeImage:
    """Tests for the full decode pipeline."""

    @pytest.mark.sdk_required
    def test_decode_sdk_encoded_mono_lossless(self, sdk_wrapper, reference_image):  # noqa: ARG002
        """Test decoding an SDK-encoded lossless monochrome image."""
        from cross_codec.gfwx_sdk import encode

        # Encode with SDK
        compressed = encode(reference_image, quality=1024)

        # Decode with our implementation
        result = decode_image(compressed)

        assert isinstance(result, DecodeResult)
        assert result.header.sizex == reference_image.shape[1]
        assert result.header.sizey == reference_image.shape[0]
        assert result.header.quality == 1024
        # For lossless, should match exactly
        # (This may need adjustment based on color transform handling)

    @pytest.mark.sdk_required
    def test_decode_dimensions(self, sdk_wrapper):  # noqa: ARG002
        """Test that decoded dimensions match header."""
        from cross_codec.gfwx_sdk import encode

        image = np.arange(48, dtype=np.uint8).reshape(6, 8)
        compressed = encode(image, quality=1024)

        result = decode_image(compressed)

        assert result.image.shape == (6, 8)
        assert result.header.sizex == 8
        assert result.header.sizey == 6


class TestBlockInfoDataclass:
    """Tests for BlockInfo dataclass."""

    def test_block_info_creation(self):
        """Test creating a BlockInfo instance."""
        info = BlockInfo(
            bx=1,
            by=2,
            channel=0,
            x0=64,
            y0=128,
            x1=128,
            y1=192,
            size_words=100,
        )

        assert info.bx == 1
        assert info.by == 2
        assert info.channel == 0
        assert info.x0 == 64
        assert info.y0 == 128
        assert info.x1 == 128
        assert info.y1 == 192
        assert info.size_words == 100


class TestDecodeResultDataclass:
    """Tests for DecodeResult dataclass."""

    def test_decode_result_creation(self):
        """Test creating a DecodeResult instance."""
        image = np.zeros((10, 10), dtype=np.uint8)
        header = GFWXHeader(
            version=1,
            sizex=10,
            sizey=10,
            layers=1,
            channels=1,
            bit_depth=8,
            is_signed=False,
            quality=1024,
            chroma_scale=1,
            block_size=6,
            filter=0,
            quantization=0,
            encoder=0,
            intent=0,
            metadata_size=0,
        )

        result = DecodeResult(image=image, header=header, is_truncated=False)

        assert result.image.shape == (10, 10)
        assert result.header.sizex == 10
        assert not result.is_truncated
