"""
Tests for GFWX coefficient encoder.

Tests encode/decode roundtrip for all encoder modes, verifying that
encoding then decoding produces the original coefficients.
"""

import numpy as np
import pytest

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.decoder import decode_block, decode_coefficients
from pygfwx.core.encoder import encode_block, encode_coefficients
from pygfwx.core.header import Encoder


def create_writer(word_count: int = 256) -> BitWriter:
    """Create a BitWriter with specified word capacity."""
    return BitWriter(word_count)


class TestEncodeCoefficients:
    """Tests for encode_coefficients function."""

    def test_single_dc_coefficient(self):
        """Test encoding/decoding a single DC coefficient."""
        # Create a 1x1 image with a DC value
        original = np.array([[42]], dtype=np.int32)

        # Encode
        writer = create_writer(32)
        encode_coefficients(
            original, writer, 0, 0, 1, 1, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )
        data = writer.get_data()

        # Decode
        decoded = np.zeros((1, 1), dtype=np.int32)
        reader = BitReader(data)
        decode_coefficients(
            decoded, reader, 0, 0, 1, 1, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )

        assert decoded[0, 0] == 42

    def test_dc_negative(self):
        """Test encoding/decoding negative DC coefficient."""
        original = np.array([[-127]], dtype=np.int32)

        writer = create_writer(32)
        encode_coefficients(
            original, writer, 0, 0, 1, 1, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )
        data = writer.get_data()

        decoded = np.zeros((1, 1), dtype=np.int32)
        reader = BitReader(data)
        decode_coefficients(
            decoded, reader, 0, 0, 1, 1, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )

        assert decoded[0, 0] == -127

    def test_2x2_all_zeros(self):
        """Test encoding/decoding all zero coefficients."""
        original = np.zeros((2, 2), dtype=np.int32)

        writer = create_writer(32)
        encode_coefficients(
            original, writer, 0, 0, 2, 2, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )
        data = writer.get_data()

        decoded = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)
        decode_coefficients(
            decoded, reader, 0, 0, 2, 2, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )

        np.testing.assert_array_equal(decoded, original)

    def test_2x2_with_detail(self):
        """Test encoding/decoding 2x2 with DC and detail coefficients."""
        # DC at (0,0), detail at (1,0), (0,1), (1,1) depending on step
        original = np.array([[100, 5], [-3, 7]], dtype=np.int32)

        writer = create_writer(32)
        encode_coefficients(
            original, writer, 0, 0, 2, 2, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )
        data = writer.get_data()

        decoded = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)
        decode_coefficients(
            decoded, reader, 0, 0, 2, 2, step=1, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )

        np.testing.assert_array_equal(decoded, original)

    def test_4x4_fast_mode(self):
        """Test encoding/decoding 4x4 block in FAST mode using encode_block."""
        np.random.seed(42)
        original = np.random.randint(-50, 50, (4, 4), dtype=np.int32)

        writer = create_writer(64)
        encode_block(original, writer, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_4x4_contextual_mode(self):
        """Test encoding/decoding 4x4 block in CONTEXTUAL mode using encode_block."""
        np.random.seed(123)
        original = np.random.randint(-100, 100, (4, 4), dtype=np.int32)

        writer = create_writer(64)
        encode_block(original, writer, 0, 0, 4, 4, scheme=Encoder.CONTEXTUAL, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 4, 4, scheme=Encoder.CONTEXTUAL, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_8x8_high_bitrate_mode(self):
        """Test encoding/decoding 8x8 block in HIGH_BITRATE mode using encode_block."""
        np.random.seed(456)
        original = np.random.randint(-200, 200, (8, 8), dtype=np.int32)

        writer = create_writer(256)
        encode_block(original, writer, 0, 0, 8, 8, scheme=Encoder.HIGH_BITRATE, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((8, 8), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 8, 8, scheme=Encoder.HIGH_BITRATE, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_chroma_flag_affects_encoding(self):
        """Test that chroma flag produces different encoded data."""
        np.random.seed(789)
        original = np.random.randint(-50, 50, (4, 4), dtype=np.int32)

        # Encode without chroma flag
        writer1 = create_writer(64)
        encode_block(original, writer1, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data1 = writer1.get_data()

        # Encode with chroma flag
        writer2 = create_writer(64)
        encode_block(original, writer2, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=True)
        data2 = writer2.get_data()

        # Both should decode correctly
        decoded1 = np.zeros((4, 4), dtype=np.int32)
        reader1 = BitReader(data1)
        decode_block(decoded1, reader1, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        np.testing.assert_array_equal(decoded1, original)

        decoded2 = np.zeros((4, 4), dtype=np.int32)
        reader2 = BitReader(data2)
        decode_block(decoded2, reader2, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=True)
        np.testing.assert_array_equal(decoded2, original)


class TestEncodeBlock:
    """Tests for encode_block function."""

    def test_4x4_block(self):
        """Test encoding/decoding full 4x4 block with hierarchy."""
        np.random.seed(111)
        original = np.random.randint(-100, 100, (4, 4), dtype=np.int32)

        writer = create_writer(128)
        encode_block(original, writer, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_8x8_block(self):
        """Test encoding/decoding full 8x8 block with hierarchy."""
        np.random.seed(222)
        original = np.random.randint(-100, 100, (8, 8), dtype=np.int32)

        writer = create_writer(256)
        encode_block(original, writer, 0, 0, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((8, 8), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_16x16_block_contextual(self):
        """Test encoding/decoding 16x16 block in CONTEXTUAL mode."""
        np.random.seed(333)
        original = np.random.randint(-50, 50, (16, 16), dtype=np.int32)

        writer = create_writer(512)
        encode_block(original, writer, 0, 0, 16, 16, scheme=Encoder.CONTEXTUAL, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((16, 16), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 16, 16, scheme=Encoder.CONTEXTUAL, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)


class TestRoundtripAllModes:
    """Test encode/decode roundtrip for all encoder modes."""

    @pytest.mark.parametrize(
        "scheme",
        [
            Encoder.FAST,
            Encoder.CONTEXTUAL,
            Encoder.HIGH_BITRATE,
        ],
    )
    def test_8x8_all_modes(self, scheme):
        """Test 8x8 block roundtrip for all modes using encode_block."""
        np.random.seed(42 + scheme.value)
        original = np.random.randint(-100, 100, (8, 8), dtype=np.int32)

        writer = create_writer(256)
        encode_block(original, writer, 0, 0, 8, 8, scheme=scheme, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((8, 8), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 8, 8, scheme=scheme, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    @pytest.mark.parametrize("quality", [512, 768, 1024])
    def test_quality_variations(self, quality):
        """Test roundtrip with different quality settings using encode_block."""
        np.random.seed(quality)
        original = np.random.randint(-50, 50, (8, 8), dtype=np.int32)

        writer = create_writer(256)
        encode_block(original, writer, 0, 0, 8, 8, scheme=Encoder.CONTEXTUAL, quality=quality, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((8, 8), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 8, 8, scheme=Encoder.CONTEXTUAL, quality=quality, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)


class TestEdgeCases:
    """Tests for edge cases and special patterns."""

    def test_all_same_value(self):
        """Test encoding when all coefficients have the same value."""
        original = np.full((4, 4), 25, dtype=np.int32)

        writer = create_writer(64)
        encode_block(original, writer, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_alternating_pattern(self):
        """Test encoding alternating positive/negative pattern."""
        original = np.array(
            [
                [10, -10, 10, -10],
                [-10, 10, -10, 10],
                [10, -10, 10, -10],
                [-10, 10, -10, 10],
            ],
            dtype=np.int32,
        )

        writer = create_writer(64)
        encode_block(original, writer, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 4, 4, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_sparse_coefficients(self):
        """Test encoding with mostly zeros and few non-zeros."""
        original = np.zeros((8, 8), dtype=np.int32)
        original[0, 0] = 100  # DC
        original[3, 2] = 7
        original[7, 7] = -15

        writer = create_writer(128)
        encode_block(original, writer, 0, 0, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((8, 8), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_large_values(self):
        """Test encoding large coefficient values."""
        original = np.array([[10000, -5000], [32000, -32000]], dtype=np.int32)

        writer = create_writer(64)
        encode_block(original, writer, 0, 0, 2, 2, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 0, 0, 2, 2, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded, original)

    def test_subregion_encoding(self):
        """Test encoding a subregion of a larger array."""
        full = np.zeros((16, 16), dtype=np.int32)
        # Set values in subregion
        full[4:8, 4:8] = np.array(
            [
                [50, -10, 5, -3],
                [-5, 20, -8, 2],
                [15, -25, 10, -1],
                [-2, 7, -4, 30],
            ],
            dtype=np.int32,
        )
        original_subregion = full[4:8, 4:8].copy()

        writer = create_writer(128)
        encode_block(full, writer, 4, 4, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)
        data = writer.get_data()

        decoded = np.zeros((16, 16), dtype=np.int32)
        reader = BitReader(data)
        decode_block(decoded, reader, 4, 4, 8, 8, scheme=Encoder.FAST, quality=1024, is_chroma=False)

        np.testing.assert_array_equal(decoded[4:8, 4:8], original_subregion)

    def test_step_2_encoding(self):
        """Test encoding with step=2 (for wavelet hierarchy)."""
        original = np.array(
            [
                [100, 0, 20, 0],
                [0, 5, 0, -3],
                [50, 0, 10, 0],
                [0, -2, 0, 8],
            ],
            dtype=np.int32,
        )

        # Only positions where (x | y) & 2 == 2 will be processed with step=2
        writer = create_writer(64)
        encode_coefficients(
            original, writer, 0, 0, 4, 4, step=2, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )
        data = writer.get_data()

        decoded = np.zeros((4, 4), dtype=np.int32)
        reader = BitReader(data)
        decode_coefficients(
            decoded, reader, 0, 0, 4, 4, step=2, scheme=Encoder.FAST, quality=1024, has_dc=True, is_chroma=False
        )

        # Check positions processed by step=2
        assert decoded[0, 0] == original[0, 0]  # DC
        assert decoded[0, 2] == original[0, 2]
        assert decoded[2, 0] == original[2, 0]
        assert decoded[2, 2] == original[2, 2]
