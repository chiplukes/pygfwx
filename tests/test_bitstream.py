"""
Tests for the bitstream reader/writer module.
"""

import numpy as np
import pytest

from pygfwx.core.bitstream import BitReader, BitstreamOverflowError, BitWriter


class TestBitReader:
    """Tests for BitReader class."""

    def test_read_single_byte(self):
        """Test reading 8 bits from a simple buffer.

        GFWX uses little-endian word storage with MSB-first bit access.
        Bytes [0xAB, 0x00, 0x00, 0x00] become uint32 0x000000AB in LE.
        MSB-first access of 8 bits gives 0x00 (high byte of word).
        To get 0xAB from first 8 bits, we need it in the MSB position.
        """
        # Put 0xAB in the MSB of the 32-bit word (little-endian bytes)
        data = bytes([0x00, 0x00, 0x00, 0xAB])  # LE uint32 = 0xAB000000
        reader = BitReader(data)

        # Read MSB first: should get 0xAB
        assert reader.get_bits(8) == 0xAB

    def test_read_multiple_small_reads(self):
        """Test reading bits in small chunks.

        For bit-by-bit access, the word value determines order.
        0xAB000000 = 10101011 00000000 00000000 00000000 in binary.
        MSB-first reads: 1, 0, 1, 0, 1, 0, 1, 1
        """
        # 0xAB in MSB position
        data = bytes([0x00, 0x00, 0x00, 0xAB])  # LE uint32 = 0xAB000000
        reader = BitReader(data)

        # Read bit by bit from MSB
        assert reader.get_bits(1) == 1  # 1
        assert reader.get_bits(1) == 0  # 0
        assert reader.get_bits(1) == 1  # 1
        assert reader.get_bits(1) == 0  # 0
        assert reader.get_bits(1) == 1  # 1
        assert reader.get_bits(1) == 0  # 0
        assert reader.get_bits(1) == 1  # 1
        assert reader.get_bits(1) == 1  # 1

    def test_read_cross_word_boundary(self):
        """Test reading bits that cross a 32-bit word boundary."""
        # Two words: 0xFFFFFFFF, 0x00000000
        data = np.array([0xFFFFFFFF, 0x00000000], dtype=np.uint32)
        reader = BitReader(data)

        # Read 16 bits from first word
        assert reader.get_bits(16) == 0xFFFF
        # Read 32 bits crossing boundary (16 from word 0, 16 from word 1)
        assert reader.get_bits(32) == 0xFFFF0000

    def test_read_full_32_bits(self):
        """Test reading exactly 32 bits at once."""
        data = np.array([0xDEADBEEF], dtype=np.uint32)
        reader = BitReader(data)
        assert reader.get_bits(32) == 0xDEADBEEF

    def test_get_zeros_simple(self):
        """Test counting zeros followed by a 1."""
        # 0x80000000 = 10000000... (1 followed by zeros)
        data = np.array([0x80000000], dtype=np.uint32)
        reader = BitReader(data)
        assert reader.get_zeros(32) == 0  # Zero zeros before the 1

        # 0x40000000 = 01000000... (0, then 1)
        data = np.array([0x40000000], dtype=np.uint32)
        reader = BitReader(data)
        assert reader.get_zeros(32) == 1  # One zero before the 1

    def test_get_zeros_max_reached(self):
        """Test that get_zeros stops at max_zeros."""
        # All zeros
        data = np.array([0x00000000, 0x00000000], dtype=np.uint32)
        reader = BitReader(data)
        assert reader.get_zeros(12) == 12  # Should stop at max

    def test_get_zeros_cross_word(self):
        """Test counting zeros across word boundary."""
        # First word all zeros, second word starts with 1
        data = np.array([0x00000000, 0x80000000], dtype=np.uint32)
        reader = BitReader(data)
        assert reader.get_zeros(64) == 32  # 32 zeros in first word, then 1

    def test_overflow_detection(self):
        """Test that overflow is detected properly."""
        data = bytes([0x00, 0x00, 0x00, 0x00])  # One word
        reader = BitReader(data)
        reader.get_bits(32)  # Read all bits

        with pytest.raises(BitstreamOverflowError):
            reader.get_bits(1)

    def test_position_tracking(self):
        """Test position_bits property."""
        data = np.array([0xFFFFFFFF, 0x00000000], dtype=np.uint32)
        reader = BitReader(data)

        assert reader.position_bits == 0
        reader.get_bits(8)
        assert reader.position_bits == 8
        reader.get_bits(24)
        assert reader.position_bits == 32
        reader.get_bits(16)
        assert reader.position_bits == 48

    def test_flush_read_word(self):
        """Test flushing to next word boundary."""
        data = np.array([0xFFFFFFFF, 0xAAAAAAAA], dtype=np.uint32)
        reader = BitReader(data)

        reader.get_bits(8)  # Read 8 bits
        assert reader.position_bits == 8

        reader.flush_read_word()
        assert reader.position_bits == 32  # Moved to word 1, bit 0

        assert reader.get_bits(32) == 0xAAAAAAAA


class TestBitWriter:
    """Tests for BitWriter class."""

    def test_write_single_byte(self):
        """Test writing 8 bits."""
        writer = BitWriter(1)
        writer.put_bits(0xAB, 8)
        writer.flush_write_word()

        # Should be in MSB position of the word
        assert writer.buffer[0] == 0xAB000000

    def test_write_multiple_small_writes(self):
        """Test writing bits in small chunks."""
        writer = BitWriter(1)

        # Write 10101011 bit by bit
        writer.put_bits(1, 1)
        writer.put_bits(0, 1)
        writer.put_bits(1, 1)
        writer.put_bits(0, 1)
        writer.put_bits(1, 1)
        writer.put_bits(0, 1)
        writer.put_bits(1, 1)
        writer.put_bits(1, 1)
        writer.flush_write_word()

        assert (writer.buffer[0] >> 24) == 0xAB

    def test_write_full_32_bits(self):
        """Test writing exactly 32 bits at once."""
        writer = BitWriter(1)
        writer.put_bits(0xDEADBEEF, 32)

        assert writer.buffer[0] == 0xDEADBEEF

    def test_write_cross_word_boundary(self):
        """Test writing bits that cross a 32-bit word boundary."""
        writer = BitWriter(2)

        writer.put_bits(0xFFFF, 16)  # First 16 bits
        writer.put_bits(0xDEADBEEF, 32)  # Crosses boundary
        writer.flush_write_word()

        assert writer.buffer[0] == 0xFFFFDEAD
        assert (writer.buffer[1] >> 16) == 0xBEEF

    def test_roundtrip(self):
        """Test that written data can be read back correctly."""
        writer = BitWriter(10)

        # Write various patterns
        writer.put_bits(0x7F, 7)
        writer.put_bits(0xABCD, 16)
        writer.put_bits(0x1, 1)
        writer.put_bits(0xDEADBEEF, 32)
        writer.flush_write_word()

        # Read back
        reader = BitReader(writer.buffer)

        assert reader.get_bits(7) == 0x7F
        assert reader.get_bits(16) == 0xABCD
        assert reader.get_bits(1) == 0x1
        assert reader.get_bits(32) == 0xDEADBEEF

    def test_get_data(self):
        """Test getting written data as bytes."""
        writer = BitWriter(2)
        writer.put_bits(0xDEADBEEF, 32)
        writer.put_bits(0xCAFE, 16)

        data = writer.get_data()
        # Should be 8 bytes (2 words) after flush
        assert len(data) == 8

    def test_overflow_detection(self):
        """Test that overflow is detected properly."""
        writer = BitWriter(1)
        writer.put_bits(0xFFFFFFFF, 32)  # Fill the buffer

        with pytest.raises(BitstreamOverflowError):
            writer.put_bits(0x1, 1)


class TestBitstreamRoundtrip:
    """End-to-end roundtrip tests."""

    def test_random_patterns(self):
        """Test roundtrip with random bit patterns."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            # Generate random values with random bit widths
            values = []
            total_bits = 0
            for _ in range(20):
                bits = rng.integers(1, 33)
                max_val = (1 << bits) - 1
                val = int(rng.integers(0, max_val + 1, dtype=np.uint64))
                values.append((val, bits))
                total_bits += bits

            # Write
            words_needed = (total_bits + 31) // 32 + 1
            writer = BitWriter(words_needed)
            for val, bits in values:
                writer.put_bits(val, bits)
            writer.flush_write_word()

            # Read
            reader = BitReader(writer.buffer)
            for expected_val, bits in values:
                actual_val = reader.get_bits(bits)
                assert actual_val == expected_val, f"Expected {expected_val}, got {actual_val} for {bits} bits"

    def test_golomb_like_pattern(self):
        """Test pattern similar to Golomb-Rice coding."""
        writer = BitWriter(10)

        # Simulate Golomb coding: unary zeros + binary suffix
        test_cases = [
            (3, 4),  # 3 zeros then 4-bit value
            (0, 2),  # 0 zeros then 2-bit value
            (7, 5),  # 7 zeros then 5-bit value
        ]

        for zeros, suffix_bits in test_cases:
            # Write zeros
            writer.put_bits(0, zeros)
            # Write terminating 1
            writer.put_bits(1, 1)
            # Write suffix
            writer.put_bits(0xF & ((1 << suffix_bits) - 1), suffix_bits)

        writer.flush_write_word()

        # Read back using get_zeros
        reader = BitReader(writer.buffer)

        for expected_zeros, suffix_bits in test_cases:
            actual_zeros = reader.get_zeros(32)
            assert actual_zeros == expected_zeros
            suffix = reader.get_bits(suffix_bits)
            assert suffix == (0xF & ((1 << suffix_bits) - 1))
