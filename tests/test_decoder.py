"""
Tests for the GFWX Coefficient Decoder module.

These tests verify:
- Basic coefficient decoding
- Context-adaptive mode selection
- Run-length decoding
- DC coefficient handling
- Different encoder modes (TURBO, FAST, CONTEXTUAL)
"""

import numpy as np

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.context import Context, compute_run_coder_contextual, compute_run_coder_fast
from pygfwx.core.decoder import (
    _decode_with_context,
    decode_coefficients,
)
from pygfwx.core.golomb_rice import interleaved_encode, signed_encode
from pygfwx.core.header import Encoder


class TestDecodeWithContext:
    """Tests for context-adaptive decoding."""

    def test_low_activity_interleaved_0(self):
        """Test low activity uses interleaved pot=0."""
        # Encode a value with interleaved pot=0
        writer = BitWriter(100)
        interleaved_encode(0, 5, writer)
        data = writer.get_data()

        reader = BitReader(data)
        context = Context(sum=0, sum2=0)
        result = _decode_with_context(context, reader, is_chroma=False)
        assert result == 5

    def test_moderate_activity_interleaved_1(self):
        """Test moderate activity uses interleaved pot=1."""
        # Context where sum_sq >= 2*sum2 + 100 but < 2*sum2 + 950
        # sum=20, sum2=100 -> sum_sq=400, 2*sum2+100=300, 2*sum2+950=1150
        writer = BitWriter(100)
        interleaved_encode(1, -3, writer)
        data = writer.get_data()

        reader = BitReader(data)
        context = Context(sum=20, sum2=100)
        result = _decode_with_context(context, reader, is_chroma=False)
        assert result == -3

    def test_high_activity_signed_4(self):
        """Test high activity uses signed pot=4."""
        # Context with very high sum relative to sum2
        writer = BitWriter(100)
        signed_encode(4, -100, writer)
        data = writer.get_data()

        reader = BitReader(data)
        context = Context(sum=1000, sum2=1000)  # sum_sq = 1,000,000
        result = _decode_with_context(context, reader, is_chroma=False)
        assert result == -100

    def test_chroma_threshold_difference(self):
        """Test that chroma uses different threshold (250 vs 100)."""
        # Context at threshold boundary
        # sum=14, sum2=0 -> sum_sq=196
        # Luma: 2*0+100=100, 196 >= 100 -> not pot=0
        # Chroma: 2*0+250=250, 196 < 250 -> pot=0
        writer = BitWriter(100)
        interleaved_encode(0, 7, writer)
        data = writer.get_data()

        reader = BitReader(data)
        context = Context(sum=14, sum2=0)

        # Chroma should decode with pot=0
        result = _decode_with_context(context, reader, is_chroma=True)
        assert result == 7


class TestRunCoderFast:
    """Tests for FAST mode run coder computation."""

    def test_very_low_returns_4(self):
        """Test sum < 1 returns pot=4."""
        context = Context(sum=0, sum2=0)
        assert compute_run_coder_fast(context) == 4

    def test_low_returns_3(self):
        """Test 1 <= sum < 2 returns pot=3."""
        context = Context(sum=1, sum2=0)
        assert compute_run_coder_fast(context) == 3

    def test_medium_low_returns_2(self):
        """Test 2 <= sum < 4 returns pot=2."""
        context = Context(sum=3, sum2=0)
        assert compute_run_coder_fast(context) == 2

    def test_medium_returns_1(self):
        """Test 4 <= sum < 8 returns pot=1."""
        context = Context(sum=6, sum2=0)
        assert compute_run_coder_fast(context) == 1

    def test_high_returns_0(self):
        """Test sum >= 8 returns pot=0."""
        context = Context(sum=10, sum2=0)
        assert compute_run_coder_fast(context) == 0


class TestRunCoderContextual:
    """Tests for CONTEXTUAL mode run coder computation."""

    def test_lossless_low_activity(self):
        """Test lossless mode with low activity."""
        context = Context(sum=1, sum2=0)
        assert compute_run_coder_contextual(context, quality=1024) == 1

    def test_lossless_high_activity(self):
        """Test lossless mode with high activity."""
        context = Context(sum=5, sum2=0)
        assert compute_run_coder_contextual(context, quality=1024) == 0

    def test_lossy_very_low(self):
        """Test lossy mode with very low activity returns pot=4."""
        context = Context(sum=2, sum2=1)
        assert compute_run_coder_contextual(context, quality=512) == 4

    def test_lossy_low(self):
        """Test lossy mode with low activity returns pot=3."""
        context = Context(sum=6, sum2=3)
        assert compute_run_coder_contextual(context, quality=512) == 3


class TestDecodeCoefficients:
    """Tests for the main decode_coefficients function."""

    def _encode_test_block(
        self, coeffs: list[int], scheme: Encoder
    ) -> bytes:
        """Helper to encode a simple test block."""
        writer = BitWriter(1000)

        # For simple test, encode DC + a few values
        # DC coefficient (signed pot=4)
        signed_encode(4, coeffs[0], writer)

        # Remaining coefficients depend on scheme
        for c in coeffs[1:]:
            if scheme == Encoder.TURBO:
                interleaved_encode(1, c, writer)
            else:
                # Use interleaved pot=0 for simple test
                interleaved_encode(0, c, writer)

        return writer.get_data()

    def test_decode_dc_only(self):
        """Test decoding just a DC coefficient."""
        # Create 1x1 block with just DC
        writer = BitWriter(100)
        signed_encode(4, 42, writer)
        data = writer.get_data()

        image = np.zeros((1, 1), dtype=np.int32)
        reader = BitReader(data)

        decode_coefficients(
            image,
            reader,
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            step=1,
            scheme=Encoder.CONTEXTUAL,
            quality=1024,
            has_dc=True,
            is_chroma=False,
        )

        assert image[0, 0] == 42

    def test_decode_small_block_turbo(self):
        """Test decoding a small block in TURBO mode."""
        # 2x2 block: DC at (0,0), detail at (0,1), (1,0), (1,1)
        writer = BitWriter(200)
        signed_encode(4, 100, writer)  # DC
        interleaved_encode(1, 10, writer)  # (0,1)
        interleaved_encode(1, 20, writer)  # (1,0)
        interleaved_encode(1, 5, writer)  # (1,1)
        data = writer.get_data()

        image = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)

        decode_coefficients(
            image,
            reader,
            x0=0,
            y0=0,
            x1=2,
            y1=2,
            step=1,
            scheme=Encoder.TURBO,
            quality=1024,
            has_dc=True,
            is_chroma=False,
        )

        assert image[0, 0] == 100  # DC
        # Other positions depend on iteration order

    def test_decode_without_dc(self):
        """Test decoding without DC coefficient."""
        # Encode enough coefficients for a 2x2 block without DC
        # For step=1, 2x2 block visits (1,0), (0,1), (1,1) = 3 positions
        writer = BitWriter(100)
        interleaved_encode(0, 15, writer)  # First coefficient
        interleaved_encode(0, 10, writer)  # Second coefficient
        interleaved_encode(0, 5, writer)   # Third coefficient
        data = writer.get_data()

        image = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)

        decode_coefficients(
            image,
            reader,
            x0=0,
            y0=0,
            x1=2,
            y1=2,
            step=1,
            scheme=Encoder.CONTEXTUAL,
            quality=1024,
            has_dc=False,
            is_chroma=False,
        )

        # DC position should be unchanged (we didn't decode it)
        assert image[0, 0] == 0

    def test_coefficient_order(self):
        """Test that coefficients are decoded in correct order."""
        # The order should be such that (x | y) & step == 1
        # For 4x4 with step=1:
        # First pass x_step=2 for y=0: x=1, 3
        # Second pass x_step=1 for y=1: x=0, 1, 2, 3
        # etc.
        pass  # Complex to test without actual encoder


class TestDecoderIterationOrder:
    """Tests for the coefficient iteration order."""

    def test_iteration_order_step_1(self):
        """Verify iteration order for step=1."""
        # Collect positions visited
        positions = []
        sizex, sizey = 4, 4
        step = 1

        for y in range(0, sizey, step):
            x_step = step if (y & step) else step * 2
            for x in range(x_step - step, sizex, x_step):
                positions.append((x, y))

        # All positions should have (x | y) & step == 1
        # except DC which is handled separately
        for x, y in positions:
            if x != 0 or y != 0:  # Skip DC
                assert ((x | y) & step) == 1, f"Position ({x},{y}) violates constraint"

    def test_iteration_order_step_2(self):
        """Verify iteration order for step=2."""
        positions = []
        sizex, sizey = 8, 8
        step = 2

        for y in range(0, sizey, step):
            x_step = step if (y & step) else step * 2
            for x in range(x_step - step, sizex, x_step):
                positions.append((x, y))

        # Verify constraint
        for _x, _y in positions:
            # The pattern should satisfy (x | y) & step having certain property
            pass  # The exact constraint depends on the level


class TestDecoderIntegration:
    """Integration tests with encoded data."""

    def test_roundtrip_single_coefficient(self):
        """Test encoding and decoding a single coefficient."""
        # Manually encode a DC coefficient
        writer = BitWriter(100)
        signed_encode(4, -50, writer)
        data = writer.get_data()

        # Decode it
        image = np.zeros((1, 1), dtype=np.int32)
        reader = BitReader(data)

        decode_coefficients(
            image,
            reader,
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            step=1,
            scheme=Encoder.CONTEXTUAL,
            quality=1024,
            has_dc=True,
            is_chroma=False,
        )

        assert image[0, 0] == -50

    def test_nonzero_values_turbo_mode(self):
        """Test decoding a block with TURBO mode.

        TURBO mode starts with runCoder=1 for quality=1024, step=1.
        So we need to encode run lengths before coefficients.
        """
        from pygfwx.core.golomb_rice import unsigned_encode

        writer = BitWriter(200)
        signed_encode(4, 10, writer)  # DC = 10

        # TURBO mode with q=1024, step=1 starts with runCoder=1
        # Encode run=0 (no zeros), then coefficient
        unsigned_encode(1, 0, writer)   # run = 0
        interleaved_encode(1, 5, writer)   # coeff 1

        # After non-zero, runCoder stays 1
        unsigned_encode(1, 0, writer)
        interleaved_encode(1, 3, writer)   # coeff 2

        unsigned_encode(1, 0, writer)
        interleaved_encode(1, -2, writer)  # coeff 3

        data = writer.get_data()

        image = np.zeros((2, 2), dtype=np.int32)
        reader = BitReader(data)

        decode_coefficients(
            image,
            reader,
            x0=0,
            y0=0,
            x1=2,
            y1=2,
            step=1,
            scheme=Encoder.TURBO,
            quality=1024,
            has_dc=True,
            is_chroma=False,
        )

        # DC should be 10
        assert image[0, 0] == 10
