"""
Tests for the GFWX Wavelet Lifting module.

These tests verify:
- Helper functions (round_fraction, median, cubic)
- Forward transform (lift)
- Inverse transform (unlift)
- Roundtrip (lift then unlift recovers original)
- Both filter types (LINEAR and CUBIC)
"""

import numpy as np
import pytest

from pygfwx.core.header import Filter
from pygfwx.core.lifting import (
    _cubic,
    _median,
    _round_fraction,
    lift,
    lift_full,
    unlift,
    unlift_full,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_round_fraction_positive(self):
        """Test rounding positive fractions."""
        assert _round_fraction(7, 4) == 2  # 1.75 -> 2
        assert _round_fraction(5, 4) == 1  # 1.25 -> 1
        assert _round_fraction(6, 4) == 2  # 1.5 -> 2 (away from zero)

    def test_round_fraction_negative(self):
        """Test rounding negative fractions (C++ truncate-toward-zero behavior)."""
        # C++ uses truncation toward zero: (num - denom/2) / denom
        # -7 - 2 = -9, -9 / 4 = -2 (truncated toward zero)
        assert _round_fraction(-7, 4) == -2
        # -5 - 2 = -7, -7 / 4 = -1 (truncated toward zero)
        assert _round_fraction(-5, 4) == -1
        # -6 - 2 = -8, -8 / 4 = -2
        assert _round_fraction(-6, 4) == -2

    def test_round_fraction_exact(self):
        """Test exact division."""
        assert _round_fraction(8, 4) == 2
        # -8 - 2 = -10, -10 / 4 = -2 (truncated toward zero)
        assert _round_fraction(-8, 4) == -2

    def test_median_ordered(self):
        """Test median with ordered inputs."""
        assert _median(1, 2, 3) == 2
        assert _median(3, 2, 1) == 2

    def test_median_duplicates(self):
        """Test median with duplicate values."""
        assert _median(1, 1, 2) == 1
        assert _median(1, 2, 2) == 2
        assert _median(5, 5, 5) == 5

    def test_median_various(self):
        """Test median with various orderings."""
        assert _median(1, 3, 2) == 2
        assert _median(2, 1, 3) == 2
        assert _median(2, 3, 1) == 2
        assert _median(3, 1, 2) == 2

    def test_cubic_center_values(self):
        """Test cubic interpolation with center values."""
        # When c1=c2, result should be close to c1=c2
        result = _cubic(10, 20, 20, 10)
        assert result == 20  # median clamps to c1=c2

    def test_cubic_clamping(self):
        """Test that cubic result is clamped by median."""
        # Extreme values that would overshoot
        result = _cubic(100, 10, 20, 100)
        # Should be clamped to [10, 20]
        assert 10 <= result <= 20

    def test_cubic_linear_region(self):
        """Test cubic in smoothly varying region."""
        # Linearly increasing: 0, 10, 20, 30
        result = _cubic(0, 10, 20, 30)
        # Interpolation should give ~15, clamped to [10, 20]
        assert 10 <= result <= 20


class TestLiftLinear:
    """Tests for forward lifting with LINEAR filter."""

    def test_lift_small_uniform(self):
        """Test lifting a small uniform image."""
        image = np.full((4, 4), 100, dtype=np.int32)
        lift(image, 0, 0, 4, 4, 1, Filter.LINEAR)

        # DC coefficient should preserve energy
        # Other coefficients should be small for uniform image
        assert image[0, 0] != 0  # DC preserved
        # Detail coefficients should be zero for uniform
        # (may not be exactly zero due to boundary effects)

    def test_lift_preserves_dtype(self):
        """Test that lifting preserves array dtype."""
        image = np.array([[10, 20], [30, 40]], dtype=np.int32)
        lift(image, 0, 0, 2, 2, 1, Filter.LINEAR)
        assert image.dtype == np.int32

    def test_lift_with_step_2(self):
        """Test lifting starting from step=2."""
        image = np.arange(16, dtype=np.int32).reshape(4, 4)
        lift(image, 0, 0, 4, 4, 2, Filter.LINEAR)
        # Should only do one level of transform
        # since step starts at 2

    def test_lift_subregion(self):
        """Test lifting a subregion of an image."""
        image = np.arange(64, dtype=np.int32).reshape(8, 8)
        original = image.copy()
        # Lift only center 4x4 region
        lift(image, 2, 2, 6, 6, 1, Filter.LINEAR)

        # Corners should be unchanged
        assert np.array_equal(image[0:2, :], original[0:2, :])
        assert np.array_equal(image[6:8, :], original[6:8, :])


class TestUnliftLinear:
    """Tests for inverse lifting with LINEAR filter."""

    def test_unlift_small(self):
        """Test unlifting a small image."""
        # Create wavelet coefficients
        image = np.array([[100, 10], [5, 2]], dtype=np.int32)
        unlift(image, 0, 0, 2, 2, 1, Filter.LINEAR)
        # Should produce a valid image

    def test_unlift_preserves_dtype(self):
        """Test that unlifting preserves array dtype."""
        image = np.array([[100, 10], [5, 2]], dtype=np.int32)
        unlift(image, 0, 0, 2, 2, 1, Filter.LINEAR)
        assert image.dtype == np.int32


class TestRoundtripLinear:
    """Roundtrip tests for LINEAR filter (should be lossless)."""

    def test_roundtrip_2x2(self):
        """Test lift/unlift roundtrip on 2x2 image."""
        original = np.array([[10, 20], [30, 40]], dtype=np.int32)
        image = original.copy()

        lift(image, 0, 0, 2, 2, 1, Filter.LINEAR)
        unlift(image, 0, 0, 2, 2, 1, Filter.LINEAR)

        assert np.array_equal(image, original)

    def test_roundtrip_4x4(self):
        """Test lift/unlift roundtrip on 4x4 image."""
        original = np.arange(16, dtype=np.int32).reshape(4, 4)
        image = original.copy()

        lift(image, 0, 0, 4, 4, 1, Filter.LINEAR)
        unlift(image, 0, 0, 4, 4, 1, Filter.LINEAR)

        assert np.array_equal(image, original)

    def test_roundtrip_8x8(self):
        """Test lift/unlift roundtrip on 8x8 image."""
        original = np.arange(64, dtype=np.int32).reshape(8, 8)
        image = original.copy()

        lift(image, 0, 0, 8, 8, 1, Filter.LINEAR)
        unlift(image, 0, 0, 8, 8, 1, Filter.LINEAR)

        assert np.array_equal(image, original)

    def test_roundtrip_non_power_of_2(self):
        """Test roundtrip on non-power-of-2 dimensions."""
        original = np.arange(30, dtype=np.int32).reshape(5, 6)
        image = original.copy()

        lift(image, 0, 0, 6, 5, 1, Filter.LINEAR)
        unlift(image, 0, 0, 6, 5, 1, Filter.LINEAR)

        assert np.array_equal(image, original)

    def test_roundtrip_random(self):
        """Test roundtrip on random data."""
        np.random.seed(42)
        original = np.random.randint(-1000, 1000, (16, 16), dtype=np.int32)
        image = original.copy()

        lift(image, 0, 0, 16, 16, 1, Filter.LINEAR)
        unlift(image, 0, 0, 16, 16, 1, Filter.LINEAR)

        assert np.array_equal(image, original)

    def test_roundtrip_negative_values(self):
        """Test roundtrip with negative values."""
        original = np.array([[-100, 50], [25, -75]], dtype=np.int32)
        image = original.copy()

        lift(image, 0, 0, 2, 2, 1, Filter.LINEAR)
        unlift(image, 0, 0, 2, 2, 1, Filter.LINEAR)

        assert np.array_equal(image, original)


class TestLiftCubic:
    """Tests for forward lifting with CUBIC filter."""

    def test_lift_cubic_small(self):
        """Test cubic lifting on small image."""
        image = np.arange(16, dtype=np.int32).reshape(4, 4)
        lift(image, 0, 0, 4, 4, 1, Filter.CUBIC)
        # Should complete without error

    def test_lift_cubic_preserves_dtype(self):
        """Test that cubic lifting preserves dtype."""
        image = np.array([[10, 20], [30, 40]], dtype=np.int32)
        lift(image, 0, 0, 2, 2, 1, Filter.CUBIC)
        assert image.dtype == np.int32


class TestRoundtripCubic:
    """Roundtrip tests for CUBIC filter (may have small rounding errors)."""

    def test_roundtrip_cubic_2x2(self):
        """Test cubic roundtrip on 2x2 (may have rounding)."""
        original = np.array([[10, 20], [30, 40]], dtype=np.int32)
        image = original.copy()

        lift(image, 0, 0, 2, 2, 1, Filter.CUBIC)
        unlift(image, 0, 0, 2, 2, 1, Filter.CUBIC)

        # Cubic may have small rounding errors
        diff = np.abs(image - original)
        assert np.max(diff) <= 1  # Allow ±1 rounding error

    def test_roundtrip_cubic_8x8(self):
        """Test cubic roundtrip on 8x8."""
        original = np.arange(64, dtype=np.int32).reshape(8, 8)
        image = original.copy()

        lift(image, 0, 0, 8, 8, 1, Filter.CUBIC)
        unlift(image, 0, 0, 8, 8, 1, Filter.CUBIC)

        # Allow small rounding errors
        diff = np.abs(image - original)
        assert np.max(diff) <= 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_lift_full(self):
        """Test lift_full convenience function."""
        original = np.arange(16, dtype=np.int32).reshape(4, 4)
        image = original.copy()

        lift_full(image, Filter.LINEAR)
        # Should be transformed (not equal to original)
        assert not np.array_equal(image, original)

    def test_unlift_full(self):
        """Test unlift_full convenience function."""
        original = np.arange(16, dtype=np.int32).reshape(4, 4)
        image = original.copy()

        lift_full(image, Filter.LINEAR)
        unlift_full(image, Filter.LINEAR)

        assert np.array_equal(image, original)


class TestEnergyCompaction:
    """Tests for wavelet energy compaction properties."""

    def test_uniform_image_energy(self):
        """Test that uniform images have energy in DC."""
        image = np.full((8, 8), 100, dtype=np.int32)
        lift_full(image, Filter.LINEAR)

        # DC should contain most energy
        dc_energy = image[0, 0] ** 2
        total_energy = np.sum(image**2)

        # DC should be significant portion
        assert dc_energy > total_energy * 0.5

    def test_smooth_gradient_compaction(self):
        """Test that smooth gradients produce sparse wavelet coefficients."""
        image = np.zeros((8, 8), dtype=np.int32)
        for y in range(8):
            for x in range(8):
                image[y, x] = x + y  # Smooth gradient

        lift_full(image, Filter.LINEAR)

        # For a smooth gradient, most coefficients should be zero or small.
        # Count how many coefficients are exactly zero.
        zero_count = np.sum(image == 0)

        # A smooth gradient should have many zero coefficients
        # (the wavelet transform is efficient for smooth data)
        assert zero_count > 32  # At least half should be zero


class TestIntegrationWithSDK:
    """Integration tests that could be validated against SDK."""

    @pytest.mark.skip(reason="Requires SDK integration for validation")
    def test_matches_sdk_linear(self):
        """Test that our lifting matches SDK output."""
        pass

    @pytest.mark.skip(reason="Requires SDK integration for validation")
    def test_matches_sdk_cubic(self):
        """Test that our cubic lifting matches SDK output."""
        pass
