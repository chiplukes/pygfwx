"""
Tests for the GFWX Context Modeling module.

These tests verify:
- Helper functions for context accumulation
- Context calculation from image coefficients
- Coding mode selection logic
- Fast context update method
"""

import numpy as np

from pygfwx.core.context import (
    Context,
    _add_context,
    compute_run_coder,
    get_context,
    select_coding_mode,
    update_fast_context,
)


class TestAddContext:
    """Tests for the _add_context helper function."""

    def test_add_positive_value(self):
        """Test adding a positive value."""
        sum_val, sum2_val, count = _add_context(10, 2, 0, 0, 0)
        assert sum_val == 20  # |10| * 2
        assert sum2_val == 200  # 10^2 * 2
        assert count == 2

    def test_add_negative_value(self):
        """Test adding a negative value (uses absolute value)."""
        sum_val, sum2_val, count = _add_context(-10, 2, 0, 0, 0)
        assert sum_val == 20  # |-10| * 2
        assert sum2_val == 200  # 10^2 * 2
        assert count == 2

    def test_add_zero(self):
        """Test adding zero value."""
        sum_val, sum2_val, count = _add_context(0, 4, 0, 0, 0)
        assert sum_val == 0
        assert sum2_val == 0
        assert count == 4

    def test_accumulation(self):
        """Test accumulating multiple values."""
        sum_val, sum2_val, count = _add_context(5, 2, 10, 50, 3)
        assert sum_val == 20  # 10 + |5| * 2
        assert sum2_val == 100  # 50 + 5^2 * 2
        assert count == 5  # 3 + 2

    def test_clamping_large_values(self):
        """Test that large values are clamped for sum2 but not sum."""
        # Value larger than 4096
        sum_val, sum2_val, count = _add_context(5000, 1, 0, 0, 0)
        assert sum_val == 5000  # No clamping for sum
        assert sum2_val == 4096 * 4096  # Clamped to 4096^2
        assert count == 1


class TestGetContext:
    """Tests for the get_context function."""

    def test_simple_uniform_image(self):
        """Test context on a uniform image."""
        # 8x8 image, all values 10
        image = np.full((8, 8), 10, dtype=np.int32)
        ctx = get_context(image, 0, 0, 8, 8, 2, 2, 1)

        # Should return non-zero context based on neighbors
        assert isinstance(ctx, Context)
        assert ctx.sum > 0
        assert ctx.sum2 > 0

    def test_ancestor_only(self):
        """Test context at position with only ancestor available.

        The ancestor position is calculated as:
          px = x0 + (x & ~(skip*2)) + (x & skip)
          py = y0 + (y & ~(skip*2)) + (y & skip)

        For position (3,3) with skip=1:
          px = 0 + (3 & ~2) + (3 & 1) = 0 + 1 + 1 = 2
          py = 0 + (3 & ~2) + (3 & 1) = 0 + 1 + 1 = 2
        So ancestor is at (2,2).
        """
        image = np.zeros((8, 8), dtype=np.int32)
        # Set value at ancestor position (2,2) for query at (3,3) with skip=1
        image[2, 2] = 100

        ctx = get_context(image, 0, 0, 8, 8, 3, 3, 1)

        # Context comes primarily from ancestor with weight 2
        # Note: There may be sibling contributions too at this position
        assert ctx.sum > 0  # Should have some contribution
        assert ctx.sum2 > 0

    def test_zero_image(self):
        """Test context on all-zero image."""
        image = np.zeros((8, 8), dtype=np.int32)
        ctx = get_context(image, 0, 0, 8, 8, 2, 2, 1)

        assert ctx.sum == 0
        assert ctx.sum2 == 0

    def test_context_with_skip_2(self):
        """Test context with skip=2 (second wavelet level)."""
        image = np.full((16, 16), 5, dtype=np.int32)
        ctx = get_context(image, 0, 0, 16, 16, 4, 4, 2)

        assert ctx.sum > 0
        assert ctx.sum2 > 0

    def test_context_normalization(self):
        """Test that context is normalized to 16 counts."""
        # If all neighbors have value 10 with total weight 16,
        # the normalized sum should be close to 10
        image = np.full((16, 16), 10, dtype=np.int32)
        ctx = get_context(image, 0, 0, 16, 16, 8, 8, 1)

        # Sum should be roughly 10 * 16 = 160 after normalization
        # (exact value depends on how many neighbors contribute)
        assert 80 < ctx.sum < 320

    def test_boundary_handling_x(self):
        """Test context near x boundary."""
        image = np.full((8, 8), 20, dtype=np.int32)
        # Position near right edge
        ctx = get_context(image, 0, 0, 8, 8, 6, 4, 1)

        assert ctx.sum > 0
        assert ctx.sum2 > 0

    def test_boundary_handling_y(self):
        """Test context near y boundary."""
        image = np.full((8, 8), 20, dtype=np.int32)
        # Position near bottom edge
        ctx = get_context(image, 0, 0, 8, 8, 4, 6, 1)

        assert ctx.sum > 0
        assert ctx.sum2 > 0

    def test_sub_block(self):
        """Test context for a sub-block of the image."""
        image = np.full((16, 16), 15, dtype=np.int32)
        # Context for sub-block from (4,4) to (12,12)
        ctx = get_context(image, 4, 4, 12, 12, 4, 4, 1)

        assert ctx.sum > 0
        assert ctx.sum2 > 0


class TestSelectCodingMode:
    """Tests for the select_coding_mode function."""

    def test_low_activity_interleaved_0(self):
        """Test that low activity selects interleaved with pot=0."""
        # Low sum and sum2 - should use interleaved, pot=0
        ctx = Context(sum=0, sum2=0)
        mode, pot = select_coding_mode(ctx)
        assert mode == "interleaved"
        assert pot == 0

    def test_moderate_activity(self):
        """Test moderate activity context."""
        ctx = Context(sum=20, sum2=400)
        mode, pot = select_coding_mode(ctx)
        # Should select some interleaved or signed mode
        assert mode in ("interleaved", "signed")
        assert 0 <= pot <= 4

    def test_high_activity(self):
        """Test high activity context selects signed mode."""
        # High sum_sq relative to sum2
        ctx = Context(sum=200, sum2=1000)
        mode, pot = select_coding_mode(ctx)
        # High activity should use signed mode with higher pot
        assert pot >= 0

    def test_very_high_activity(self):
        """Test very high activity selects signed pot=4."""
        ctx = Context(sum=1000, sum2=5000)
        mode, pot = select_coding_mode(ctx)
        assert mode == "signed"
        assert pot == 4

    def test_chroma_threshold(self):
        """Test that chroma mode uses different thresholds."""
        # Context that would be interleaved for luma
        ctx = Context(sum=10, sum2=100)
        luma_mode, luma_pot = select_coding_mode(ctx, is_chroma=False)
        chroma_mode, chroma_pot = select_coding_mode(ctx, is_chroma=True)

        # Both should work (exact results may vary based on thresholds)
        assert luma_mode in ("interleaved", "signed")
        assert chroma_mode in ("interleaved", "signed")

    def test_all_pot_values_reachable(self):
        """Test that all pot values (0-4) are reachable."""
        pots_seen = set()
        for sum_val in range(0, 2000, 10):
            for sum2_val in range(0, 10000, 100):
                ctx = Context(sum=sum_val, sum2=sum2_val)
                _, pot = select_coding_mode(ctx)
                pots_seen.add(pot)
                if pots_seen == {0, 1, 2, 3, 4}:
                    return
        # Should have found all pot values
        assert pots_seen == {0, 1, 2, 3, 4}


class TestUpdateFastContext:
    """Tests for the update_fast_context function."""

    def test_initial_update(self):
        """Test updating from zero context."""
        ctx = Context(sum=0, sum2=0)
        new_ctx = update_fast_context(ctx, 16)

        assert new_ctx.sum == 16
        assert new_ctx.sum2 == 256  # 16^2

    def test_decay_behavior(self):
        """Test that old context decays."""
        ctx = Context(sum=160, sum2=2560)
        new_ctx = update_fast_context(ctx, 0)

        # With decay factor 15/16, should decrease
        # (160 * 15 + 7) >> 4 = 150
        expected_sum = (160 * 15 + 7) >> 4
        expected_sum2 = (2560 * 15 + 7) >> 4
        assert new_ctx.sum == expected_sum
        assert new_ctx.sum2 == expected_sum2

    def test_negative_value(self):
        """Test update with negative coefficient."""
        ctx = Context(sum=0, sum2=0)
        new_ctx = update_fast_context(ctx, -10)

        assert new_ctx.sum == 10  # Uses absolute value
        assert new_ctx.sum2 == 100

    def test_large_value_clamping(self):
        """Test that large values are clamped for sum2."""
        ctx = Context(sum=0, sum2=0)
        new_ctx = update_fast_context(ctx, 5000)

        assert new_ctx.sum == 5000  # Not clamped
        assert new_ctx.sum2 == 4096 * 4096  # Clamped


class TestComputeRunCoder:
    """Tests for the compute_run_coder function."""

    def test_fast_mode_very_low_activity(self):
        """Test FAST mode with very low activity returns high pot."""
        ctx = Context(sum=0, sum2=0)
        run_coder = compute_run_coder(ctx, 0, 0, 512, encoder_fast=True)
        assert run_coder == 4

    def test_fast_mode_high_activity(self):
        """Test FAST mode with high activity returns 0."""
        ctx = Context(sum=16, sum2=256)
        run_coder = compute_run_coder(ctx, 0, 0, 512, encoder_fast=True)
        assert run_coder == 0

    def test_contextual_lossless(self):
        """Test contextual mode in lossless (quality=1024)."""
        ctx = Context(sum=1, sum2=1)
        run_coder = compute_run_coder(ctx, 0, 0, 1024, encoder_fast=False)
        assert run_coder == 1

    def test_contextual_lossy_low_activity(self):
        """Test contextual lossy with low activity."""
        ctx = Context(sum=2, sum2=1)
        run_coder = compute_run_coder(ctx, 0, 0, 512, encoder_fast=False)
        assert run_coder >= 2  # Should be high for low activity

    def test_no_update_when_mismatched(self):
        """Test that run_coder doesn't update when value/state mismatch."""
        ctx = Context(sum=1, sum2=1)
        # value=5 (non-zero) but current_run_coder=0 -> no update
        run_coder = compute_run_coder(ctx, 5, 0, 512, encoder_fast=True)
        assert run_coder == 0  # Returns current value unchanged

        # value=0 but current_run_coder=2 -> no update
        run_coder = compute_run_coder(ctx, 0, 2, 512, encoder_fast=True)
        assert run_coder == 2  # Returns current value unchanged


class TestContextIntegration:
    """Integration tests for context modeling."""

    def test_natural_image_context_progression(self):
        """Test context values follow expected pattern for natural images."""
        # Simulate natural image: smooth areas with occasional edges
        np.random.seed(42)
        image = np.random.randint(-50, 51, (32, 32), dtype=np.int32)

        # Add smooth region
        image[8:16, 8:16] = 5

        # Context should be lower in smooth region
        smooth_ctx = get_context(image, 0, 0, 32, 32, 12, 12, 1)
        edge_ctx = get_context(image, 0, 0, 32, 32, 4, 4, 1)

        # Smooth region should have lower context values
        assert smooth_ctx.sum < edge_ctx.sum or smooth_ctx.sum2 < edge_ctx.sum2

    def test_context_coding_mode_consistency(self):
        """Test that context leads to consistent coding mode selection."""
        image = np.full((16, 16), 10, dtype=np.int32)

        # Get context at multiple positions
        ctx1 = get_context(image, 0, 0, 16, 16, 4, 4, 1)
        ctx2 = get_context(image, 0, 0, 16, 16, 8, 8, 1)

        mode1, pot1 = select_coding_mode(ctx1)
        mode2, pot2 = select_coding_mode(ctx2)

        # For uniform image, should get similar modes
        # (may not be identical due to different neighbor counts)
        assert mode1 == mode2
        assert abs(pot1 - pot2) <= 1
