"""
Basic sanity tests to verify test infrastructure works.
"""

import numpy as np
import pytest

from pygfwx.utils.reference_images import (
    create_impulse_image,
    create_reference_image,
    create_uniform_image,
)


class TestInfrastructure:
    """Tests to verify the testing infrastructure is working."""

    def test_numpy_import(self):
        """Verify numpy is available."""
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6

    def test_pygfwx_import(self):
        """Verify pygfwx package can be imported."""
        import pygfwx

        assert hasattr(pygfwx, "__version__")
        assert pygfwx.__version__ == "0.1.0"


class TestReferenceImages:
    """Tests for reference image generators."""

    def test_reference_image_mono(self):
        """Test primary reference image generation (mono)."""
        img = create_reference_image(size=64, channels=1, bit_depth=8)
        assert img.shape == (64, 64)
        assert img.dtype == np.uint8
        # Should have variety (not uniform)
        assert img.min() != img.max()

    def test_reference_image_rgb(self):
        """Test primary reference image generation (RGB)."""
        img = create_reference_image(size=64, channels=3, bit_depth=8)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        # Each channel should be different
        assert not np.array_equal(img[:, :, 0], img[:, :, 1])
        assert not np.array_equal(img[:, :, 1], img[:, :, 2])

    def test_reference_image_16bit(self):
        """Test 16-bit reference image."""
        img = create_reference_image(size=64, channels=1, bit_depth=16)
        assert img.shape == (64, 64)
        assert img.dtype == np.uint16
        # Should use extended range
        assert img.max() > 255

    def test_reference_image_has_frequency_content(self):
        """Verify reference image has both high and low frequency content."""
        img = create_reference_image(size=64, channels=1, bit_depth=8).astype(np.float64)

        # Check quadrants have expected characteristics
        half = 32

        # Top-left should be smooth (low frequency) - low variance
        tl = img[:half, :half]
        tl_var = np.var(tl)

        # Top-right should have stripes (high H frequency) - high variance
        tr = img[:half, half:]
        tr_var = np.var(tr)

        # Bottom-left should have stripes (high V frequency) - high variance
        bl = img[half:, :half]
        bl_var = np.var(bl)

        # The striped regions should have higher variance than smooth region
        assert tr_var > tl_var, "Top-right should have higher variance than top-left"
        assert bl_var > tl_var, "Bottom-left should have higher variance than top-left"

    def test_reference_image_reproducible(self):
        """Reference image should be deterministic."""
        img1 = create_reference_image(size=64, channels=1, bit_depth=8)
        img2 = create_reference_image(size=64, channels=1, bit_depth=8)
        assert np.array_equal(img1, img2)

    def test_uniform_image(self):
        """Test uniform image generation."""
        img = create_uniform_image(size=64, value=128, channels=1, bit_depth=8)
        assert img.shape == (64, 64)
        assert np.all(img == 128)

    def test_uniform_image_rgb(self):
        """Test uniform RGB image generation."""
        img = create_uniform_image(size=64, value=100, channels=3, bit_depth=8)
        assert img.shape == (64, 64, 3)
        assert np.all(img == 100)

    def test_impulse_image(self):
        """Test impulse image generation."""
        img = create_impulse_image(size=64, channels=1, bit_depth=8)
        assert img.shape == (64, 64)
        assert img[32, 32] == 255
        assert img.sum() == 255  # Only one bright pixel


class TestFixtures:
    """Tests for pytest fixtures using reference images."""

    def test_reference_image_fixture(self, reference_image):
        """Test the primary reference image fixture."""
        assert reference_image.shape == (64, 64)
        assert reference_image.dtype == np.uint8

    def test_reference_image_rgb_fixture(self, reference_image_rgb):
        """Test the RGB reference image fixture."""
        assert reference_image_rgb.shape == (64, 64, 3)
        assert reference_image_rgb.dtype == np.uint8

    def test_uniform_image_fixture(self, uniform_image):
        """Test uniform image fixture."""
        assert uniform_image.shape == (64, 64)
        assert np.all(uniform_image == 128)

    def test_impulse_image_fixture(self, impulse_image):
        """Test impulse image fixture."""
        assert impulse_image.shape == (64, 64)
        assert impulse_image[32, 32] == 255


class TestUtilityFunctions:
    """Tests for utility functions in test_utils.py."""

    def test_arrays_equal(self):
        """Test arrays_equal function."""
        from tests.test_utils import arrays_equal

        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        c = np.array([1, 2, 4])

        assert arrays_equal(a, b)
        assert not arrays_equal(a, c)

    def test_max_abs_diff(self):
        """Test max_abs_diff function."""
        from tests.test_utils import max_abs_diff

        a = np.array([1, 2, 3])
        b = np.array([1, 5, 3])

        assert max_abs_diff(a, b) == 3.0

    def test_diff_count(self):
        """Test diff_count function."""
        from tests.test_utils import diff_count

        a = np.array([1, 2, 3, 4])
        b = np.array([1, 5, 3, 6])

        assert diff_count(a, b) == 2

    def test_hex_dump(self):
        """Test hex_dump function."""
        from tests.test_utils import hex_dump

        data = b"GFWX\x01\x00\x00\x00"
        dump = hex_dump(data)
        assert "47 46 57 58" in dump  # "GFWX" in hex
        assert "GFWX" in dump  # ASCII representation

    def test_find_first_diff(self):
        """Test find_first_diff function."""
        from tests.test_utils import find_first_diff

        a = b"GFWX\x01\x00"
        b_same = b"GFWX\x01\x00"
        b_diff = b"GFWX\x02\x00"

        assert find_first_diff(a, b_same) is None
        assert find_first_diff(a, b_diff) == 4


@pytest.mark.sdk
class TestSDKMarker:
    """Tests marked as requiring SDK (will be skipped if SDK not available)."""

    def test_sdk_required(self, sdk_wrapper):
        """This test requires the SDK and will be skipped if not available."""
        # This test will be skipped until SDK wrapper is implemented
        assert sdk_wrapper is not None
