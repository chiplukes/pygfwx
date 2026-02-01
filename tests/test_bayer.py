"""
Tests for GFWX Bayer mode support.

Tests cover:
- Bayer pattern utilities (pattern detection, sub-image iteration)
- Bayer test image generation
- Quality selection for sub-images
- Integration with block decoder (when SDK is available)
"""

import numpy as np
import pytest

from pygfwx.core.bayer import (
    BayerPattern,
    BayerSubImage,
    create_bayer_test_image,
    create_uniform_bayer_image,
    extract_bayer_subimage,
    get_bayer_pattern,
    get_bayer_sub_images,
    get_quality_for_subimage,
    insert_bayer_subimage,
    intent_is_bayer,
    is_chroma_subimage,
    iter_bayer_offsets,
    iter_bayer_offsets_for_lifting,
    validate_bayer_dimensions,
)
from pygfwx.core.header import Intent


class TestIntentIsBayer:
    """Test Bayer intent detection."""

    def test_bayer_intents_detected(self):
        """All Bayer intents should be detected."""
        assert intent_is_bayer(Intent.BAYER_RGGB)
        assert intent_is_bayer(Intent.BAYER_BGGR)
        assert intent_is_bayer(Intent.BAYER_GRBG)
        assert intent_is_bayer(Intent.BAYER_GBRG)
        assert intent_is_bayer(Intent.BAYER_GENERIC)

    def test_non_bayer_intents_not_detected(self):
        """Non-Bayer intents should not be detected."""
        assert not intent_is_bayer(Intent.GENERIC)
        assert not intent_is_bayer(Intent.MONO)
        assert not intent_is_bayer(Intent.RGB)
        assert not intent_is_bayer(Intent.RGBA)
        assert not intent_is_bayer(Intent.BGR)
        assert not intent_is_bayer(Intent.BGRA)

    def test_integer_values(self):
        """Test with raw integer values."""
        assert intent_is_bayer(2)  # BAYER_RGGB
        assert intent_is_bayer(6)  # BAYER_GENERIC
        assert not intent_is_bayer(1)  # MONO
        assert not intent_is_bayer(7)  # RGB


class TestGetBayerPattern:
    """Test Bayer pattern extraction from intent."""

    def test_valid_patterns(self):
        """Valid Bayer intents should return patterns."""
        assert get_bayer_pattern(Intent.BAYER_RGGB) == BayerPattern.RGGB
        assert get_bayer_pattern(Intent.BAYER_BGGR) == BayerPattern.BGGR
        assert get_bayer_pattern(Intent.BAYER_GRBG) == BayerPattern.GRBG
        assert get_bayer_pattern(Intent.BAYER_GBRG) == BayerPattern.GBRG
        assert get_bayer_pattern(Intent.BAYER_GENERIC) == BayerPattern.GENERIC

    def test_non_bayer_returns_none(self):
        """Non-Bayer intents should return None."""
        assert get_bayer_pattern(Intent.GENERIC) is None
        assert get_bayer_pattern(Intent.RGB) is None
        assert get_bayer_pattern(Intent.MONO) is None


class TestGetBayerSubImages:
    """Test Bayer sub-image information."""

    def test_rggb_sub_images(self):
        """Test RGGB pattern sub-images."""
        subs = get_bayer_sub_images(BayerPattern.RGGB)
        assert len(subs) == 4

        # Find each sub-image by position
        sub_by_pos = {(s.ox, s.oy): s for s in subs}

        # RGGB: R at (0,0), G at (1,0)&(0,1), B at (1,1)
        assert sub_by_pos[(0, 0)].color == "R"
        assert sub_by_pos[(0, 0)].is_chroma is True  # R is chroma

        assert sub_by_pos[(0, 1)].color == "G"
        assert sub_by_pos[(0, 1)].is_chroma is False  # G is luma

        assert sub_by_pos[(1, 0)].color == "G"
        assert sub_by_pos[(1, 0)].is_chroma is False  # G is luma

        assert sub_by_pos[(1, 1)].color == "B"
        assert sub_by_pos[(1, 1)].is_chroma is True  # B is chroma

    def test_grbg_sub_images(self):
        """Test GRBG pattern sub-images."""
        subs = get_bayer_sub_images(BayerPattern.GRBG)
        sub_by_pos = {(s.ox, s.oy): s for s in subs}

        # GRBG: G at (0,0), R at (1,0), B at (0,1), G at (1,1)
        assert sub_by_pos[(0, 0)].color == "G"
        assert sub_by_pos[(0, 0)].is_chroma is False

        assert sub_by_pos[(1, 0)].color == "R"
        assert sub_by_pos[(1, 0)].is_chroma is True

        assert sub_by_pos[(0, 1)].color == "B"
        assert sub_by_pos[(0, 1)].is_chroma is True

        assert sub_by_pos[(1, 1)].color == "G"
        assert sub_by_pos[(1, 1)].is_chroma is False


class TestIterBayerOffsets:
    """Test Bayer offset iteration."""

    def test_all_offsets(self):
        """iter_bayer_offsets should yield all 4 positions."""
        offsets = list(iter_bayer_offsets())
        assert len(offsets) == 4
        assert (0, 0) in offsets
        assert (0, 1) in offsets
        assert (1, 0) in offsets
        assert (1, 1) in offsets

    def test_lifting_offsets(self):
        """iter_bayer_offsets_for_lifting skips (0,0)."""
        offsets = list(iter_bayer_offsets_for_lifting())
        # Should be (0,1), (1,0), (1,1) in that order
        assert len(offsets) == 3
        assert (0, 0) not in offsets
        assert offsets == [(0, 1), (1, 0), (1, 1)]


class TestIsChromaSubimage:
    """Test chroma detection for sub-images."""

    def test_rggb_chroma_positions(self):
        """RGGB: R(0,0) and B(1,1) are chroma."""
        assert is_chroma_subimage(0, 0, BayerPattern.RGGB) is True  # R
        assert is_chroma_subimage(1, 0, BayerPattern.RGGB) is False  # G
        assert is_chroma_subimage(0, 1, BayerPattern.RGGB) is False  # G
        assert is_chroma_subimage(1, 1, BayerPattern.RGGB) is True  # B

    def test_grbg_chroma_positions(self):
        """GRBG: R(1,0) and B(0,1) are chroma."""
        assert is_chroma_subimage(0, 0, BayerPattern.GRBG) is False  # G
        assert is_chroma_subimage(1, 0, BayerPattern.GRBG) is True  # R
        assert is_chroma_subimage(0, 1, BayerPattern.GRBG) is True  # B
        assert is_chroma_subimage(1, 1, BayerPattern.GRBG) is False  # G


class TestGetQualityForSubimage:
    """Test quality selection for sub-images."""

    def test_sdk_logic(self):
        """Test SDK quality selection: (0,0) is luma, others are chroma."""
        luma_q = 1024
        chroma_q = 512

        # (0,0) uses luma quality
        assert get_quality_for_subimage(0, 0, luma_q, chroma_q) == luma_q

        # Others use chroma quality
        assert get_quality_for_subimage(1, 0, luma_q, chroma_q) == chroma_q
        assert get_quality_for_subimage(0, 1, luma_q, chroma_q) == chroma_q
        assert get_quality_for_subimage(1, 1, luma_q, chroma_q) == chroma_q


class TestExtractInsertSubimage:
    """Test sub-image extraction and insertion."""

    def test_extract_subimage(self):
        """Test extracting a Bayer sub-image."""
        # Create a 4x4 image with known values
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        # 0  1  2  3
        # 4  5  6  7
        # 8  9  10 11
        # 12 13 14 15

        # Extract (0,0) sub-image: [0, 2], [8, 10]
        sub_00 = extract_bayer_subimage(image, 0, 0)
        assert sub_00.shape == (2, 2)
        np.testing.assert_array_equal(sub_00, [[0, 2], [8, 10]])

        # Extract (1,0) sub-image: [1, 3], [9, 11]
        sub_10 = extract_bayer_subimage(image, 1, 0)
        np.testing.assert_array_equal(sub_10, [[1, 3], [9, 11]])

        # Extract (0,1) sub-image: [4, 6], [12, 14]
        sub_01 = extract_bayer_subimage(image, 0, 1)
        np.testing.assert_array_equal(sub_01, [[4, 6], [12, 14]])

        # Extract (1,1) sub-image: [5, 7], [13, 15]
        sub_11 = extract_bayer_subimage(image, 1, 1)
        np.testing.assert_array_equal(sub_11, [[5, 7], [13, 15]])

    def test_insert_subimage(self):
        """Test inserting a sub-image back."""
        image = np.zeros((4, 4), dtype=np.uint8)
        subimage = np.array([[100, 101], [102, 103]], dtype=np.uint8)

        insert_bayer_subimage(image, subimage, 1, 0)

        # Should insert at (1,0), (3,0), (1,2), (3,2)
        assert image[0, 1] == 100
        assert image[0, 3] == 101
        assert image[2, 1] == 102
        assert image[2, 3] == 103

        # Other positions should be zero
        assert image[0, 0] == 0
        assert image[1, 1] == 0

    def test_extract_insert_roundtrip(self):
        """Extract then insert should preserve data."""
        original = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        reconstructed = np.zeros_like(original)

        for ox, oy in iter_bayer_offsets():
            sub = extract_bayer_subimage(original, ox, oy)
            insert_bayer_subimage(reconstructed, sub, ox, oy)

        np.testing.assert_array_equal(reconstructed, original)


class TestCreateBayerTestImage:
    """Test Bayer test image generation."""

    def test_basic_generation(self):
        """Test basic test image generation."""
        image = create_bayer_test_image(8, 8)
        assert image.shape == (8, 8)
        assert image.dtype == np.uint8

    def test_different_patterns(self):
        """Test that different patterns produce different images."""
        rggb = create_bayer_test_image(8, 8, BayerPattern.RGGB)
        bggr = create_bayer_test_image(8, 8, BayerPattern.BGGR)

        # Different patterns should produce different images
        assert not np.array_equal(rggb, bggr)

    def test_different_dtypes(self):
        """Test different data types."""
        img8 = create_bayer_test_image(8, 8, dtype=np.uint8)
        img16 = create_bayer_test_image(8, 8, dtype=np.uint16)

        assert img8.dtype == np.uint8
        assert img16.dtype == np.uint16

        # 16-bit should have larger values
        assert img16.max() > img8.max()


class TestCreateUniformBayerImage:
    """Test uniform Bayer image generation."""

    def test_rggb_uniform(self):
        """Test RGGB uniform image with known values."""
        image = create_uniform_bayer_image(
            4, 4, r_value=100, g_value=150, b_value=200, pattern=BayerPattern.RGGB
        )

        # RGGB pattern:
        # R G R G    (row 0)
        # G B G B    (row 1)
        # R G R G    (row 2)
        # G B G B    (row 3)

        # Check R positions (0,0), (2,0), (0,2), (2,2)
        assert image[0, 0] == 100
        assert image[0, 2] == 100
        assert image[2, 0] == 100
        assert image[2, 2] == 100

        # Check G positions
        assert image[0, 1] == 150  # G at (1,0) in RGGB
        assert image[1, 0] == 150  # G at (0,1) in RGGB

        # Check B positions (1,1), (3,1), (1,3), (3,3)
        assert image[1, 1] == 200
        assert image[1, 3] == 200
        assert image[3, 1] == 200
        assert image[3, 3] == 200

    def test_grbg_uniform(self):
        """Test GRBG uniform image."""
        image = create_uniform_bayer_image(
            4, 4, r_value=100, g_value=150, b_value=200, pattern=BayerPattern.GRBG
        )

        # GRBG pattern:
        # G R G R    (row 0)
        # B G B G    (row 1)
        # G R G R    (row 2)
        # B G B G    (row 3)

        assert image[0, 0] == 150  # G at (0,0)
        assert image[0, 1] == 100  # R at (1,0)
        assert image[1, 0] == 200  # B at (0,1)
        assert image[1, 1] == 150  # G at (1,1)


class TestValidateBayerDimensions:
    """Test dimension validation."""

    def test_valid_dimensions(self):
        """Valid dimensions should not raise."""
        validate_bayer_dimensions(8, 8)
        validate_bayer_dimensions(2, 2)
        validate_bayer_dimensions(100, 100)

    def test_too_small(self):
        """Dimensions less than 2x2 should raise."""
        with pytest.raises(ValueError, match="at least 2x2"):
            validate_bayer_dimensions(1, 8)
        with pytest.raises(ValueError, match="at least 2x2"):
            validate_bayer_dimensions(8, 1)
        with pytest.raises(ValueError, match="at least 2x2"):
            validate_bayer_dimensions(1, 1)


class TestBayerSubImageDataclass:
    """Test BayerSubImage dataclass."""

    def test_creation(self):
        """Test creating BayerSubImage."""
        sub = BayerSubImage(ox=0, oy=1, color="G", is_chroma=False)
        assert sub.ox == 0
        assert sub.oy == 1
        assert sub.color == "G"
        assert sub.is_chroma is False


class TestBayerBlockDecoder:
    """Integration tests for Bayer mode with block decoder."""

    def test_bayer_import_in_decoder(self):
        """Test that block_decoder imports Bayer utilities."""
        from pygfwx.core.block_decoder import decode_image  # noqa: F401

        # Import should not fail
        pass

    def test_decode_image_with_bayer_intent(self, sdk_wrapper):
        """Test that decode_image handles Bayer intent."""
        # Create a Bayer test image
        image = create_uniform_bayer_image(
            32, 32, r_value=100, g_value=150, b_value=200, pattern=BayerPattern.RGGB
        )

        # Encode with SDK using Bayer intent
        compressed = sdk_wrapper.encode(
            image,
            channels=1,
            quality=1024,
            intent=int(Intent.BAYER_RGGB),
        )

        # Verify header has Bayer intent
        header = sdk_wrapper.read_header(compressed)
        assert header.intent == Intent.BAYER_RGGB

        # Decode with PyGFWX
        from pygfwx.core.block_decoder import decode_image

        result = decode_image(compressed)

        # Should match original
        assert result.image.shape == image.shape
        # For lossless, should be exact match
        np.testing.assert_array_equal(result.image, image)

    def test_decode_bayer_lossy(self, sdk_wrapper):
        """Test lossy Bayer decoding."""
        image = create_uniform_bayer_image(
            32, 32, r_value=100, g_value=150, b_value=200, pattern=BayerPattern.RGGB
        )

        # Encode with lossy quality
        compressed = sdk_wrapper.encode(
            image,
            channels=1,
            quality=512,  # Lossy
            intent=int(Intent.BAYER_RGGB),
        )

        # Decode with PyGFWX
        from pygfwx.core.block_decoder import decode_image

        result = decode_image(compressed)

        # Should be close to original
        assert result.image.shape == image.shape
        # Check approximate values
        r_positions = result.image[0::2, 0::2]  # R at (0,0) pattern
        g1_positions = result.image[0::2, 1::2]  # G at (1,0)
        b_positions = result.image[1::2, 1::2]  # B at (1,1)

        assert np.abs(r_positions.mean() - 100) < 20
        assert np.abs(g1_positions.mean() - 150) < 20
        assert np.abs(b_positions.mean() - 200) < 20


class TestBayerPatternEnum:
    """Test BayerPattern enum values."""

    def test_pattern_values_match_intent(self):
        """Pattern values should match Intent values."""
        assert BayerPattern.RGGB == Intent.BAYER_RGGB
        assert BayerPattern.BGGR == Intent.BAYER_BGGR
        assert BayerPattern.GRBG == Intent.BAYER_GRBG
        assert BayerPattern.GBRG == Intent.BAYER_GBRG
        assert BayerPattern.GENERIC == Intent.BAYER_GENERIC
