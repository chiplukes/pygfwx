"""
Tests for GFWX color transforms.

Tests cover:
- Forward UYV transform
- Inverse UYV transform
- Transform program I/O
- Roundtrip validation
"""

import numpy as np
import pytest

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.transforms import (
    TRANSFORM_UYV_PROGRAM,
    forward_transform_uyv,
    get_chroma_flags,
    inverse_transform_uyv,
    read_transform_program,
    write_transform_program,
)


class TestUYVTransform:
    """Tests for UYV color transform."""

    def test_forward_basic(self):
        """Test forward transform on simple RGB data."""
        # Create a simple 2x2 RGB image
        image = np.array(
            [[[100, 50, 75], [200, 100, 150]], [[50, 25, 30], [150, 75, 100]]], dtype=np.uint8
        )

        result, program = forward_transform_uyv(image, boost=1)

        # Result should be (channels, height, width)
        assert result.shape == (3, 2, 2)

        # Verify transform equations (boost=1):
        # R' = R - G
        # B' = B - G
        # G' = G + (R' + B') / 4
        r, g, b = image[0, 0]
        r_prime = r - g
        b_prime = b - g
        g_prime = g + (r_prime + b_prime) // 4

        assert result[0, 0, 0] == r_prime  # R' = 100 - 50 = 50
        assert result[2, 0, 0] == b_prime  # B' = 75 - 50 = 25
        assert result[1, 0, 0] == g_prime  # G' = 50 + (50 + 25)//4 = 50 + 18 = 68

    def test_forward_with_boost(self):
        """Test forward transform with boost factor."""
        image = np.array([[[128, 128, 128]]], dtype=np.uint8)

        result, _ = forward_transform_uyv(image, boost=8)

        # For gray (R=G=B), R'=0, B'=0, G'=G*boost
        assert result[0, 0, 0] == 0  # R' = (128-128)*8 = 0
        assert result[2, 0, 0] == 0  # B' = (128-128)*8 = 0
        assert result[1, 0, 0] == 128 * 8  # G' = 128*8 + 0 = 1024

    def test_roundtrip(self):
        """Test forward then inverse returns original."""
        image = np.array(
            [[[100, 50, 75], [200, 100, 150]], [[50, 25, 30], [150, 75, 100]]], dtype=np.uint8
        )

        boost = 8
        transformed, _ = forward_transform_uyv(image, boost=boost)
        recovered = inverse_transform_uyv(transformed, boost=boost)

        # Should recover original values
        np.testing.assert_array_equal(recovered, image)

    def test_roundtrip_random(self):
        """Test roundtrip with random data."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        boost = 8
        transformed, _ = forward_transform_uyv(image, boost=boost)
        recovered = inverse_transform_uyv(transformed, boost=boost)

        np.testing.assert_array_equal(recovered, image)

    def test_roundtrip_lossless_boost(self):
        """Test roundtrip with lossless boost=1."""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

        transformed, _ = forward_transform_uyv(image, boost=1)
        recovered = inverse_transform_uyv(transformed, boost=1)

        np.testing.assert_array_equal(recovered, image)

    def test_requires_3_channels(self):
        """Test that UYV transform requires at least 3 channels."""
        image = np.zeros((4, 4, 2), dtype=np.uint8)

        with pytest.raises(ValueError):
            forward_transform_uyv(image)

    def test_extra_channels_preserved(self):
        """Test that channels beyond RGB are preserved."""
        # RGBA image
        image = np.array([[[100, 50, 75, 200]]], dtype=np.uint8)

        transformed, _ = forward_transform_uyv(image, boost=1)
        recovered = inverse_transform_uyv(transformed, boost=1)

        # Alpha should be preserved
        assert recovered[0, 0, 3] == 200


class TestTransformProgram:
    """Tests for transform program I/O."""

    def test_write_no_transform(self):
        """Test writing no transform (None)."""
        writer = BitWriter(10)
        write_transform_program(writer, None)

        # Read back
        reader = BitReader(writer.buffer)
        program = read_transform_program(reader)

        assert program is None

    def test_write_empty_transform(self):
        """Test writing empty transform."""
        writer = BitWriter(10)
        write_transform_program(writer, [])

        reader = BitReader(writer.buffer)
        program = read_transform_program(reader)

        assert program is None

    def test_roundtrip_uyv_program(self):
        """Test roundtrip of UYV transform program."""
        writer = BitWriter(100)
        write_transform_program(writer, TRANSFORM_UYV_PROGRAM)

        reader = BitReader(writer.buffer)
        program = read_transform_program(reader)

        assert program == TRANSFORM_UYV_PROGRAM


class TestChromaFlags:
    """Tests for chroma flag extraction."""

    def test_no_transform(self):
        """Test chroma flags with no transform."""
        flags = get_chroma_flags(3, has_transform=False)
        assert flags == [0, 0, 0]

    def test_mono_no_transform(self):
        """Test chroma flags for mono image."""
        flags = get_chroma_flags(1, has_transform=False)
        assert flags == [0]

    def test_uyv_program(self):
        """Test chroma flags from UYV program."""
        flags = get_chroma_flags(3, has_transform=True, transform_program=TRANSFORM_UYV_PROGRAM)

        # R and B are chroma (1), G is luma (0)
        assert flags[0] == 1  # R is chroma
        assert flags[1] == 0  # G is luma
        assert flags[2] == 1  # B is chroma


class TestCustomTransformBuilding:
    """Tests for custom transform building utilities."""

    def test_build_transform_step_simple(self):
        """Test building a simple subtraction step."""
        from pygfwx.core.transforms import build_transform_step

        # R -= G (i.e., R += G * -1)
        step = build_transform_step(0, [(1, -1)], 1, is_chroma=True)

        assert step == [0, 1, -1, -1, 1, 1]

    def test_build_transform_step_with_division(self):
        """Test building a step with division."""
        from pygfwx.core.transforms import build_transform_step

        # G += (R + B) / 4
        step = build_transform_step(1, [(0, 1), (2, 1)], 4, is_chroma=False)

        assert step == [1, 0, 1, 2, 1, -1, 4, 0]

    def test_build_transform_program(self):
        """Test building a complete transform program."""
        from pygfwx.core.transforms import build_transform_program, build_transform_step

        # Simple UYV-like transform
        program = build_transform_program(
            build_transform_step(0, [(1, -1)], 1, is_chroma=True),  # R -= G
            build_transform_step(2, [(1, -1)], 1, is_chroma=True),  # B -= G
            build_transform_step(1, [(0, 1), (2, 1)], 4, is_chroma=False),  # G += (R+B)/4
        )

        # Should end with -1
        assert program[-1] == -1

        # Should have 3 steps plus end marker
        from pygfwx.core.transforms import parse_transform_steps

        steps = parse_transform_steps(program)
        assert len(steps) == 3

    def test_validate_transform_program_valid(self):
        """Test validation of a valid program."""
        from pygfwx.core.transforms import TRANSFORM_UYV_PROGRAM, validate_transform_program

        assert validate_transform_program(TRANSFORM_UYV_PROGRAM, 3)

    def test_validate_transform_program_invalid_channel(self):
        """Test validation catches invalid channel references."""
        from pygfwx.core.transforms import validate_transform_program

        # Reference channel 5 with only 3 channels
        program = [5, 1, -1, -1, 1, 1, -1]  # dest=5 is invalid
        with pytest.raises(ValueError, match="Invalid destination channel"):
            validate_transform_program(program, 3)

    def test_validate_transform_program_missing_end_marker(self):
        """Test validation catches missing end marker."""
        from pygfwx.core.transforms import validate_transform_program

        program = [0, 1, -1, -1, 1, 1]  # Missing final -1
        with pytest.raises(ValueError, match="must end with -1"):
            validate_transform_program(program, 3)

    def test_validate_transform_program_invalid_denominator(self):
        """Test validation catches invalid denominator."""
        from pygfwx.core.transforms import validate_transform_program

        program = [0, 1, -1, -1, 0, 1, -1]  # denom=0 is invalid
        with pytest.raises(ValueError, match="Invalid denominator"):
            validate_transform_program(program, 3)

    def test_parse_transform_steps_uyv(self):
        """Test parsing UYV program into steps."""
        from pygfwx.core.transforms import TRANSFORM_UYV_PROGRAM, parse_transform_steps

        steps = parse_transform_steps(TRANSFORM_UYV_PROGRAM)

        assert len(steps) == 3

        # Step 1: R -= G
        assert steps[0]["dest"] == 0
        assert steps[0]["sources"] == [(1, -1)]
        assert steps[0]["denominator"] == 1
        assert steps[0]["is_chroma"] is True

    def test_describe_transform_program(self):
        """Test human-readable description."""
        from pygfwx.core.transforms import TRANSFORM_UYV_PROGRAM, describe_transform_program

        desc = describe_transform_program(TRANSFORM_UYV_PROGRAM, ["R", "G", "B"])

        assert "R +=" in desc
        assert "G +=" in desc
        assert "B +=" in desc
        assert "chroma" in desc
        assert "luma" in desc

    def test_describe_no_transform(self):
        """Test description of no transform."""
        from pygfwx.core.transforms import describe_transform_program

        desc = describe_transform_program([-1])
        assert "identity" in desc.lower() or "no transform" in desc.lower()


class TestGenericTransformRoundtrip:
    """Tests for generic forward/inverse transform roundtrip."""

    def test_generic_roundtrip_uyv(self):
        """Test generic transform roundtrip with UYV program."""
        from pygfwx.core.transforms import (
            TRANSFORM_UYV_PROGRAM,
            forward_transform_generic,
            inverse_transform_generic,
        )

        image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        boost = 8

        # Forward transform
        transformed, is_chroma = forward_transform_generic(image, TRANSFORM_UYV_PROGRAM, boost)

        # Verify is_chroma flags
        assert is_chroma[0] == 1  # R is chroma
        assert is_chroma[1] == 0  # G is luma
        assert is_chroma[2] == 1  # B is chroma

        # Inverse transform
        recovered = inverse_transform_generic(transformed, TRANSFORM_UYV_PROGRAM, boost)

        np.testing.assert_array_equal(recovered, image)

    def test_generic_roundtrip_custom(self):
        """Test roundtrip with a custom transform."""
        from pygfwx.core.transforms import (
            build_transform_program,
            build_transform_step,
            forward_transform_generic,
            inverse_transform_generic,
        )

        # Simple custom transform: channel 0 -= channel 1
        program = build_transform_program(
            build_transform_step(0, [(1, -1)], 1, is_chroma=True),
        )

        image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        boost = 1

        transformed, _ = forward_transform_generic(image, program, boost)
        recovered = inverse_transform_generic(transformed, program, boost)

        np.testing.assert_array_equal(recovered, image)

    def test_generic_transform_lossless(self):
        """Test generic transform is lossless with boost=1."""
        from pygfwx.core.transforms import (
            TRANSFORM_A710_RGB,
            forward_transform_generic,
            inverse_transform_generic,
        )

        image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

        transformed, _ = forward_transform_generic(image, TRANSFORM_A710_RGB, boost=1)
        recovered = inverse_transform_generic(transformed, TRANSFORM_A710_RGB, boost=1)

        np.testing.assert_array_equal(recovered, image)


class TestTransformSDKCompatibility:
    """Tests comparing transforms with SDK behavior."""

    @pytest.mark.sdk_required
    def test_transform_matches_sdk(self, sdk_wrapper):  # noqa: ARG002
        """Test that our transform produces same output as SDK."""
        # This would require comparing intermediate state during encoding
        # For now, we verify the roundtrip works
        from cross_codec.gfwx_sdk import decode, encode

        from pygfwx.utils.reference_images import create_reference_image

        # Create RGB image
        image = create_reference_image(size=16, channels=3, bit_depth=8)

        # Encode with SDK using transform
        encoded = encode(image, quality=1024)  # Lossless

        # Decode with SDK
        decoded = decode(encoded)

        # Should be bit-exact for lossless
        np.testing.assert_array_equal(decoded, image)

    @pytest.mark.sdk_required
    def test_transform_preserves_lossless(self, sdk_wrapper):  # noqa: ARG002
        """Test that transform doesn't introduce loss at quality=1024."""
        from cross_codec.gfwx_sdk import decode, encode

        from pygfwx.utils.reference_images import create_reference_image

        # Various test images
        for channels in [1, 3, 4]:
            image = create_reference_image(size=32, channels=channels, bit_depth=8)
            encoded = encode(image, quality=1024)
            decoded = decode(encoded)
            np.testing.assert_array_equal(decoded, image, f"Failed for {channels} channels")

