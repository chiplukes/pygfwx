"""
Tests for GFWX header parsing.
"""

import sys
from pathlib import Path

import pytest

from pygfwx.core.header import (
    GFWX_MAGIC,
    QUALITY_MAX,
    Encoder,
    Filter,
    GFWXHeader,
    HeaderParseError,
    Intent,
    create_default_header,
    parse_header,
    write_header,
)

# Add project root for cross_codec import
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_codec.gfwx_sdk import GFWXWrapper, is_sdk_available

from pygfwx.utils.reference_images import create_reference_image


class TestHeaderConstants:
    """Test header constants."""

    def test_magic_number(self):
        """Test magic number value."""
        # 'GFWX' - SDK writes as: 'G' | ('F' << 8) | ('W' << 16) | ('X' << 24)
        expected = ord("G") | (ord("F") << 8) | (ord("W") << 16) | (ord("X") << 24)
        assert expected == GFWX_MAGIC
        assert GFWX_MAGIC == 0x58574647

    def test_quality_max(self):
        """Test maximum quality value."""
        assert QUALITY_MAX == 1024


class TestHeaderEnums:
    """Test header enum values."""

    def test_filter_values(self):
        """Test filter enum values match SDK."""
        assert Filter.LINEAR == 0
        assert Filter.CUBIC == 1

    def test_encoder_values(self):
        """Test encoder enum values match SDK."""
        assert Encoder.TURBO == 0
        assert Encoder.FAST == 1
        assert Encoder.CONTEXTUAL == 2
        assert Encoder.HIGH_BITRATE == 3

    def test_intent_values(self):
        """Test intent enum values match SDK ordering."""
        # SDK: IntentGeneric = 0, IntentMono = 1, IntentBayerRGGB = 2, ...
        assert Intent.GENERIC == 0
        assert Intent.MONO == 1
        assert Intent.BAYER_RGGB == 2
        assert Intent.BAYER_BGGR == 3
        assert Intent.BAYER_GRBG == 4
        assert Intent.BAYER_GBRG == 5
        assert Intent.BAYER_GENERIC == 6
        assert Intent.RGB == 7
        assert Intent.RGBA == 8
        assert Intent.BGR == 9
        assert Intent.BGRA == 10


class TestHeaderParsing:
    """Test header parsing."""

    def test_invalid_magic(self):
        """Test that invalid magic number raises error."""
        # Create data with wrong magic
        data = b"XXXX" + b"\x00" * 32

        with pytest.raises(HeaderParseError, match="Invalid magic"):
            parse_header(data)

    def test_data_too_short(self):
        """Test that short data raises error."""
        data = b"GFWX"  # Only magic, no other fields

        with pytest.raises(HeaderParseError, match="too short"):
            parse_header(data)


@pytest.mark.skipif(not is_sdk_available(), reason="SDK not available")
class TestHeaderParsingWithSDK:
    """Test header parsing against SDK-compressed data."""

    @pytest.fixture
    def sdk(self):
        """Create SDK wrapper."""
        return GFWXWrapper()

    def test_parse_mono_lossless(self, sdk):
        """Test parsing header from lossless mono image."""
        image = create_reference_image(size=64, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024, filter=Filter.LINEAR)

        header, header_size = parse_header(compressed)

        assert header.version == 1
        assert header.sizex == 64
        assert header.sizey == 64
        assert header.channels == 1
        assert header.bit_depth == 8
        assert not header.is_signed
        assert header.quality == 1024
        assert header.filter == Filter.LINEAR
        assert header.is_lossless

    def test_parse_rgb_lossy(self, sdk):
        """Test parsing header from lossy RGB image."""
        image = create_reference_image(size=32, channels=3, bit_depth=8)
        compressed = sdk.encode(image, quality=512, filter=Filter.CUBIC, use_transform=False)

        header, header_size = parse_header(compressed)

        assert header.sizex == 32
        assert header.sizey == 32
        assert header.channels == 3
        assert header.quality == 512
        assert header.filter == Filter.CUBIC
        assert not header.is_lossless

    def test_parse_16bit(self, sdk):
        """Test parsing header from 16-bit image."""
        image = create_reference_image(size=32, channels=1, bit_depth=16)
        compressed = sdk.encode(image, quality=1024)

        header, header_size = parse_header(compressed)

        assert header.bit_depth == 16
        assert not header.is_signed  # uint16

    def test_parse_encoder_type(self, sdk):
        """Test parsing different encoder types."""
        image = create_reference_image(size=32, channels=1, bit_depth=8)

        # Test FAST encoder
        compressed_fast = sdk.encode(
            image,
            quality=1024,
            encoder=Encoder.FAST,
        )
        header_fast, _ = parse_header(compressed_fast)
        assert header_fast.encoder == Encoder.FAST

        # Test CONTEXTUAL encoder
        compressed_ctx = sdk.encode(
            image,
            quality=1024,
            encoder=Encoder.CONTEXTUAL,
        )
        header_ctx, _ = parse_header(compressed_ctx)
        assert header_ctx.encoder == Encoder.CONTEXTUAL

    def test_header_matches_sdk(self, sdk):
        """Test that parsed header matches SDK's read_header."""
        image = create_reference_image(size=48, channels=3, bit_depth=8)
        compressed = sdk.encode(
            image,
            quality=768,
            filter=Filter.CUBIC,
            encoder=Encoder.CONTEXTUAL,
            use_transform=False,
        )

        # Parse with our code
        our_header, _ = parse_header(compressed)

        # Parse with SDK
        sdk_header = sdk.read_header(compressed)

        # Compare fields
        assert our_header.sizex == sdk_header.sizex
        assert our_header.sizey == sdk_header.sizey
        assert our_header.channels == sdk_header.channels
        assert our_header.bit_depth == sdk_header.bit_depth
        assert our_header.is_signed == sdk_header.is_signed
        assert our_header.quality == sdk_header.quality
        assert our_header.filter.value == sdk_header.filter.value
        assert our_header.encoder.value == sdk_header.encoder.value

    def test_header_size_calculation(self, sdk):
        """Test that header size is calculated correctly."""
        image = create_reference_image(size=32, channels=1, bit_depth=8)
        compressed = sdk.encode(image, quality=1024)

        header, header_size = parse_header(compressed)

        # Header should be 32 bytes (fixed) + metadata
        expected_size = 32 + header.metadata_size * 4
        assert header_size == expected_size


class TestWriteHeader:
    """Tests for header writing."""

    def test_write_minimal_header(self):
        """Test writing a minimal header."""
        header = create_default_header(width=64, height=48)
        data = write_header(header)

        # Should be exactly 32 bytes (no metadata)
        assert len(data) == 32

        # Should start with magic number
        assert data[:4] == b"GFWX"

    def test_roundtrip_header(self):
        """Test that written header can be parsed back."""
        original = create_default_header(
            width=128,
            height=96,
            channels=3,
            quality=768,
            bit_depth=16,
            filter_type=Filter.CUBIC,
            encoder=Encoder.CONTEXTUAL,
            intent=Intent.RGB,
            chroma_scale=2,
            block_size=8,
        )
        data = write_header(original)
        parsed, size = parse_header(data)

        assert parsed.sizex == original.sizex
        assert parsed.sizey == original.sizey
        assert parsed.channels == original.channels
        assert parsed.layers == original.layers
        assert parsed.quality == original.quality
        assert parsed.bit_depth == original.bit_depth
        assert parsed.is_signed == original.is_signed
        assert parsed.filter == original.filter
        assert parsed.encoder == original.encoder
        assert parsed.intent == original.intent
        assert parsed.chroma_scale == original.chroma_scale
        assert parsed.block_size == original.block_size
        assert size == 32

    def test_roundtrip_all_quality_values(self):
        """Test roundtrip for various quality values."""
        for quality in [1, 100, 512, 768, 1024]:
            header = create_default_header(width=32, height=32, quality=quality)
            data = write_header(header)
            parsed, _ = parse_header(data)
            assert parsed.quality == quality

    def test_roundtrip_signed_unsigned(self):
        """Test roundtrip for signed/unsigned."""
        for is_signed in [False, True]:
            header = create_default_header(width=32, height=32, is_signed=is_signed)
            data = write_header(header)
            parsed, _ = parse_header(data)
            assert parsed.is_signed == is_signed

    def test_write_with_metadata(self):
        """Test writing header with metadata."""
        header = create_default_header(width=32, height=32)
        metadata = b"TEST" * 4  # 16 bytes = 4 words
        data = write_header(header, metadata=metadata)

        # Should be 32 + 16 = 48 bytes
        assert len(data) == 48

        # Parse and verify metadata size
        parsed, size = parse_header(data)
        assert parsed.metadata_size == 4  # 4 words
        assert size == 48

    def test_write_metadata_must_be_multiple_of_4(self):
        """Test that metadata must be multiple of 4 bytes."""
        header = create_default_header(width=32, height=32)

        with pytest.raises(ValueError):
            write_header(header, metadata=b"ABC")  # 3 bytes, not multiple of 4

    def test_roundtrip_all_filters(self):
        """Test roundtrip for all filter types."""
        for filter_type in Filter:
            header = create_default_header(width=32, height=32, filter_type=filter_type)
            data = write_header(header)
            parsed, _ = parse_header(data)
            assert parsed.filter == filter_type

    def test_roundtrip_all_encoders(self):
        """Test roundtrip for all encoder modes."""
        for encoder in Encoder:
            header = create_default_header(width=32, height=32, encoder=encoder)
            data = write_header(header)
            parsed, _ = parse_header(data)
            assert parsed.encoder == encoder


class TestCreateDefaultHeader:
    """Tests for create_default_header helper."""

    def test_minimal_defaults(self):
        """Test minimal header creation."""
        header = create_default_header(width=100, height=50)
        assert header.sizex == 100
        assert header.sizey == 50
        assert header.channels == 1
        assert header.layers == 1
        assert header.quality == QUALITY_MAX
        assert header.bit_depth == 8
        assert header.is_signed is False
        assert header.filter == Filter.LINEAR
        assert header.encoder == Encoder.CONTEXTUAL  # CONTEXTUAL is the default for best compression

    def test_custom_values(self):
        """Test header with custom values."""
        header = create_default_header(
            width=256,
            height=128,
            channels=4,
            quality=512,
            bit_depth=16,
            is_signed=True,
            filter_type=Filter.CUBIC,
        )
        assert header.sizex == 256
        assert header.sizey == 128
        assert header.channels == 4
        assert header.quality == 512
        assert header.bit_depth == 16
        assert header.is_signed is True
        assert header.filter == Filter.CUBIC


@pytest.mark.skipif(not is_sdk_available(), reason="SDK not available")
class TestWriteHeaderSDKComparison:
    """Tests comparing written headers with SDK."""

    @pytest.fixture
    def sdk(self):
        """Create SDK wrapper."""
        return GFWXWrapper()

    def test_header_bytes_match_sdk(self, sdk):
        """Test that our header bytes match SDK header exactly."""
        from pygfwx.utils.reference_images import create_reference_image

        # Create reference image and encode with SDK
        image = create_reference_image(size=32, channels=1, bit_depth=8)
        sdk_compressed = sdk.encode(image, quality=512)

        # Parse SDK header to get exact parameters
        sdk_header = sdk.read_header(sdk_compressed)

        # Create our header with same parameters
        our_header = GFWXHeader(
            version=1,
            sizex=sdk_header.sizex,
            sizey=sdk_header.sizey,
            layers=sdk_header.layers,
            channels=sdk_header.channels,
            bit_depth=sdk_header.bit_depth,
            is_signed=sdk_header.is_signed,
            quality=sdk_header.quality,
            chroma_scale=sdk_header.chroma_scale,
            block_size=sdk_header.block_size,
            filter=Filter(sdk_header.filter.value),
            quantization=sdk_header.quantization,
            encoder=Encoder(sdk_header.encoder.value),
            intent=Intent(sdk_header.intent.value),
            metadata_size=0,  # SDK has metadata, but we'll compare fixed part only
        )

        # Write our header
        our_bytes = write_header(our_header)

        # Compare fixed header portion (first 32 bytes)
        # Note: SDK may have metadata, so just compare fixed part
        assert our_bytes[:32] == sdk_compressed[:32]

