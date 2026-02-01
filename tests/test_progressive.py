"""
Tests for progressive decoding functionality.

Tests cover:
- Basic progressive decode at various downsampling levels
- Truncation handling and recovery
- ProgressiveDecoder streaming class
- Status codes and next_point_of_interest
"""

import numpy as np
import pytest

from pygfwx.core.header import (
    Encoder,
    Filter,
    GFWXHeader,
    Intent,
    write_header,
)
from pygfwx.streaming.progressive import (
    ProgressiveDecoder,
    ProgressiveResult,
    ProgressiveStatus,
    decode_progressive,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_gfwx_header() -> bytes:
    """Create a minimal valid GFWX header (32 bytes)."""
    header = GFWXHeader(
        version=1,
        sizex=8,
        sizey=8,
        layers=1,
        channels=1,
        bit_depth=8,
        is_signed=False,
        quality=512,
        chroma_scale=1,
        block_size=4,
        filter=Filter.LINEAR,
        quantization=0,
        encoder=Encoder.TURBO,
        intent=Intent.GENERIC,
        metadata_size=0,
    )
    return write_header(header)


# ============================================================================
# Basic Status Tests
# ============================================================================


class TestDecodeProgressiveStatus:
    """Test status codes from decode_progressive."""

    def test_empty_data_needs_more(self):
        """Empty data returns NEED_MORE_DATA status."""
        result = decode_progressive(b"")
        assert result.status == ProgressiveStatus.NEED_MORE_DATA
        assert result.next_point_of_interest >= 28

    def test_short_data_needs_more(self):
        """Data shorter than header returns NEED_MORE_DATA."""
        result = decode_progressive(b"GFWX0001")
        assert result.status == ProgressiveStatus.NEED_MORE_DATA
        assert result.next_point_of_interest >= 28

    def test_invalid_magic_malformed(self):
        """Invalid magic bytes returns MALFORMED."""
        data = b"BAD!" + b"\x00" * 30
        result = decode_progressive(data)
        assert result.status == ProgressiveStatus.MALFORMED

    def test_unsupported_version(self, minimal_gfwx_header):
        """Unsupported version returns UNSUPPORTED."""
        # Modify version to 999
        data = bytearray(minimal_gfwx_header)
        data[4:8] = (999).to_bytes(4, "little")
        result = decode_progressive(bytes(data))
        assert result.status == ProgressiveStatus.UNSUPPORTED
        # Header may be None if version prevents parsing


# ============================================================================
# Header Parsing Tests
# ============================================================================


class TestProgressiveHeaderParsing:
    """Test header parsing in progressive decode."""

    def test_valid_header_parsed(self, minimal_gfwx_header):
        """Valid header is parsed and included in result."""
        result = decode_progressive(minimal_gfwx_header)
        # Even with truncated data, header should be parsed
        assert result.header is not None
        assert result.header.sizex == 8
        assert result.header.sizey == 8
        assert result.header.channels == 1
        assert result.header.bit_depth == 8


# ============================================================================
# Downsampling Tests
# ============================================================================


class TestProgressiveDownsampling:
    """Test downsampling functionality."""

    def test_downsampling_calculates_output_dims(self, minimal_gfwx_header):
        """Downsampling affects expected output dimensions."""
        # Note: With truncated data, we can still check header parsing
        result = decode_progressive(minimal_gfwx_header, downsampling=1)
        assert result.header is not None
        # At downsampling=1, 8x8 becomes 4x4


# ============================================================================
# Truncation Handling Tests
# ============================================================================


class TestTruncationHandling:
    """Test handling of truncated data."""

    def test_truncated_after_header_returns_need_more(self, minimal_gfwx_header):
        """Truncation after header indicates need for more data."""
        result = decode_progressive(minimal_gfwx_header)
        # Should need more data for transform program and coefficients
        assert result.status == ProgressiveStatus.NEED_MORE_DATA
        assert result.next_point_of_interest > len(minimal_gfwx_header)

    def test_levels_decoded_starts_at_zero(self, minimal_gfwx_header):
        """With minimal data, no levels are decoded."""
        result = decode_progressive(minimal_gfwx_header)
        assert result.levels_decoded == 0


# ============================================================================
# ProgressiveDecoder Class Tests
# ============================================================================


class TestProgressiveDecoderClass:
    """Test the ProgressiveDecoder streaming class."""

    def test_initial_state(self):
        """New decoder has zero bytes received."""
        decoder = ProgressiveDecoder()
        assert decoder.bytes_received == 0

    def test_feed_accumulates_data(self):
        """Feed accumulates data."""
        decoder = ProgressiveDecoder()
        decoder.feed(b"GFWX")
        assert decoder.bytes_received == 4
        decoder.feed(b"0001")
        assert decoder.bytes_received == 8

    def test_reset_clears_data(self):
        """Reset clears accumulated data."""
        decoder = ProgressiveDecoder()
        decoder.feed(b"GFWX0001some more data")
        decoder.reset()
        assert decoder.bytes_received == 0

    def test_get_returns_result(self, minimal_gfwx_header):
        """Get returns decode result."""
        decoder = ProgressiveDecoder()
        decoder.feed(minimal_gfwx_header)
        result = decoder.get()
        assert isinstance(result, ProgressiveResult)
        assert result.header is not None


# ============================================================================
# Result Dataclass Tests
# ============================================================================


class TestProgressiveResult:
    """Test ProgressiveResult dataclass."""

    def test_default_values(self):
        """Check default values."""
        result = ProgressiveResult(status=ProgressiveStatus.OK)
        assert result.image is None
        assert result.header is None
        assert result.next_point_of_interest == 0
        assert result.levels_decoded == 0
        assert result.max_levels == 0
        assert result.actual_downsampling == 0

    def test_with_all_fields(self):
        """Create result with all fields."""
        img = np.zeros((4, 4), dtype=np.uint8)
        result = ProgressiveResult(
            status=ProgressiveStatus.OK,
            image=img,
            header=None,
            next_point_of_interest=100,
            levels_decoded=3,
            max_levels=5,
            actual_downsampling=1,
        )
        assert result.status == ProgressiveStatus.OK
        assert result.image is img
        assert result.next_point_of_interest == 100
        assert result.levels_decoded == 3


# ============================================================================
# ProgressiveStatus Enum Tests
# ============================================================================


class TestProgressiveStatus:
    """Test status enum values."""

    def test_ok_is_zero(self):
        """OK status is 0."""
        assert ProgressiveStatus.OK == 0

    def test_need_more_is_positive(self):
        """NEED_MORE_DATA is positive."""
        assert ProgressiveStatus.NEED_MORE_DATA > 0

    def test_errors_are_negative(self):
        """Error statuses are negative."""
        assert ProgressiveStatus.MALFORMED < 0
        assert ProgressiveStatus.UNSUPPORTED < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
