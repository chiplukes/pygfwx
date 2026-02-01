"""
Tests for GFWX metadata support.

Tests the metadata module's utilities for reading, writing, and manipulating
optional metadata blocks in GFWX files.
"""

import struct

import pytest

from pygfwx.core.header import (
    create_default_header,
    write_header,
)
from pygfwx.core.metadata import (
    FIXED_HEADER_SIZE,
    MetadataChunk,
    create_binary_metadata,
    create_chunked_metadata,
    create_json_metadata,
    create_key_value_metadata,
    create_text_metadata,
    describe_metadata,
    find_chunk,
    get_data_start_offset,
    get_metadata_size,
    get_metadata_word_count,
    pad_to_word_boundary,
    read_chunked_metadata,
    read_json_metadata,
    read_key_value_metadata,
    read_metadata_raw,
    read_text_metadata,
    validate_metadata,
)


class TestPadding:
    """Test padding utilities."""

    def test_pad_empty(self):
        """Padding empty bytes returns empty."""
        assert pad_to_word_boundary(b"") == b""

    def test_pad_already_aligned(self):
        """Data already on 4-byte boundary unchanged."""
        assert pad_to_word_boundary(b"TEST") == b"TEST"
        assert pad_to_word_boundary(b"ABCDEFGH") == b"ABCDEFGH"

    def test_pad_1_byte(self):
        """1 byte padded to 4."""
        assert pad_to_word_boundary(b"X") == b"X\x00\x00\x00"

    def test_pad_2_bytes(self):
        """2 bytes padded to 4."""
        assert pad_to_word_boundary(b"XY") == b"XY\x00\x00"

    def test_pad_3_bytes(self):
        """3 bytes padded to 4."""
        assert pad_to_word_boundary(b"XYZ") == b"XYZ\x00"

    def test_pad_5_bytes(self):
        """5 bytes padded to 8."""
        assert pad_to_word_boundary(b"ABCDE") == b"ABCDE\x00\x00\x00"


class TestValidation:
    """Test metadata validation."""

    def test_validate_valid(self):
        """Valid metadata passes."""
        assert validate_metadata(b"") is True
        assert validate_metadata(b"TEST") is True
        assert validate_metadata(b"ABCDEFGH") is True

    def test_validate_invalid(self):
        """Invalid metadata fails."""
        assert validate_metadata(b"X") is False
        assert validate_metadata(b"XY") is False
        assert validate_metadata(b"XYZ") is False

    def test_word_count(self):
        """Word count calculation."""
        assert get_metadata_word_count(b"") == 0
        assert get_metadata_word_count(b"TEST") == 1
        assert get_metadata_word_count(b"ABCDEFGH") == 2

    def test_word_count_invalid(self):
        """Word count raises for invalid metadata."""
        with pytest.raises(ValueError):
            get_metadata_word_count(b"XYZ")


class TestTextMetadata:
    """Test text metadata utilities."""

    def test_create_text_simple(self):
        """Create text metadata."""
        result = create_text_metadata("test")
        assert result == b"test"  # Already 4 bytes

    def test_create_text_with_padding(self):
        """Create text metadata with padding."""
        result = create_text_metadata("hi")
        assert result == b"hi\x00\x00"

    def test_roundtrip_text(self):
        """Text roundtrip preserves content."""
        original = "Hello, GFWX!"
        metadata = create_text_metadata(original)
        recovered = read_text_metadata(metadata)
        assert recovered == original

    def test_roundtrip_unicode(self):
        """Unicode text roundtrip."""
        original = "こんにちは 🎉"
        metadata = create_text_metadata(original)
        recovered = read_text_metadata(metadata)
        assert recovered == original

    def test_read_strips_padding(self):
        """Reading strips null padding."""
        metadata = b"test\x00\x00\x00\x00"
        assert read_text_metadata(metadata) == "test"


class TestKeyValueMetadata:
    """Test key-value metadata utilities."""

    def test_create_key_value(self):
        """Create key-value metadata."""
        pairs = {"author": "test", "version": "1.0"}
        metadata = create_key_value_metadata(pairs)
        assert validate_metadata(metadata)

    def test_roundtrip_key_value(self):
        """Key-value roundtrip preserves content."""
        original = {"author": "John Doe", "software": "PyGFWX", "version": "1.0"}
        metadata = create_key_value_metadata(original)
        recovered = read_key_value_metadata(metadata)
        assert recovered == original

    def test_key_value_with_special_chars(self):
        """Key-value with special characters in values."""
        original = {"path": "C:\\test\\file.txt", "query": "a=1&b=2"}
        metadata = create_key_value_metadata(original)
        recovered = read_key_value_metadata(metadata)
        assert recovered == original


class TestJsonMetadata:
    """Test JSON metadata utilities."""

    def test_create_json_dict(self):
        """Create JSON metadata from dict."""
        obj = {"width": 100, "height": 200}
        metadata = create_json_metadata(obj)
        assert validate_metadata(metadata)

    def test_roundtrip_json_dict(self):
        """JSON dict roundtrip."""
        original = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
        metadata = create_json_metadata(original)
        recovered = read_json_metadata(metadata)
        assert recovered == original

    def test_roundtrip_json_list(self):
        """JSON list roundtrip."""
        original = [1, "two", 3.0, None, True]
        metadata = create_json_metadata(original)
        recovered = read_json_metadata(metadata)
        assert recovered == original

    def test_json_complex(self):
        """Complex JSON structure."""
        original = {
            "camera": {"make": "Canon", "model": "EOS R5"},
            "settings": {"iso": 100, "shutter": "1/250", "aperture": 2.8},
            "tags": ["landscape", "nature"],
        }
        metadata = create_json_metadata(original)
        recovered = read_json_metadata(metadata)
        assert recovered == original


class TestChunkedMetadata:
    """Test chunk-based metadata utilities."""

    def test_chunk_to_bytes(self):
        """MetadataChunk serialization."""
        chunk = MetadataChunk(type_id=b"TEST", data=b"hello")
        data = chunk.to_bytes()
        # Type (4) + length (4) + data (5) = 13 bytes
        assert len(data) == 13
        assert data[:4] == b"TEST"
        assert struct.unpack("<I", data[4:8])[0] == 5
        assert data[8:] == b"hello"

    def test_chunk_from_bytes(self):
        """MetadataChunk parsing."""
        data = b"TEST\x05\x00\x00\x00hello"
        chunk, consumed = MetadataChunk.from_bytes(data)
        assert chunk.type_id == b"TEST"
        assert chunk.data == b"hello"
        assert consumed == 13

    def test_chunk_type_must_be_4_bytes(self):
        """Chunk type must be exactly 4 bytes."""
        chunk = MetadataChunk(type_id=b"BAD", data=b"test")
        with pytest.raises(ValueError):
            chunk.to_bytes()

    def test_create_chunked(self):
        """Create chunked metadata."""
        chunks = [
            MetadataChunk(type_id=b"HDR1", data=b"header info"),
            MetadataChunk(type_id=b"DATA", data=b"some data"),
        ]
        metadata = create_chunked_metadata(chunks)
        assert validate_metadata(metadata)

    def test_roundtrip_chunked(self):
        """Chunked metadata roundtrip."""
        original = [
            MetadataChunk(type_id=b"INFO", data=b"information"),
            MetadataChunk(type_id=b"EXIF", data=b"\x00\x01\x02\x03"),
        ]
        metadata = create_chunked_metadata(original)
        recovered = read_chunked_metadata(metadata)

        assert len(recovered) == len(original)
        for orig, recv in zip(original, recovered, strict=True):
            assert orig.type_id == recv.type_id
            assert orig.data == recv.data

    def test_find_chunk(self):
        """Find chunk by type ID."""
        chunks = [
            MetadataChunk(type_id=b"AAA1", data=b"first"),
            MetadataChunk(type_id=b"BBB2", data=b"second"),
            MetadataChunk(type_id=b"CCC3", data=b"third"),
        ]
        result = find_chunk(chunks, b"BBB2")
        assert result is not None
        assert result.data == b"second"

    def test_find_chunk_not_found(self):
        """Find chunk returns None for missing type."""
        chunks = [MetadataChunk(type_id=b"AAA1", data=b"data")]
        result = find_chunk(chunks, b"XXXX")
        assert result is None


class TestBinaryMetadata:
    """Test binary metadata utilities."""

    def test_create_binary(self):
        """Create binary metadata."""
        data = bytes([0, 1, 2, 3, 4, 5])
        metadata = create_binary_metadata(data)
        assert validate_metadata(metadata)
        assert metadata[:6] == data

    def test_binary_preserves_content(self):
        """Binary metadata preserves all bytes."""
        original = bytes(range(256))
        metadata = create_binary_metadata(original)
        # Original should be at start, padded at end
        assert metadata[:256] == original


class TestReadFromFile:
    """Test reading metadata from GFWX file data."""

    def _create_gfwx_header_with_metadata(self, metadata: bytes) -> bytes:
        """Helper to create GFWX header with metadata."""
        header = create_default_header(width=16, height=16)
        return write_header(header, metadata=metadata)

    def test_read_no_metadata(self):
        """Read metadata from file with none."""
        data = self._create_gfwx_header_with_metadata(b"")
        metadata = read_metadata_raw(data)
        assert metadata == b""

    def test_read_with_metadata(self):
        """Read metadata from file with some."""
        original = b"TEST" * 4  # 16 bytes
        data = self._create_gfwx_header_with_metadata(original)
        metadata = read_metadata_raw(data)
        assert metadata == original

    def test_get_metadata_size(self):
        """Get metadata size from file data."""
        original = b"ABCD" * 8  # 32 bytes = 8 words
        data = self._create_gfwx_header_with_metadata(original)
        size = get_metadata_size(data)
        assert size == 32

    def test_get_data_start_offset_no_metadata(self):
        """Data offset with no metadata."""
        data = self._create_gfwx_header_with_metadata(b"")
        offset = get_data_start_offset(data)
        assert offset == FIXED_HEADER_SIZE

    def test_get_data_start_offset_with_metadata(self):
        """Data offset with metadata."""
        metadata = b"TEST" * 4  # 16 bytes
        data = self._create_gfwx_header_with_metadata(metadata)
        offset = get_data_start_offset(data)
        assert offset == FIXED_HEADER_SIZE + 16

    def test_read_too_short(self):
        """Read raises for data too short."""
        with pytest.raises(ValueError):
            read_metadata_raw(b"short")


class TestDescribeMetadata:
    """Test metadata description utility."""

    def test_describe_empty(self):
        """Describe empty metadata."""
        desc = describe_metadata(b"")
        assert "No metadata" in desc

    def test_describe_json(self):
        """Describe JSON metadata."""
        metadata = create_json_metadata({"key": "value"})
        desc = describe_metadata(metadata)
        assert "JSON" in desc
        assert "key" in desc

    def test_describe_text(self):
        """Describe text metadata."""
        metadata = create_text_metadata("Hello World")
        desc = describe_metadata(metadata)
        assert "Text" in desc or "Hello" in desc

    def test_describe_binary(self):
        """Describe binary metadata."""
        metadata = bytes([0x00, 0x01, 0x02, 0x03])
        desc = describe_metadata(metadata)
        assert "bytes" in desc.lower() or "word" in desc.lower()


class TestIntegrationWithHeader:
    """Test metadata integration with header module."""

    def test_header_with_metadata_roundtrip(self):
        """Full header with metadata roundtrip."""
        from pygfwx.core.header import parse_header, read_metadata

        # Create header with metadata
        header = create_default_header(width=64, height=64, channels=3)
        original_metadata = create_json_metadata(
            {"author": "PyGFWX", "version": "1.0", "settings": {"quality": header.quality}}
        )

        # Write header with metadata
        data = write_header(header, metadata=original_metadata)

        # Parse header back
        parsed_header, header_size = parse_header(data)

        # Verify header metadata size
        assert parsed_header.metadata_size == len(original_metadata) // 4

        # Read metadata using header module
        recovered = read_metadata(data, parsed_header)
        assert recovered == original_metadata

        # Verify JSON content
        content = read_json_metadata(recovered)
        assert content["author"] == "PyGFWX"
        assert content["settings"]["quality"] == header.quality

    def test_header_size_includes_metadata(self):
        """Header size calculation includes metadata."""
        from pygfwx.core.header import parse_header

        metadata = b"TEST" * 10  # 40 bytes = 10 words
        header = create_default_header(width=32, height=32)
        data = write_header(header, metadata=metadata)

        _, header_size = parse_header(data)
        assert header_size == FIXED_HEADER_SIZE + len(metadata)
