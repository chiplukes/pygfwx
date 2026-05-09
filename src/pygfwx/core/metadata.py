"""
GFWX Metadata Support.

This module provides utilities for reading, writing, and manipulating
optional metadata blocks in GFWX files.

Metadata is stored as an array of 32-bit words immediately after the
fixed header (32 bytes). The SDK comment notes:
    "clients can read metadata themselves by accessing the size (in words)
     at word[7] and the metadata at word[8+]"

The metadata format is application-defined - GFWX just stores raw bytes.
This module provides helpers for common use cases like:
- Key-value string pairs
- JSON data
- Raw binary data
- Chunk-based structured metadata

Metadata Format (in GFWX file):
    - 32-bit word count at header byte 28 (word 7)
    - Raw metadata bytes starting at byte 32 (word 8+)
    - Total size = metadata_word_count * 4 bytes
"""

import json
import struct
from dataclasses import dataclass
from typing import Any

# Fixed header size in bytes (before metadata)
FIXED_HEADER_SIZE = 32


@dataclass
class MetadataChunk:  # cm:c6d7e8 — MetadataChunk: chunk-based metadata (4-byte type + length + data)
    """
    A chunk of metadata with a type identifier.

    Chunks can be concatenated to build structured metadata.
    Each chunk has:
    - 4-byte type identifier (magic)
    - 4-byte length (in bytes, not including header)
    - Variable data
    """

    type_id: bytes  # 4-byte type identifier
    data: bytes  # Chunk data

    def to_bytes(self) -> bytes:
        """Serialize chunk to bytes."""
        if len(self.type_id) != 4:
            raise ValueError("Chunk type_id must be exactly 4 bytes")
        return self.type_id + struct.pack("<I", len(self.data)) + self.data

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple["MetadataChunk", int]:
        """
        Parse a chunk from bytes.

        Returns:
            Tuple of (chunk, bytes_consumed).
        """
        if len(data) - offset < 8:
            raise ValueError("Not enough data for chunk header")

        type_id = data[offset : offset + 4]
        (length,) = struct.unpack("<I", data[offset + 4 : offset + 8])

        if len(data) - offset - 8 < length:
            raise ValueError("Not enough data for chunk body")

        chunk_data = data[offset + 8 : offset + 8 + length]
        return cls(type_id=type_id, data=chunk_data), 8 + length


def read_metadata_raw(data: bytes) -> bytes:  # cm:f9a0b1 — read_metadata_raw(): extract raw metadata bytes from GFWX data
    """
    Read raw metadata bytes from GFWX file data.

    Args:
        data: Complete GFWX file data (must include header).

    Returns:
        Raw metadata bytes (empty if no metadata).

    Raises:
        ValueError: If data is too short or malformed.
    """
    if len(data) < FIXED_HEADER_SIZE:
        raise ValueError(f"Data too short: need {FIXED_HEADER_SIZE} bytes, got {len(data)}")

    # Metadata size is at bytes 28-31 (word 7)
    (metadata_words,) = struct.unpack("<I", data[28:32])

    if metadata_words == 0:
        return b""

    metadata_bytes = metadata_words * 4
    metadata_end = FIXED_HEADER_SIZE + metadata_bytes

    if len(data) < metadata_end:
        raise ValueError(f"Data too short for metadata: need {metadata_end} bytes, got {len(data)}")

    return data[FIXED_HEADER_SIZE:metadata_end]


def get_metadata_size(data: bytes) -> int:
    """
    Get the metadata size in bytes from GFWX file data.

    Args:
        data: GFWX file data (at least 32 bytes).

    Returns:
        Metadata size in bytes.
    """
    if len(data) < FIXED_HEADER_SIZE:
        raise ValueError(f"Data too short: need {FIXED_HEADER_SIZE} bytes")

    (metadata_words,) = struct.unpack("<I", data[28:32])
    return metadata_words * 4


def get_data_start_offset(data: bytes) -> int:
    """
    Get the byte offset where compressed image data starts.

    This is after the fixed header and metadata.

    Args:
        data: GFWX file data (at least 32 bytes).

    Returns:
        Offset to start of compressed data.
    """
    return FIXED_HEADER_SIZE + get_metadata_size(data)


# ==============================================================================
# String/Text Metadata Utilities
# ==============================================================================


def create_text_metadata(text: str, encoding: str = "utf-8") -> bytes:
    """
    Create metadata from a text string.

    The text is encoded and padded to a multiple of 4 bytes.

    Args:
        text: The text to store.
        encoding: Text encoding (default utf-8).

    Returns:
        Padded metadata bytes.
    """
    data = text.encode(encoding)
    return pad_to_word_boundary(data)


def read_text_metadata(metadata: bytes, encoding: str = "utf-8") -> str:
    """
    Read a text string from metadata bytes.

    Padding null bytes are stripped.

    Args:
        metadata: The metadata bytes.
        encoding: Text encoding (default utf-8).

    Returns:
        The decoded text string.
    """
    # Strip trailing null bytes (padding)
    data = metadata.rstrip(b"\x00")
    return data.decode(encoding)


# ==============================================================================
# Key-Value Metadata Utilities
# ==============================================================================


def create_key_value_metadata(pairs: dict[str, str]) -> bytes:
    """
    Create metadata from key-value string pairs.

    Format: "key1=value1\\nkey2=value2\\n..."

    Args:
        pairs: Dictionary of string key-value pairs.

    Returns:
        Padded metadata bytes.
    """
    lines = [f"{key}={value}" for key, value in pairs.items()]
    text = "\n".join(lines)
    return create_text_metadata(text)


def read_key_value_metadata(metadata: bytes) -> dict[str, str]:
    """
    Read key-value pairs from metadata bytes.

    Args:
        metadata: The metadata bytes.

    Returns:
        Dictionary of string key-value pairs.
    """
    text = read_text_metadata(metadata)
    pairs = {}
    for line in text.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


# ==============================================================================
# JSON Metadata Utilities
# ==============================================================================


def create_json_metadata(obj: Any) -> bytes:
    """
    Create metadata from a JSON-serializable object.

    Args:
        obj: JSON-serializable Python object.

    Returns:
        Padded metadata bytes containing JSON.
    """
    json_str = json.dumps(obj)
    return create_text_metadata(json_str)


def read_json_metadata(metadata: bytes) -> Any:
    """
    Read a JSON object from metadata bytes.

    Args:
        metadata: The metadata bytes.

    Returns:
        Deserialized Python object.
    """
    text = read_text_metadata(metadata)
    return json.loads(text)


# ==============================================================================
# Chunk-Based Metadata Utilities
# ==============================================================================


def create_chunked_metadata(chunks: list[MetadataChunk]) -> bytes:
    """
    Create metadata from a list of chunks.

    Each chunk has a 4-byte type and variable-length data.
    The result is padded to a word boundary.

    Args:
        chunks: List of MetadataChunk objects.

    Returns:
        Padded metadata bytes.
    """
    data = b"".join(chunk.to_bytes() for chunk in chunks)
    return pad_to_word_boundary(data)


def read_chunked_metadata(metadata: bytes) -> list[MetadataChunk]:
    """
    Read chunks from metadata bytes.

    Args:
        metadata: The metadata bytes.

    Returns:
        List of MetadataChunk objects.
    """
    chunks = []
    offset = 0
    while offset < len(metadata):
        # Check if remaining data is just padding
        remaining = metadata[offset:]
        if remaining == b"\x00" * len(remaining):
            break
        if len(remaining) < 8:
            break  # Not enough for a chunk header
        chunk, consumed = MetadataChunk.from_bytes(metadata, offset)
        chunks.append(chunk)
        offset += consumed
    return chunks


def find_chunk(chunks: list[MetadataChunk], type_id: bytes) -> MetadataChunk | None:
    """
    Find a chunk by type ID.

    Args:
        chunks: List of chunks to search.
        type_id: 4-byte type identifier to find.

    Returns:
        The chunk if found, None otherwise.
    """
    for chunk in chunks:
        if chunk.type_id == type_id:
            return chunk
    return None


# ==============================================================================
# Binary Metadata Utilities
# ==============================================================================


def create_binary_metadata(data: bytes) -> bytes:
    """
    Create metadata from raw binary data.

    Pads to word boundary with null bytes.

    Args:
        data: Raw binary data.

    Returns:
        Padded metadata bytes (multiple of 4).
    """
    return pad_to_word_boundary(data)


def pad_to_word_boundary(data: bytes) -> bytes:
    """
    Pad data to a 4-byte (word) boundary.

    Args:
        data: Input bytes.

    Returns:
        Padded bytes (length is multiple of 4).
    """
    remainder = len(data) % 4
    if remainder == 0:
        return data
    padding = 4 - remainder
    return data + b"\x00" * padding


def validate_metadata(metadata: bytes) -> bool:
    """
    Validate that metadata is properly formatted.

    Args:
        metadata: The metadata bytes.

    Returns:
        True if valid (length is multiple of 4), False otherwise.
    """
    return len(metadata) % 4 == 0


def get_metadata_word_count(metadata: bytes) -> int:
    """
    Get the number of 32-bit words in metadata.

    Args:
        metadata: The metadata bytes.

    Returns:
        Number of 32-bit words.

    Raises:
        ValueError: If metadata length is not a multiple of 4.
    """
    if len(metadata) % 4 != 0:
        raise ValueError(f"Metadata length {len(metadata)} is not a multiple of 4")
    return len(metadata) // 4


# ==============================================================================
# Metadata Description
# ==============================================================================


def describe_metadata(metadata: bytes) -> str:
    """
    Generate a human-readable description of metadata.

    Attempts to detect the format and provide appropriate info.

    Args:
        metadata: The metadata bytes.

    Returns:
        Human-readable description string.
    """
    if len(metadata) == 0:
        return "No metadata"

    lines = [f"Metadata: {len(metadata)} bytes ({get_metadata_word_count(metadata)} words)"]

    # Try to detect format
    try:
        # Try JSON
        obj = read_json_metadata(metadata)
        lines.append("Format: JSON")
        lines.append(f"Content: {obj!r}")
        return "\n".join(lines)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    try:
        # Try key-value
        pairs = read_key_value_metadata(metadata)
        if pairs and all("=" not in v for v in pairs.values()):
            lines.append("Format: Key-Value pairs")
            for k, v in pairs.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)
    except UnicodeDecodeError:
        pass

    try:
        # Try plain text
        text = read_text_metadata(metadata)
        if text.isprintable() or text.replace("\n", "").replace("\t", "").isprintable():
            lines.append("Format: Text")
            lines.append(f"Content: {text!r}")
            return "\n".join(lines)
    except UnicodeDecodeError:
        pass

    try:
        # Try chunks
        chunks = read_chunked_metadata(metadata)
        if chunks:
            lines.append(f"Format: Chunked ({len(chunks)} chunks)")
            for i, chunk in enumerate(chunks):
                type_str = chunk.type_id.decode("ascii", errors="replace")
                lines.append(f"  Chunk {i}: type={type_str!r}, {len(chunk.data)} bytes")
            return "\n".join(lines)
    except Exception:
        pass

    # Fall back to hex dump
    lines.append("Format: Binary")
    hex_preview = metadata[:32].hex()
    if len(metadata) > 32:
        hex_preview += "..."
    lines.append(f"Hex: {hex_preview}")

    return "\n".join(lines)
