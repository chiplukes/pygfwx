"""
GFWX Bitstream Reader/Writer.

This module provides the Bits class for reading and writing bits from/to
a buffer of 32-bit words. It is the Python equivalent of the C++ Bits struct.

The bitstream is stored as an array of 32-bit unsigned integers. Bits are
read/written from the MSB of each word first. The bit index tracks position
within the current word (0-31).
"""

import numpy as np


class BitstreamOverflowError(Exception):
    """Raised when the bitstream buffer overflows."""

    pass


class BitReader:  # cm:c3d4e5 — BitReader: word-aligned bit extraction from compressed stream
    """
    Reads bits from a buffer of 32-bit words.

    Bits are read from MSB to LSB within each word. The reader tracks
    the current word position and bit position within that word.

    Attributes:
        buffer: numpy array of uint32 words
        word_index: current word position in buffer
        bit_index: bit position within current word (0-31)
        overflow: True if buffer overflow occurred
    """

    def __init__(self, data: bytes | np.ndarray):
        """
        Initialize the bit reader.

        Args:
            data: Either raw bytes or a numpy array of uint32 values.
                  If bytes, must be a multiple of 4 in length.

        Note:
            The GFWX bitstream stores words in little-endian byte order.
            Bits are accessed MSB-first within each word (native value).
        """
        if isinstance(data, bytes):
            # Pad to 4-byte boundary if needed
            padding = (4 - len(data) % 4) % 4
            if padding:
                data = data + b"\x00" * padding
            # Convert to uint32 array - little-endian (file format)
            self.buffer = np.frombuffer(data, dtype="<u4").copy()
        else:
            self.buffer = np.asarray(data, dtype=np.uint32)

        self.word_index = 0
        self.bit_index = 0
        self.overflow = False

    @property
    def buffer_end(self) -> int:
        """Return the number of words in the buffer."""
        return len(self.buffer)

    def _check_overflow(self) -> bool:
        """Check and set overflow flag if at end of buffer."""
        if self.word_index >= self.buffer_end:
            self.overflow = True
            return True
        return False

    def get_bits(self, bits: int) -> int:  # cm:f6a7b8 — get_bits(): read N bits MSB-first
        """
        Read n bits from the stream.

        Bits are read from MSB to LSB. If reading would overflow the
        buffer, returns 0 and sets the overflow flag.

        Args:
            bits: Number of bits to read (1-32)

        Returns:
            The unsigned integer value of the bits read.

        Raises:
            BitstreamOverflowError: If buffer overflow occurs.
        """
        if bits == 0:
            return 0

        if self._check_overflow():
            raise BitstreamOverflowError("Buffer overflow in get_bits")

        new_bits = self.bit_index + bits

        # Get bits from current word, shifted to put requested bits at top
        x = (int(self.buffer[self.word_index]) << self.bit_index) & 0xFFFFFFFF

        if new_bits >= 32:
            # Need to advance to next word
            self.word_index += 1
            new_bits -= 32

            if new_bits > 0:
                # Need bits from the next word
                if self._check_overflow():
                    raise BitstreamOverflowError("Buffer overflow in get_bits (cross-word)")
                # Get remaining bits from next word
                x |= int(self.buffer[self.word_index]) >> (32 - self.bit_index)

        self.bit_index = new_bits

        # Shift to get the requested bits as the low bits
        return (x >> (32 - bits)) & ((1 << bits) - 1)

    def get_zeros(self, max_zeros: int) -> int:  # cm:c9d0e1 — get_zeros(): unary zero run count (Golomb prefix)
        """
        Read unary zeros from the stream until a 1-bit or max_zeros reached.

        This is used for Golomb-Rice coding. Counts consecutive 0 bits
        until a 1-bit is found or max_zeros is reached. The terminating
        1-bit is consumed but not counted.

        Args:
            max_zeros: Maximum number of zeros to read before stopping.

        Returns:
            The count of zeros read (0 to max_zeros).

        Raises:
            BitstreamOverflowError: If buffer overflow occurs.
        """
        if self._check_overflow():
            raise BitstreamOverflowError("Buffer overflow in get_zeros")

        new_bits = self.bit_index
        b = int(self.buffer[self.word_index])
        x = 0

        while True:
            if new_bits == 31:
                # At last bit of word
                self.word_index += 1
                if (b & 1) or (x + 1 == max_zeros):
                    # Found a 1-bit at the last position, or reached max
                    self.bit_index = 0
                    if not (b & 1):
                        x += 1  # Count the zero if we stopped due to max
                    return x

                x += 1  # Count the zero at position 31

                if self._check_overflow():
                    raise BitstreamOverflowError("Buffer overflow in get_zeros (cross-word)")

                b = int(self.buffer[self.word_index])
                new_bits = 0
                continue

            # Check bit at position new_bits (from MSB)
            bit_mask = 1 << (31 - new_bits)
            if (b & bit_mask) or (x + 1 == max_zeros):
                # Found a 1-bit or reached max
                self.bit_index = new_bits + 1
                if not (b & bit_mask):
                    x += 1  # Count the zero if we stopped due to max
                return x

            x += 1
            new_bits += 1

    def flush_read_word(self) -> None:
        """
        Advance to the start of the next word.

        This skips any remaining bits in the current word.
        """
        if self.bit_index <= 0:
            return
        self.word_index += 1
        self.bit_index = 0

    @property
    def position_bits(self) -> int:
        """Return the current bit position in the stream."""
        return self.word_index * 32 + self.bit_index

    @property
    def remaining_bits(self) -> int:
        """Return the number of bits remaining in the stream."""
        total_bits = self.buffer_end * 32
        return total_bits - self.position_bits


class BitWriter:  # cm:f2a3b4 — BitWriter: bit packing into 32-bit word buffer
    """
    Writes bits to a buffer of 32-bit words.

    Bits are written from MSB to LSB within each word. The writer tracks
    the current word position and bit position, plus a write cache for
    the current word being assembled.

    Attributes:
        buffer: numpy array of uint32 words (output)
        word_index: current word position in buffer
        bit_index: bit position within current word (0-31)
        write_cache: current word being assembled
        overflow: True if buffer overflow occurred
    """

    def __init__(self, size: int):
        """
        Initialize the bit writer with a buffer of given size.

        Args:
            size: Number of 32-bit words to allocate.
        """
        self.buffer = np.zeros(size, dtype=np.uint32)
        self.word_index = 0
        self.bit_index = 0
        self.write_cache = 0
        self.overflow = False

    @property
    def buffer_end(self) -> int:
        """Return the number of words in the buffer."""
        return len(self.buffer)

    def put_bits(self, x: int, bits: int) -> None:  # cm:c5d6e7 — put_bits(): write N bits MSB-first
        """
        Write n bits to the stream.

        Bits are written from MSB to LSB. The value x should have
        at most 'bits' significant bits.

        Args:
            x: The value to write (only lowest 'bits' bits are used)
            bits: Number of bits to write (1-32)

        Raises:
            BitstreamOverflowError: If buffer overflow occurs.
        """
        if bits == 0:
            return

        x = x & ((1 << bits) - 1)  # Mask to requested bits

        new_bits = self.bit_index + bits

        if self.word_index >= self.buffer_end:
            self.overflow = True
            raise BitstreamOverflowError("Buffer overflow in put_bits")

        if new_bits < 32:
            # Fits in current cache
            self.write_cache = ((self.write_cache << bits) | x) & 0xFFFFFFFF
        elif bits == 32 and new_bits == 32:
            # Exactly one full word
            new_bits = 0
            self.buffer[self.word_index] = x
            self.word_index += 1
            self.write_cache = 0
        else:
            # Crosses word boundary
            new_bits -= 32
            # Write completed word
            word = ((self.write_cache << (bits - new_bits)) | (x >> new_bits)) & 0xFFFFFFFF
            self.buffer[self.word_index] = word
            self.word_index += 1
            self.write_cache = x & ((1 << new_bits) - 1) if new_bits > 0 else 0

        self.bit_index = new_bits

    def flush_write_word(self) -> None:
        """
        Flush any remaining bits in the cache to the buffer.

        Pads with zeros to complete the current word.
        """
        remaining = (32 - self.bit_index) % 32
        if remaining > 0:
            self.put_bits(0, remaining)

    def get_data(self) -> bytes:
        """
        Get the written data as bytes.

        Flushes the write cache first to ensure all bits are written.

        Returns:
            The written data as a bytes object.
        """
        self.flush_write_word()
        return self.buffer[: self.word_index].tobytes()

    @property
    def position_bits(self) -> int:
        """Return the current bit position in the stream."""
        return self.word_index * 32 + self.bit_index

    @property
    def bytes_written(self) -> int:
        """Return the number of complete bytes written."""
        return (self.position_bits + 7) // 8
