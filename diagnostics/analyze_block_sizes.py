"""Analyze block sizes between Python and SDK encoders."""

import struct
import sys

import numpy as np

sys.path.insert(0, ".")

from cross_codec.gfwx_sdk import encode as sdk_encode
from src.pygfwx.core.codec import encode as py_encode
from src.pygfwx.core.header import Intent, parse_header


def analyze_encoded_data(data, label, sizex, sizey):
    """Parse and analyze encoded data structure."""
    print(f"\n=== {label} ({len(data)} bytes) ===")

    # Parse header
    header, header_size = parse_header(data)
    print(f"  Header size: {header_size} bytes")
    print(f"  block_size: {header.block_size}")
    print(f"  quality: {header.quality}")
    print(f"  encoder: {header.encoder}")

    block_size_log = header.block_size

    pos = header_size
    step = max(sizex, sizey)
    level = 0

    while step >= 1 and pos < len(data):
        bs = step << block_size_log
        block_count_x = (sizex + bs - 1) // bs
        block_count_y = (sizey + bs - 1) // bs
        num_blocks = block_count_x * block_count_y  # 1 channel

        print(f"\nLevel {level} (step={step}):")
        print(f"  Block grid: {block_count_x}x{block_count_y} = {num_blocks} blocks")
        print(f"  Sizes at byte {pos}:")

        block_sizes = []
        for i in range(num_blocks):
            if pos + 4 > len(data):
                print(f"    ERROR: Ran out of data at block {i}")
                break
            size = struct.unpack_from("<I", data, pos)[0]
            block_sizes.append(size)
            pos += 4

        print(f"    Sizes (words): {block_sizes}")
        total_data = sum(block_sizes) * 4
        print(f"    Total data: {total_data} bytes, data starts at byte {pos}")

        # Skip the data
        pos += total_data

        step //= 2
        level += 1

    print(f"\nEnd position: {pos}, file size: {len(data)}")


def main():
    np.random.seed(42)
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

    py_data = py_encode(img, quality=64, intent=Intent.GENERIC)
    sdk_data = sdk_encode(img, quality=64, intent=Intent.GENERIC)

    print("=" * 60)
    print("8x8 random image comparison")
    print(f"Python: {len(py_data)} bytes, SDK: {len(sdk_data)} bytes")
    print("=" * 60)

    analyze_encoded_data(py_data, "Python", 8, 8)
    analyze_encoded_data(sdk_data, "SDK", 8, 8)


def main():
    np.random.seed(42)
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

    py_data = py_encode(img, quality=64, intent=Intent.GENERIC)
    sdk_data = sdk_encode(img, quality=64, intent=Intent.GENERIC)

    print("=" * 60)
    print("8x8 random image comparison")
    print(f"Python: {len(py_data)} bytes, SDK: {len(sdk_data)} bytes")
    print("=" * 60)

    analyze_encoded_data(py_data, "Python", 8, 8)
    analyze_encoded_data(sdk_data, "SDK", 8, 8)


if __name__ == "__main__":
    main()
