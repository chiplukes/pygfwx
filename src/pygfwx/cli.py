"""
Command-line interface for PyGFWX.

Usage:
    pygfwx compress INPUT OUTPUT [-q QUALITY] [--filter cubic|linear] [--encoder MODE]
    pygfwx decompress INPUT OUTPUT [--downsample LEVEL]
    pygfwx info FILE
"""

import argparse
import sys
from pathlib import Path


def main():  # cm:d4e5f6b — main(): CLI entry point (compress/decompress/info subcommands)
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pygfwx",
        description="PyGFWX - Python GFWX Wavelet Codec",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress an image to GFWX format")
    compress_parser.add_argument("input", help="Input image file (PNG, JPEG, TIFF)")
    compress_parser.add_argument("output", help="Output GFWX file")
    compress_parser.add_argument(
        "-q", "--quality", type=int, default=1024, help="Quality (1-1024, default: 1024 = lossless)"
    )
    compress_parser.add_argument(
        "--filter",
        choices=["linear", "cubic"],
        default="linear",
        help="Wavelet filter: linear (5/3, best lossless) or cubic (9/7, best lossy). Default: linear",
    )
    compress_parser.add_argument(
        "--encoder",
        choices=["contextual", "fast", "high-bitrate", "turbo"],
        default="contextual",
        help="Encoder mode (default: contextual)",
    )

    compress_parser.add_argument(
        "--transform",
        choices=["uyv", "a710"],
        default=None,
        help="Color transform before lifting: uyv (YUV-like) or a710 (high-quality). Default: none",
    )

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a GFWX file")
    decompress_parser.add_argument("input", help="Input GFWX file")
    decompress_parser.add_argument("output", help="Output image file (PNG, TIFF, etc.)")
    decompress_parser.add_argument(
        "--downsample",
        type=int,
        default=0,
        metavar="LEVEL",
        help="Decode at reduced resolution: 0=full, 1=half, 2=quarter, … (default: 0)",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show GFWX file information")
    info_parser.add_argument("file", help="GFWX file to inspect")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compress":
        _cmd_compress(args)
    elif args.command == "decompress":
        _cmd_decompress(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_compress(args: argparse.Namespace) -> None:
    """Implement the compress subcommand."""
    from pygfwx import QUALITY_MAX, encode
    from pygfwx.core.header import Encoder, Filter
    from pygfwx.utils.image_io import get_bit_depth, load_image

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    quality = args.quality
    if not (1 <= quality <= QUALITY_MAX):
        print(f"Error: quality must be between 1 and {QUALITY_MAX}, got {quality}", file=sys.stderr)
        sys.exit(1)

    filter_map = {"linear": Filter.LINEAR, "cubic": Filter.CUBIC}
    encoder_map = {
        "contextual": Encoder.CONTEXTUAL,
        "fast": Encoder.FAST,
        "high-bitrate": Encoder.HIGH_BITRATE,
        "turbo": Encoder.TURBO,
    }

    try:
        image = load_image(input_path)
    except Exception as exc:
        print(f"Error loading image: {exc}", file=sys.stderr)
        sys.exit(1)

    h = image.shape[0]
    w = image.shape[1]
    channels = 1 if image.ndim == 2 else image.shape[2]
    bit_depth = get_bit_depth(image)

    print(f"Input:  {input_path}  ({w}x{h}, {channels}ch, {bit_depth}-bit, {image.nbytes:,} bytes)")

    try:
        compressed = encode(
            image,
            quality=quality,
            filter=filter_map[args.filter],
            encoder=encoder_map[args.encoder],
            color_transform=args.transform,
        )
    except Exception as exc:
        print(f"Error during compression: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        output_path.write_bytes(compressed)
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        sys.exit(1)

    ratio = image.nbytes / len(compressed) if compressed else 0
    mode = "lossless" if quality == QUALITY_MAX else f"lossy q={quality}"
    print(f"Output: {output_path}  ({len(compressed):,} bytes, {ratio:.2f}x, {mode})")


def _cmd_decompress(args: argparse.Namespace) -> None:
    """Implement the decompress subcommand."""
    from pygfwx import decode
    from pygfwx.utils.image_io import save_image

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = input_path.read_bytes()
    except OSError as exc:
        print(f"Error reading input: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        image = decode(data, downsampling=args.downsample)
    except Exception as exc:
        print(f"Error during decompression: {exc}", file=sys.stderr)
        sys.exit(1)

    h = image.shape[0]
    w = image.shape[1]
    channels = 1 if image.ndim == 2 else image.shape[2]
    print(f"Decoded: {w}x{h}, {channels}ch, {image.dtype}")

    try:
        save_image(image, output_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {output_path}  ({output_path.stat().st_size:,} bytes)")


def _cmd_info(args: argparse.Namespace) -> None:
    """Implement the info subcommand."""
    from pygfwx import get_header

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        header = get_header(data)
    except Exception as exc:
        print(f"Error parsing header: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"File:         {file_path}  ({len(data):,} bytes)")
    print(f"Version:      {header.version}")
    print(f"Dimensions:   {header.sizex} x {header.sizey}")
    print(f"Layers:       {header.layers}")
    print(f"Channels:     {header.channels}")
    print(f"Bit depth:    {header.bit_depth}-bit {'signed' if header.is_signed else 'unsigned'}")
    print(f"Quality:      {header.quality}" + (" (lossless)" if header.is_lossless else ""))
    print(f"Chroma scale: {header.chroma_scale}")
    print(f"Block size:   {header.block_size}")
    print(f"Filter:       {header.filter.name}")
    print(f"Encoder:      {header.encoder.name}")
    print(f"Intent:       {header.intent.name}")
    if header.metadata_size > 0:
        print(f"Metadata:     {header.metadata_size * 4} bytes")


if __name__ == "__main__":
    main()

