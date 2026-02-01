"""
Command-line interface for PyGFWX.

Usage:
    pygfwx compress INPUT OUTPUT [-q QUALITY] [--filter cubic|linear]
    pygfwx decompress INPUT OUTPUT [--downsample LEVEL]
    pygfwx info FILE
"""

import argparse
import sys


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pygfwx",
        description="PyGFWX - Python GFWX Wavelet Codec",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress an image")
    compress_parser.add_argument("input", help="Input image file")
    compress_parser.add_argument("output", help="Output GFWX file")
    compress_parser.add_argument(
        "-q", "--quality", type=int, default=512, help="Quality (1-1024, default: 512)"
    )
    compress_parser.add_argument(
        "--filter",
        choices=["linear", "cubic"],
        default="linear",
        help="Wavelet filter (default: linear)",
    )

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a GFWX file")
    decompress_parser.add_argument("input", help="Input GFWX file")
    decompress_parser.add_argument("output", help="Output image file")
    decompress_parser.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="Downsampling level (0=full, 1=half, etc.)",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show GFWX file information")
    info_parser.add_argument("file", help="GFWX file to inspect")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "compress":
        print(f"Compressing {args.input} -> {args.output}")
        print(f"  Quality: {args.quality}")
        print(f"  Filter: {args.filter}")
        print("  (Not yet implemented)")
        sys.exit(1)

    elif args.command == "decompress":
        print(f"Decompressing {args.input} -> {args.output}")
        print(f"  Downsample: {args.downsample}")
        print("  (Not yet implemented)")
        sys.exit(1)

    elif args.command == "info":
        print(f"File: {args.file}")
        print("  (Not yet implemented)")
        sys.exit(1)


if __name__ == "__main__":
    main()
