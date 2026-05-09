"""Tests for the debug module (hexdump and visualize)."""

import numpy as np
import pytest

from pygfwx import QUALITY_MAX, encode
from pygfwx.debug.hexdump import hexdump, print_hexdump


# ── Test data ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_gfwx() -> bytes:
    """A minimal GFWX-encoded grayscale image."""
    img = np.zeros((16, 16), dtype=np.uint8)
    img[4:12, 4:12] = 200
    return encode(img, quality=QUALITY_MAX)


@pytest.fixture()
def rgb_gfwx() -> bytes:
    """A GFWX-encoded RGB image with UYV transform."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    return encode(img, quality=QUALITY_MAX, color_transform="uyv")


# ── hexdump tests ─────────────────────────────────────────────────────────────


def test_hexdump_returns_list_of_strings(simple_gfwx):
    lines = hexdump(simple_gfwx)
    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)
    assert len(lines) > 0


def test_hexdump_contains_magic_label(simple_gfwx):
    lines = hexdump(simple_gfwx)
    combined = "\n".join(lines)
    assert "GFWX" in combined


def test_hexdump_contains_header_labels(simple_gfwx):
    lines = hexdump(simple_gfwx)
    combined = "\n".join(lines)
    assert "magic" in combined
    assert "version" in combined
    assert "sizex" in combined
    assert "sizey" in combined


def test_hexdump_with_metadata():
    """Hexdump of a file with metadata labels the metadata region."""
    img = np.zeros((8, 8), dtype=np.uint8)
    # Metadata must be a multiple of 4 bytes
    data = encode(img, quality=QUALITY_MAX, metadata=b"XMP " + b"\x00" * 12)
    lines = hexdump(data)
    combined = "\n".join(lines)
    assert "metadata" in combined


def test_hexdump_with_transform(rgb_gfwx):
    """Hexdump of a file with a color transform labels the transform program."""
    lines = hexdump(rgb_gfwx)
    combined = "\n".join(lines)
    assert "transform" in combined


def test_hexdump_contains_hex_bytes(simple_gfwx):
    """Lines contain expected hex offset markers."""
    lines = hexdump(simple_gfwx)
    # At least one line should contain an offset address pattern
    hex_lines = [l for l in lines if "000000" in l or "000010" in l]
    assert hex_lines, "Expected at least one hex data line with offset"


def test_hexdump_empty_data():
    """Hexdump of empty data should not crash."""
    lines = hexdump(b"")
    # Empty data → regions list is empty → no lines
    assert isinstance(lines, list)


def test_hexdump_short_data():
    """Hexdump of data shorter than 32 bytes should not crash."""
    lines = hexdump(b"GFWX" + b"\x00" * 10)
    assert isinstance(lines, list)
    assert len(lines) > 0


def test_print_hexdump_from_bytes(simple_gfwx, capsys):
    """print_hexdump with bytes argument prints to stdout."""
    print_hexdump(simple_gfwx)
    captured = capsys.readouterr()
    assert "GFWX hexdump" in captured.out
    assert "magic" in captured.out


def test_print_hexdump_from_path(simple_gfwx, tmp_path, capsys):
    """print_hexdump with a path reads the file and prints."""
    path = tmp_path / "test.gfwx"
    path.write_bytes(simple_gfwx)
    print_hexdump(path)
    captured = capsys.readouterr()
    assert "GFWX hexdump" in captured.out


def test_print_hexdump_from_str_path(simple_gfwx, tmp_path, capsys):
    """print_hexdump accepts a string path as well."""
    path = tmp_path / "test.gfwx"
    path.write_bytes(simple_gfwx)
    print_hexdump(str(path))
    captured = capsys.readouterr()
    assert "GFWX hexdump" in captured.out


def test_hexdump_custom_width(simple_gfwx):
    """width parameter changes the number of bytes per hex row."""
    lines_16 = hexdump(simple_gfwx, width=16)
    lines_8 = hexdump(simple_gfwx, width=8)
    # Narrower width → more data rows → more total lines
    assert len(lines_8) >= len(lines_16)


# ── visualize import tests ────────────────────────────────────────────────────


def test_visualize_module_importable():
    """visualize module should be importable without matplotlib."""
    import pygfwx.debug.visualize  # noqa: F401 — just check it imports


def test_visualize_requires_matplotlib_for_plotting():
    """Functions raise ImportError if matplotlib is absent (mock test)."""
    from unittest.mock import patch

    from pygfwx.debug.visualize import _require_matplotlib

    with patch.dict("sys.modules", {"matplotlib": None}):
        with pytest.raises(ImportError, match="matplotlib"):
            _require_matplotlib()
