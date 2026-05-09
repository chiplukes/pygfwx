"""
Tests for the pygfwx CLI — compress, decompress, info subcommands.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from pygfwx.utils.image_io import load_image, save_image
from pygfwx.utils.reference_images import get_reference_mono, get_reference_rgb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the pygfwx CLI via the installed entry point."""
    return subprocess.run(
        [sys.executable, "-m", "pygfwx.cli", *args],
        capture_output=True,
        text=True,
    )


def make_test_png(tmp_path, name="test.png", channels=3) -> tuple:
    """Write a reference image to a temp PNG and return (path, array)."""
    if channels == 1:
        img = get_reference_mono()
    else:
        img = get_reference_rgb()
    path = tmp_path / name
    save_image(img, path)
    return path, img


# ---------------------------------------------------------------------------
# No-args / help
# ---------------------------------------------------------------------------


def test_no_args_exits_nonzero():
    result = run_cli()
    assert result.returncode != 0


def test_help_exits_zero():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "compress" in result.stdout
    assert "decompress" in result.stdout
    assert "info" in result.stdout


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compress_lossless(tmp_path):
    png_path, original = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"

    result = run_cli("compress", str(png_path), str(gfwx_path))
    assert result.returncode == 0, result.stderr
    assert gfwx_path.exists()
    assert gfwx_path.stat().st_size > 0
    assert "lossless" in result.stdout


def test_compress_lossy(tmp_path):
    png_path, _ = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"

    result = run_cli("compress", str(png_path), str(gfwx_path), "-q", "256")
    assert result.returncode == 0, result.stderr
    assert "q=256" in result.stdout


def test_compress_cubic_filter(tmp_path):
    png_path, _ = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"

    result = run_cli("compress", str(png_path), str(gfwx_path), "--filter", "cubic")
    assert result.returncode == 0, result.stderr


def test_compress_encoder_fast(tmp_path):
    png_path, _ = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"

    result = run_cli("compress", str(png_path), str(gfwx_path), "--encoder", "fast")
    assert result.returncode == 0, result.stderr


def test_compress_missing_input(tmp_path):
    result = run_cli("compress", str(tmp_path / "nonexistent.png"), str(tmp_path / "out.gfwx"))
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_compress_invalid_quality(tmp_path):
    png_path, _ = make_test_png(tmp_path)
    result = run_cli("compress", str(png_path), str(tmp_path / "out.gfwx"), "-q", "9999")
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# decompress
# ---------------------------------------------------------------------------


def test_decompress_roundtrip_lossless(tmp_path):
    """Lossless compress then decompress should recover the original image."""
    png_path, original = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"
    out_png = tmp_path / "recovered.png"

    run_cli("compress", str(png_path), str(gfwx_path))
    result = run_cli("decompress", str(gfwx_path), str(out_png))
    assert result.returncode == 0, result.stderr
    assert out_png.exists()

    recovered = load_image(out_png)
    assert np.array_equal(recovered, original)


def test_decompress_mono(tmp_path):
    """Mono image round-trip via CLI."""
    png_path, original = make_test_png(tmp_path, "mono.png", channels=1)
    gfwx_path = tmp_path / "mono.gfwx"
    out_png = tmp_path / "mono_out.png"

    run_cli("compress", str(png_path), str(gfwx_path))
    result = run_cli("decompress", str(gfwx_path), str(out_png))
    assert result.returncode == 0, result.stderr

    recovered = load_image(out_png)
    assert np.array_equal(recovered, original)


def test_decompress_downsample(tmp_path):
    """Downsampled decode should produce a smaller image."""
    png_path, original = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"
    out_png = tmp_path / "half.png"

    run_cli("compress", str(png_path), str(gfwx_path))
    result = run_cli("decompress", str(gfwx_path), str(out_png), "--downsample", "1")
    assert result.returncode == 0, result.stderr

    recovered = load_image(out_png)
    assert recovered.shape[0] == original.shape[0] // 2
    assert recovered.shape[1] == original.shape[1] // 2


def test_decompress_missing_input(tmp_path):
    result = run_cli("decompress", str(tmp_path / "nonexistent.gfwx"), str(tmp_path / "out.png"))
    assert result.returncode != 0
    assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_shows_header_fields(tmp_path):
    png_path, _ = make_test_png(tmp_path)
    gfwx_path = tmp_path / "out.gfwx"

    run_cli("compress", str(png_path), str(gfwx_path))
    result = run_cli("info", str(gfwx_path))
    assert result.returncode == 0, result.stderr

    # Check key fields appear in output
    assert "Dimensions" in result.stdout
    assert "Channels" in result.stdout
    assert "Bit depth" in result.stdout
    assert "Quality" in result.stdout
    assert "lossless" in result.stdout
    assert "Filter" in result.stdout
    assert "Intent" in result.stdout


def test_info_missing_file(tmp_path):
    result = run_cli("info", str(tmp_path / "nonexistent.gfwx"))
    assert result.returncode != 0
    assert "not found" in result.stderr
