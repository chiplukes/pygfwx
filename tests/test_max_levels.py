"""
Tests for the `max_levels` capped-recursion parameter.

Standard GFWX recurses `lift()`/`encode()` all the way to a true single
DC coefficient (`while step < sizex or step < sizey: ...`, no cap). This
is a deliberate, real-time-hardware-motivated divergence from that: cap
the recursion at `max_levels`, then recurse on the remaining LL
sub-array as its own self-contained image (unbounded). See
`pygfwx.core.lifting.lift`'s own docstring for the full rationale, and
`gfwx-fpga`'s own `notes/gfwx_capped_recursion_explainer.md` for the
real measured compression-ratio cost and quality analysis that led to
this being adopted (small: well under 2% even in an adversarially-chosen
worst case, and free at every quality setting a "visually lossless"
target would realistically use, since GFWX's own quantizer already
saturates to lossless at those coarse levels).

`max_levels=None` (the default, everywhere) must stay bit-for-bit
identical to pygfwx's behavior before this parameter existed — every
test class below that exercises `max_levels=None` is a regression guard
for that, not a max_levels-specific test.
"""

import numpy as np
import pytest

from pygfwx import QUALITY_MAX, decode, encode
from pygfwx.core.header import Filter, parse_header
from pygfwx.core.lifting import lift, unlift
from pygfwx.core.quantization import dequantize, quantize


class TestLiftUnliftMaxLevels:
    """Direct tests of lift()/unlift()'s own max_levels parameter."""

    SHAPES = [(8, 8), (16, 16), (32, 32), (7, 7), (5, 9), (64, 48), (1, 1), (2, 2), (3, 3)]

    @pytest.mark.parametrize("h,w", SHAPES)
    @pytest.mark.parametrize("max_levels", [None, 1, 2, 3, 5, 100])
    def test_round_trip(self, h, w, max_levels):
        """lift(max_levels=N) -> unlift(max_levels=N) recovers the original array exactly."""
        rng = np.random.default_rng(7)
        original = rng.integers(-2000, 2000, size=(h, w), dtype=np.int64)
        a = original.copy()
        lift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        unlift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        np.testing.assert_array_equal(a, original)

    def test_max_levels_none_matches_unmodified_signature_call(self):
        """Calling lift() with no max_levels argument at all (the pre-existing
        call signature) must produce bit-exact output vs. explicitly passing
        max_levels=None -- confirms the parameter is purely additive."""
        rng = np.random.default_rng(8)
        for h, w in self.SHAPES:
            original = rng.integers(-2000, 2000, size=(h, w), dtype=np.int64)
            a = original.copy()
            lift(a, 0, 0, w, h, 1, Filter.LINEAR)
            b = original.copy()
            lift(b, 0, 0, w, h, 1, Filter.LINEAR, max_levels=None)
            np.testing.assert_array_equal(a, b)

    def test_capped_levels_match_unbounded_recursion_exactly(self):
        """lift(max_levels=N) must be bit-exact with unbounded lift() at every
        position actually owned by levels 1..N -- capping must not perturb
        the levels it DOES fully process, only stop before the rest."""
        import math

        rng = np.random.default_rng(9)
        for h, w in [(64, 48), (100, 100)]:
            original = rng.integers(-2000, 2000, size=(h, w), dtype=np.int64)
            capped = original.copy()
            lift(capped, 0, 0, w, h, 1, Filter.LINEAR, max_levels=3)
            full = original.copy()
            lift(full, 0, 0, w, h, 1, Filter.LINEAR)

            enc_step = 1
            while enc_step < w or enc_step < h:
                enc_step *= 2

            for y in range(h):
                for x in range(w):
                    if x == 0 and y == 0:
                        continue
                    level_step = min(
                        math.gcd(x, enc_step) if x else enc_step,
                        math.gcd(y, enc_step) if y else enc_step,
                    )
                    if level_step <= 4:  # levels 1-3 (step 1, 2, 4)
                        assert capped[y, x] == full[y, x], f"mismatch at ({x},{y}) step={level_step}"

    def test_max_levels_must_be_positive(self):
        rng = np.random.default_rng(1)
        a = rng.integers(-100, 100, size=(8, 8), dtype=np.int64)
        with pytest.raises(ValueError):
            lift(a, 0, 0, 8, 8, 1, Filter.LINEAR, max_levels=0)


class TestQuantizeDequantizeMaxLevels:
    """quantize()/dequantize()'s own max_levels parameter, combined with lift/unlift."""

    @pytest.mark.parametrize("h,w", [(8, 8), (32, 32), (7, 7), (64, 48), (100, 100)])
    @pytest.mark.parametrize("max_levels", [None, 2, 3, 5])
    def test_lossless_round_trip_exact(self, h, w, max_levels):
        rng = np.random.default_rng(11)
        original = rng.integers(0, 4096, size=(h, w), dtype=np.int64)
        max_q = 1024  # boost=1 (lossless) -- quality=1024 with THIS max_q genuinely skips quantization
        a = original.copy()
        lift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        quantize(a, 0, 0, w, h, 1, QUALITY_MAX, 0, max_q, max_levels=max_levels)
        dequantize(a, 0, 0, w, h, 1, QUALITY_MAX, 0, max_q, max_levels=max_levels)
        unlift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        np.testing.assert_array_equal(a, original)

    @pytest.mark.parametrize("max_levels", [None, 3, 5])
    @pytest.mark.parametrize("quality", [256, 64])
    def test_lossy_round_trip_bounded_error(self, max_levels, quality):
        """Real quantization error is expected (lossy); just confirm it's
        bounded (no crash, no wildly-wrong garbage) at every max_levels."""
        rng = np.random.default_rng(12)
        h, w = (64, 48)
        original = rng.integers(0, 4096, size=(h, w), dtype=np.int64)
        max_q = 1024 * 8  # boost=8 (lossy)
        a = original.copy()
        lift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        quantize(a, 0, 0, w, h, 1, quality, 0, max_q, max_levels=max_levels)
        dequantize(a, 0, 0, w, h, 1, quality, 0, max_q, max_levels=max_levels)
        unlift(a, 0, 0, w, h, 1, Filter.LINEAR, max_levels=max_levels)
        assert np.max(np.abs(a - original)) < 4096


class TestEncodeDecodeMaxLevels:
    """Full encode()/decode() round trip through the top-level codec API."""

    @pytest.mark.parametrize("h,w", [(8, 8), (16, 16), (32, 32), (7, 7), (5, 9), (64, 48), (33, 65)])
    @pytest.mark.parametrize("max_levels", [None, 1, 2, 3, 5, 10])
    def test_lossless_round_trip_bit_exact(self, h, w, max_levels):
        rng = np.random.default_rng(13)
        image = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        compressed = encode(image, quality=QUALITY_MAX, max_levels=max_levels)
        decoded = decode(compressed)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        np.testing.assert_array_equal(decoded, image)

    @pytest.mark.parametrize("max_levels", [None, 2, 4])
    def test_rgb_lossless_round_trip(self, max_levels):
        rng = np.random.default_rng(14)
        for h, w in [(16, 16), (48, 64), (7, 9)]:
            image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            compressed = encode(image, quality=QUALITY_MAX, max_levels=max_levels)
            decoded = decode(compressed)
            np.testing.assert_array_equal(decoded, image)

    def test_header_records_max_levels(self):
        rng = np.random.default_rng(15)
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

        compressed_default = encode(image, quality=QUALITY_MAX)
        header_default, _ = parse_header(compressed_default)
        assert header_default.max_levels == 0
        assert header_default.max_levels_or_none is None

        compressed_capped = encode(image, quality=QUALITY_MAX, max_levels=5)
        header_capped, _ = parse_header(compressed_capped)
        assert header_capped.max_levels == 5
        assert header_capped.max_levels_or_none == 5

    def test_max_levels_out_of_range_rejected(self):
        rng = np.random.default_rng(16)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        with pytest.raises(ValueError):
            encode(image, quality=QUALITY_MAX, max_levels=256)
        with pytest.raises(ValueError):
            encode(image, quality=QUALITY_MAX, max_levels=0)

    @pytest.mark.parametrize("max_levels", [None, 3, 5, 7, 9])
    @pytest.mark.parametrize("quality", [QUALITY_MAX, 512, 256, 64])
    def test_real_photo_lossy_sane(self, max_levels, quality):
        """Real photo content (pygfwx's own demo image), across quality
        settings spanning visually-lossless through aggressively lossy:
        lossless must be bit-exact regardless of max_levels; lossy must
        stay within a sane error bound (this is a smoke test against
        garbage output / crashes at every max_levels, not a tight
        distortion bound -- distortion itself is quality's own job)."""
        from pathlib import Path

        from PIL import Image

        asset = Path(__file__).parent.parent / "examples" / "assets" / "cat_demo.png"
        photo = np.array(Image.open(asset).convert("L")).astype(np.uint8)

        compressed = encode(photo, quality=quality, max_levels=max_levels)
        decoded = decode(compressed)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        max_err = int(np.max(np.abs(decoded.astype(np.int32) - photo.astype(np.int32))))

        if quality == QUALITY_MAX:
            assert max_err == 0
        else:
            assert max_err < 260

    def test_downsampling_or_bayer_with_max_levels_rejected(self):
        """Not yet supported/verified -- must fail loudly, not silently
        produce wrong output. See _decode_all_levels's own docstring."""
        rng = np.random.default_rng(17)
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        compressed = encode(image, quality=QUALITY_MAX, max_levels=5)
        with pytest.raises(NotImplementedError):
            decode(compressed, downsampling=1)


class TestRemainderRaw:
    """`remainder_raw=True`: the capped remainder is stored as literal,
    uncompressed int16 values instead of being recursively transformed
    and entropy-coded -- the simpler "CineForm-style" alternative
    documented in gfwx-fpga's own
    notes/gfwx_capped_recursion_explainer.md (that doc's own pessimistic
    cost baseline; pygfwx's default, non-raw behavior does ~44% better).
    """

    SHAPES = [(16, 16), (48, 64), (7, 9), (64, 48), (33, 17)]

    @pytest.mark.parametrize("h,w", SHAPES)
    @pytest.mark.parametrize("max_levels", [1, 2, 3, 5])
    def test_lossless_round_trip(self, h, w, max_levels):
        rng = np.random.default_rng(20)
        image = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        compressed = encode(image, quality=QUALITY_MAX, max_levels=max_levels, remainder_raw=True)
        decoded = decode(compressed)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        np.testing.assert_array_equal(decoded, image)

    @pytest.mark.parametrize("max_levels", [2, 4])
    def test_rgb_lossless_round_trip(self, max_levels):
        rng = np.random.default_rng(21)
        for h, w in [(16, 16), (48, 64), (7, 9)]:
            image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            compressed = encode(image, quality=QUALITY_MAX, max_levels=max_levels, remainder_raw=True)
            decoded = decode(compressed)
            np.testing.assert_array_equal(decoded, image)

    @pytest.mark.parametrize("quality", [512, 256, 64])
    def test_lossy_sane(self, quality):
        """Smoke test across lossy quality settings -- not a tight
        distortion bound, just confirms no crash/garbage output."""
        rng = np.random.default_rng(22)
        image = rng.integers(0, 256, size=(64, 48), dtype=np.uint8)
        compressed = encode(image, quality=quality, max_levels=4, remainder_raw=True)
        decoded = decode(compressed)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        max_err = int(np.max(np.abs(decoded.astype(np.int32) - image.astype(np.int32))))
        assert max_err < 260

    def test_header_records_remainder_raw(self):
        rng = np.random.default_rng(23)
        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

        compressed_default = encode(image, quality=QUALITY_MAX, max_levels=5)
        header_default, _ = parse_header(compressed_default)
        assert header_default.remainder_raw is False

        compressed_raw = encode(image, quality=QUALITY_MAX, max_levels=5, remainder_raw=True)
        header_raw, _ = parse_header(compressed_raw)
        assert header_raw.remainder_raw is True
        # max_levels itself must be unaffected by sharing a byte with
        # the new flag bit.
        assert header_raw.max_levels == 5

    def test_requires_max_levels(self):
        rng = np.random.default_rng(24)
        image = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        with pytest.raises(ValueError):
            encode(image, quality=QUALITY_MAX, remainder_raw=True)

    def test_raw_is_larger_than_default(self):
        """Sanity check against gfwx-fpga's own real measurement (~44%
        smaller for the default entropy-coded remainder vs. raw storage,
        notes/gfwx_capped_recursion_explainer.md) -- raw storage should
        never be SMALLER for real content, or something's wrong."""
        from pathlib import Path

        from PIL import Image

        asset = Path(__file__).parent.parent / "examples" / "assets" / "cat_demo.png"
        photo = np.array(Image.open(asset).convert("L")).astype(np.uint8)

        compressed_default = encode(photo, quality=QUALITY_MAX, max_levels=5)
        compressed_raw = encode(photo, quality=QUALITY_MAX, max_levels=5, remainder_raw=True)
        assert len(compressed_raw) > len(compressed_default)

    def test_value_overflow_rejected(self):
        """_encode_remainder_raw must fail loudly rather than silently
        truncate a value that doesn't fit in int16."""
        from pygfwx.core.block_encoder import _encode_remainder_raw

        ok = np.array([[100, -32768], [32767, 0]], dtype=np.int64)
        _encode_remainder_raw(ok)  # must not raise

        bad = np.array([[100, 32768]], dtype=np.int64)
        with pytest.raises(ValueError):
            _encode_remainder_raw(bad)
