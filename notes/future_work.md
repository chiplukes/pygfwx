# PyGFWX Future Work

Items deferred from the initial architecture review. Each represents a
non-trivial development effort.

## Implement CLI Commands

**File:** `src/pygfwx/cli.py`

All three commands (`compress`, `decompress`, `info`) are currently stubs that
print "Not yet implemented" and exit with code 1.

- `compress <input> <output.gfwx>` — load image with Pillow, call `encode()`, write file
- `decompress <input.gfwx> <output>` — read file, call `decode()`, save image with Pillow
- `info <input.gfwx>` — parse header with `get_header()`, pretty-print all fields

Depends on `add-image-io` (see below) for image loading/saving helpers.

## Implement `utils/image_io.py`

**File:** `src/pygfwx/utils/image_io.py`

A thin Pillow wrapper for loading and saving images as NumPy arrays. Referenced
in project notes but not yet created.

- `load_image(path) -> np.ndarray` — supports PNG, JPEG, TIFF; handles 8/16-bit
- `save_image(image: np.ndarray, path)` — infers format from extension

This also unblocks CLI implementation and simplifies example scripts.

## Wire Color Transforms into Encode Pipeline

**Files:** `src/pygfwx/core/block_encoder.py`, `src/pygfwx/core/block_decoder.py`

`transforms.py` has complete forward/inverse implementations of UYV and A710
color transforms, but `block_encoder.py` always writes an identity transform
end-marker and `is_chroma` is always all-zeros. The decoder already has an
`_apply_inverse_transform` path that will activate once a real program is
encoded.

Steps required:
1. Add a `color_transform` parameter to `encode()` / `encode_image()` (e.g. `"uyv"`, `"a710"`, `None`)
2. Apply `forward_transform_uyv()` or `forward_transform_a710()` before `lift()`
3. Write the real transform program to the bitstream instead of the identity end-marker
4. Propagate the `is_chroma` list from the transform result into quantization

## Implement Debug Module

**Files:** `src/pygfwx/debug/hexdump.py`, `src/pygfwx/debug/visualize.py`

The `debug/` package exists but contains only an empty `__init__.py`.

- `hexdump.py` — annotated hex dump of a `.gfwx` file, labelling each field
  (magic, header fields, transform program, block sizes, block data)
- `visualize.py` — matplotlib visualizations of wavelet coefficient subbands
  at each resolution level, useful for understanding the transform output
