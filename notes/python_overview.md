# PyGFWX Python Project Overview

This document lists all Python files in the project with their purposes.
**Keep this updated as files are added or modified.**

## Main Package (`src/pygfwx/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Package exports and version | Placeholder |
| `cli.py` | Command-line interface | Stub (not yet implemented) |

### Core Module (`src/pygfwx/core/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Core module exports | Placeholder |
| `bitstream.py` | Bit-level I/O (BitReader, BitWriter classes) | Complete |
| `header.py` | GFWX header parsing/writing | Complete |
| `golomb_rice.py` | Golomb-Rice entropy coding (unsigned, interleaved, signed) | Complete |
| `context.py` | Adaptive context modeling for entropy coding | Complete |
| `decoder.py` | Coefficient decoding with run-length and context | Complete |
| `lifting.py` | Wavelet lifting transforms (5/3 LINEAR, 9/7 CUBIC) | Complete |
| `block_decoder.py` | High-level decode pipeline, block/level processing, Bayer | Complete |
| `block_encoder.py` | High-level encode pipeline, block/level processing, Bayer | Complete |
| `quantization.py` | Scalar quantization/dequantization | Complete |
| `encoder.py` | Coefficient encoding (counterpart to decoder.py) | Complete |
| `transforms.py` | Color/channel transforms (UYV, A710), generic forward/inverse, program building/parsing/validation | Complete |
| `multi_layer.py` | Multi-layer image utilities (stereo, depth maps) | Complete |
| `bayer.py` | Bayer/CFA pattern utilities for RAW camera data | Complete |
| `metadata.py` | Metadata read/write utilities (text, JSON, key-value, chunks) | Complete |
| `codec.py` | High-level encode/decode API | Complete |

### Streaming Module (`src/pygfwx/streaming/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Streaming module exports | Complete |
| `progressive.py` | Progressive decode with downsampling, truncation handling | Complete |

### Debug Module (`src/pygfwx/debug/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Debug module exports | Placeholder |
| `hexdump.py` | Hex dump utilities | Not started |
| `visualize.py` | Wavelet visualization | Not started |

### Utils Module (`src/pygfwx/utils/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Utils module exports | Complete |
| `reference_images.py` | Centralized reference image generator for all testing | Complete |
| `image_io.py` | Image loading/saving helpers | Not started |

## Cross-Codec Module (`cross_codec/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Cross-codec module exports | Created |
| `gfwx_sdk.py` | Python wrapper for GFWX SDK (ctypes) | Complete |
| `sdk_data_loader.py` | Load SDK debug dumps | Not started |

### Reference (`cross_codec/reference/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Reference implementations exports | Placeholder |
| `lifting_reference.py` | SDK lifting algorithm reference | Not started |
| `golomb_reference.py` | SDK Golomb-Rice reference | Not started |

### Validation (`cross_codec/validation/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Validation module exports | Placeholder |
| `validate_lifting.py` | Wavelet transform validation | Not started |
| `validate_quantize.py` | Quantization validation | Not started |
| `validate_encode.py` | Encoding validation | Not started |
| `validate_full.py` | Full pipeline validation | Not started |

### Compare (`cross_codec/compare/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Compare module exports | Placeholder |
| `compare_coefficients.py` | Coefficient comparison | Not started |
| `compare_bitstream.py` | Bitstream comparison | Not started |
| `side_by_side.py` | Visual comparison | Not started |

### Debug (`cross_codec/debug/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Debug module exports | Placeholder |
| `staged_encoder_match.py` | Step-by-step encoder debug | Not started |
| `staged_decoder_match.py` | Step-by-step decoder debug | Not started |

## Tests (`tests/`)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Test package marker | Created |
| `conftest.py` | Pytest fixtures (SDK detection, test images) | Complete |
| `test_utils.py` | Shared test utilities | Complete |
| `test_infrastructure.py` | Infrastructure and reference image tests | Complete |
| `test_sdk_wrapper.py` | SDK wrapper tests (roundtrip, filters, headers) | Complete |
| `test_bitstream.py` | Bit I/O tests | Complete |
| `test_header.py` | Header parsing tests | Complete |
| `test_golomb_rice.py` | Entropy coding tests | Complete |
| `test_context.py` | Context modeling tests | Complete |
| `test_decoder.py` | Coefficient decoding tests | Complete |
| `test_lifting.py` | Wavelet transform tests (LINEAR, CUBIC, roundtrip) | Complete |
| `test_block_decoder.py` | Block decoder and full pipeline tests | Complete |
| `test_quantization.py` | Quantization/dequantization tests | Complete |
| `test_transforms.py` | Color transform and program I/O tests | Complete |
| `test_encoder.py` | Coefficient encoding roundtrip tests | Complete |
| `test_progressive.py` | Progressive decode tests | Complete |
| `test_multi_layer.py` | Multi-layer image tests | Complete |
| `test_bayer.py` | Bayer/CFA pattern tests | Complete |
| `test_metadata.py` | Metadata read/write tests | Complete |
| `test_roundtrip.py` | Encode/decode roundtrip tests | Complete |
| `test_formats.py` | Multi-format tests | Not started |
| `test_sdk_comparison.py` | SDK vs PyGFWX tests | Not started |

## Examples (`examples/`)

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Examples documentation | Created |
| `basic_usage.py` | Simple encode/decode at different quality levels | Complete |
| `cat_demo.py` | Compression metrics from test image | Complete |
| `compression_walkthrough.py` | Step-by-step compression pipeline walkthrough with visualizations | Complete |
| `progressive_demo.py` | Progressive decoding with truncation and downsampling | Complete |

## Diagnostics (`diagnostics/`)

Debugging tools created during SDK compatibility investigation.

| File | Purpose | Status |
|------|---------|--------|
| `analyze_block_sizes.py` | Parse and compare encoded block structure between Python and SDK | Complete |
| `debug_encoder.py` | Compare Python vs SDK encoder outputs at various quality levels | Complete |
| `trace_bits.py` | Bit-level tracing of coefficient encoding | Complete |
| `trace_step2.py` | Trace encoding at step=2 level to debug context/run-length | Complete |
