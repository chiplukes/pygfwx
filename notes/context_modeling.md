# GFWX Context Modeling

This document describes the adaptive context modeling used in GFWX for
entropy coding decisions.

## Overview

Context modeling predicts the statistical properties of wavelet coefficients
based on already-decoded neighbors. This allows the encoder/decoder to
adapt the Golomb-Rice parameters for optimal compression.

## Context Statistics

The context consists of two statistics:
- **sum**: First moment - weighted average of absolute neighbor values
- **sum2**: Second moment - weighted average of squared neighbor values

These are normalized to 16 counts for consistent comparison.

## Neighbor Sampling

For a coefficient at position (x, y) with wavelet level skip:

### Ancestor (weight 2)
Position calculated as:
```
px = x0 + (x & ~(skip*2)) + (x & skip)
py = y0 + (y & ~(skip*2)) + (y & skip)
```
If px >= x1, subtract skip*2 (wrap around).

### Siblings (weight 2 each)
Only available when `(y & skip)` is non-zero:
- Upper sibling: `[y0 + y - skip][x0 + (x | skip)]`
- Left sibling: `[y0 + y][x0 + x - skip]` (only if `x & skip`)

### Neighbors at distance 2*skip (weights: 4, 4, 2, 2)
Available when `y >= skip*2` and `x >= skip*2`:
- North (w=4): `[y0 + y - skip*2][x0 + x]`
- West (w=4): `[y0 + y][x0 + x - skip*2]`
- Northwest (w=2): `[y0 + y - skip*2][x0 + x - skip*2]`
- Northeast (w=2): `[y0 + y - skip*2][x0 + x + skip*2]` (if available)

### Neighbors at distance 4*skip (weights: 2, 2, 1, 1)
Available when `y >= skip*4` and `x >= skip*4`:
- North-far (w=2), West-far (w=2), Northwest-far (w=1), Northeast-far (w=1)

## Value Clamping

When adding to sum2, values are clamped to 4096 to prevent overflow:
```
sum2 += min(|x|, 4096)^2 * weight
```

## Coding Mode Selection

The decision tree uses `sum_sq = sum * sum` compared against `sum2`:

| Condition | Mode | Pot |
|-----------|------|-----|
| sum_sq < 2*sum2 + threshold | interleaved | 0 |
| sum_sq < 2*sum2 + 950 | interleaved | 1 |
| sum_sq < 3*sum2 + 3000 && sum_sq < 5*sum2 + 400 | signed | 1 |
| sum_sq < 3*sum2 + 3000 | interleaved | 2 |
| sum_sq < 3*sum2 + 12000 && sum_sq < 5*sum2 + 3000 | signed | 2 |
| sum_sq < 3*sum2 + 12000 | interleaved | 3 |
| sum_sq < 4*sum2 + 44000 && sum_sq < 6*sum2 + 12000 | signed | 3 |
| sum_sq < 4*sum2 + 44000 | interleaved | 4 |
| otherwise | signed | 4 |

The `threshold` is 100 for luma, 250 for chroma.

## FAST Encoder Context Update

In FAST mode, context decays exponentially:
```
new_sum = (sum * 15 + 7) >> 4 + |value|
new_sum2 = (sum2 * 15 + 7) >> 4 + min(|value|, 4096)^2
```

## Run-Length Coder Parameter

The run coder pot (0-4) is selected based on context:
- Higher pot values for low-activity regions (many zeros)
- Lower pot values for high-activity regions

### FAST Mode Thresholds
| sum | pot |
|-----|-----|
| < 1 | 4 |
| < 2 | 3 |
| < 4 | 2 |
| < 8 | 1 |
| >= 8 | 0 |

### CONTEXTUAL Mode (lossy)
Uses both sum and sum2 with more complex thresholds.

### Lossless Mode (quality=1024)
Simple: pot = 1 if sum < 2, else 0.
