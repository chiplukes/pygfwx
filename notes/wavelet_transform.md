# GFWX Wavelet Transform Details

This document provides comprehensive documentation of the wavelet transform
used in GFWX, including mathematical foundations, implementation details,
and practical considerations.

## Introduction to Wavelets

### What is a Wavelet Transform?

A wavelet transform decomposes a signal (or image) into components at different
scales and positions. Unlike the Fourier transform which uses infinite sinusoids,
wavelets use short, localized wave-like functions.

**Key properties:**
- **Multi-resolution**: Analyze at different scales simultaneously
- **Localization**: Preserve both frequency AND position information
- **Sparse representation**: Natural images compress well (most coefficients near zero)

### Why Use Wavelets for Compression?

1. **Energy compaction**: Most image energy concentrates in few coefficients
2. **Progressive decoding**: Low-resolution preview from partial data
3. **No block artifacts**: Unlike DCT-based codecs (JPEG)
4. **Scalable quality**: Smooth degradation from lossless to lossy

## The Lifting Scheme

GFWX uses the lifting scheme (Sweldens, 1995) rather than traditional
convolution-based wavelet transforms.

### Traditional vs. Lifting

**Traditional DWT:**
```
input → [Lowpass Filter] → ↓2 → approximation (L)
      → [Highpass Filter] → ↓2 → detail (H)
```
Requires convolution and downsampling with auxiliary buffers.

**Lifting Scheme:**
```
input → split even/odd → predict → update → [L, H] interleaved
```
Works in-place with no extra memory.

### Lifting Steps

#### Step 1: Split
Separate samples into even and odd positions:
```
Input:  [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
Split:   E = [x₀, x₂, x₄, x₆]  (even indices)
         O = [x₁, x₃, x₅, x₇]  (odd indices)
```

#### Step 2: Predict (creates detail coefficients)
Predict odd samples from neighboring even samples:
```
H[i] = O[i] - Predict(E[i], E[i+1], ...)
```
The prediction residual becomes the high-frequency (detail) coefficient.

#### Step 3: Update (creates approximation coefficients)
Update even samples using the predicted values:
```
L[i] = E[i] + Update(H[i-1], H[i], ...)
```
This creates the low-frequency (approximation) coefficient.

### Why Lifting Works

- Prediction removes correlation → detail coefficients are small
- Update preserves image mean/energy at coarse scales
- Perfect reconstruction: just reverse the operations

## GFWX Filter Types

### FilterLinear (5/3 Wavelet)

The 5/3 filter uses linear interpolation with 5 taps for prediction
and 3 taps for update.

**Forward Transform:**
```python
# Predict: odd -= average of neighbors
for x in range(step, sizex, step*2):
    image[x] -= (image[x-step] + image[x+step]) // 2

# Update: even += quarter of neighbors
for x in range(step*2, sizex, step*2):
    image[x] += (image[x-step] + image[x+step]) // 4
```

**Inverse Transform:**
```python
# Undo Update
for x in range(step*2, sizex, step*2):
    image[x] -= (image[x-step] + image[x+step]) // 4

# Undo Predict
for x in range(step, sizex, step*2):
    image[x] += (image[x-step] + image[x+step]) // 2
```

**Properties:**
- Integer arithmetic only
- Perfectly invertible (no rounding drift)
- Same as JPEG 2000 reversible transform
- Best for lossless compression

**Filter coefficients:**
- Analysis lowpass: [-1/8, 1/4, 3/4, 1/4, -1/8]
- Analysis highpass: [-1/2, 1, -1/2]

### FilterCubic (9/7 Wavelet)

The 9/7 filter uses cubic interpolation with median clamping for robustness.

**Cubic Interpolation:**
```python
def cubic(c0, c1, c2, c3):
    """
    Cubic interpolation with median clamping.

    Standard cubic: (-c0 + 9*c1 + 9*c2 - c3) / 16
    Clamping prevents overshoots at edges.
    """
    result = round_fraction(-c0 + 9*(c1 + c2) - c3, 16)
    return median(result, c1, c2)

def round_fraction(num, denom):
    """Round towards nearest (not towards zero)."""
    if num < 0:
        return (num - denom // 2) // denom
    else:
        return (num + denom // 2) // denom
```

**Forward Transform:**
```python
# Predict: odd -= cubic interpolation
for x in range(step, sizex, step*2):
    c0, c1, c2, c3 = get_neighbors(image, x, step)
    image[x] -= cubic(c0, c1, c2, c3)

# Update: even += half of cubic
for x in range(step*2, sizex, step*2):
    g0, g1, g2, g3 = get_neighbors(image, x, step)
    image[x] += cubic(g0, g1, g2, g3) // 2
```

**Properties:**
- Better frequency separation than 5/3
- Preferred for lossy compression
- Median clamping prevents ringing artifacts at edges
- NOT perfectly invertible (small rounding differences)

**Why median clamping?**
Standard cubic interpolation can overshoot, creating values outside
the range of input samples. This causes ringing artifacts near edges.
Clamping to the median of the center values prevents this.

## 2D Transform

Images are 2D, so we apply the 1D transform in both directions.

### Forward Transform Sequence

```
Original Image
      │
      ▼
┌─────────────────────────┐
│ Horizontal Predict      │  odd columns -= f(even columns)
│ (all rows)              │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Horizontal Update       │  even columns += f(odd columns)
│ (all rows)              │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Vertical Predict        │  odd rows -= f(even rows)
│ (all columns)           │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Vertical Update         │  even rows += f(odd rows)
│ (all columns)           │
└─────────────────────────┘
      │
      ▼
Transformed Image (interleaved)
```

### Inverse Transform Sequence (reversed)

```
1. Vertical Undo-Update
2. Vertical Undo-Predict
3. Horizontal Undo-Update
4. Horizontal Undo-Predict
```

### Result Layout (Single Level)

After one level of 2D transform:

```
┌───────┬───────┐
│  LL   │  HL   │
│(approx│(horiz │
│  low) │detail)│
├───────┼───────┤
│  LH   │  HH   │
│(vert  │(diag  │
│detail)│detail)│
└───────┴───────┘
```

- **LL**: Low-low (approximation) - looks like smaller version of image
- **HL**: High-low (horizontal details) - vertical edges
- **LH**: Low-high (vertical details) - horizontal edges
- **HH**: High-high (diagonal details) - diagonal edges/texture

## Multi-Level Decomposition

GFWX applies multiple levels of wavelet transform for better compression.

### Pyramid Structure

```
Level 0:  [Full Image] → LL₀, HL₀, LH₀, HH₀
Level 1:  [LL₀]        → LL₁, HL₁, LH₁, HH₁
Level 2:  [LL₁]        → LL₂, HL₂, LH₂, HH₂
...and so on
```

Each level processes only the approximation band from the previous level.

### Visual Representation

```
┌─────┬─────┬───────────┐
│LL₂  │HL₂  │           │
├─────┼─────┤   HL₁     │
│LH₂  │HH₂  │           │
├─────┴─────┼───────────┤
│           │           │
│   LH₁     │   HH₁     │
│           │           │
├───────────┴───────────┤
│                       │
│         HL₀           │
│                       │
├───────────────────────┤
│                       │
│         LH₀           │
│                       │
├───────────────────────┤
│                       │
│         HH₀           │
│                       │
└───────────────────────┘
```

### Step Parameter

GFWX uses a `step` parameter to control the wavelet level:
- `step = 1`: Process full resolution
- `step = 2`: Process half resolution (skip every other sample)
- `step = 4`: Process quarter resolution
- etc.

The transform runs while `step < sizex || step < sizey`, effectively
applying as many levels as the image dimensions allow.

## Boundary Handling

At image edges, we need samples that don't exist.

### GFWX Approach

GFWX uses symmetric extension (reflection) at boundaries:

```python
def get_sample(image, x, size):
    """Get sample with symmetric boundary extension."""
    if x < 0:
        x = -x  # Reflect at left edge
    elif x >= size:
        x = 2 * size - x - 2  # Reflect at right edge
    return image[x]
```

**Example:**
```
Image:     [A, B, C, D, E]
Extended:  [B, A, B, C, D, E, D, C]
            ↑        actual        ↑
         reflected              reflected
```

### SDK Implementation

In the SDK, boundary handling is integrated into the loops:

```cpp
// For predict: if at last odd position, use only left neighbor
if (x < sizex - step)
    base[x] -= (base1[x] + base2[x]) / 2;
else
    base[x] -= base1[x];  // At boundary

// For update: if at last even position, use half
if (x < sizex - step)
    base[x] += (base1[x] + base2[x]) / 4;
else
    base[x] += base1[x] / 2;  // At boundary
```

## In-Place Memory Layout

One key advantage of lifting is that it works in-place.

### Before Transform
```
[p₀, p₁, p₂, p₃, p₄, p₅, p₆, p₇]
```

### After Horizontal Predict
```
[p₀, h₀, p₂, h₁, p₄, h₂, p₆, h₃]
      ↑        ↑        ↑        ↑
    detail   detail   detail   detail
```

### After Horizontal Update
```
[L₀, H₀, L₁, H₁, L₂, H₂, L₃, H₃]
  ↑   ↑    ↑   ↑    ↑   ↑    ↑   ↑
 approx detail (interleaved)
```

The data stays in the same array, just reinterpreted.

## Implementation Considerations

### Integer Overflow

The SDK uses integer arithmetic. Be aware of overflow:

```cpp
// Cubic multiply can produce large intermediate values
// -c0 + 9*(c1 + c2) - c3 with 16-bit inputs
// Worst case: -32768 + 9*(32767 + 32767) - (-32768)
//           = -32768 + 589,806 + 32768 = 589,806
// Fits in 32-bit, but careful with 16-bit intermediates
```

**Python consideration:**
Python has arbitrary precision integers, so overflow isn't an issue.
But to match SDK behavior exactly, you may need explicit masking.

### Boost Factor

For lossy compression, GFWX applies a "boost" factor of 8 before
wavelet transform and removes it after. This improves precision:

```python
# Encoder (lossy)
image *= 8
lift(image)
quantize(image)
encode(image)

# Decoder (lossy)
decode(image)
dequantize(image)
unlift(image)
image //= 8
```

For lossless (quality=1024), boost is 1 (no scaling).

### Why boost = 8?

The cubic filter has a maximum multiplier of ~20 (from 9+9 in the formula).
With boost=8, the maximum factor is 8*20=160, which safely fits in the
range below 256. This keeps intermediate values reasonable for 16-bit data.

## Comparison: 5/3 vs 9/7

| Aspect | 5/3 (Linear) | 9/7 (Cubic) |
|--------|--------------|-------------|
| Taps | 5 analyze, 3 synthesize | 9 analyze, 7 synthesize |
| Math | Integer only | Integer with rounding |
| Invertible | Perfect | Not quite (rounding) |
| Use case | Lossless | Lossy |
| Edge handling | Simple reflection | Median clamping |
| Quality | Good | Better (smoother) |

## References

1. Sweldens, W. (1995). "The Lifting Scheme: A New Philosophy in Biorthogonal Wavelet Constructions"
2. Daubechies, I., Sweldens, W. (1998). "Factoring Wavelet Transforms into Lifting Steps"
3. ISO/IEC 15444-1 (JPEG 2000) - Uses the same 5/3 and CDF 9/7 wavelets
4. Calderbank, A.R., et al. (1998). "Wavelet Transforms That Map Integers to Integers"
