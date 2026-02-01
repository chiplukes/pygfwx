# Wavelet Lifting Scheme

This document explains the lifting scheme used in GFWX for wavelet transforms.

## Overview

The lifting scheme (Sweldens, 1995) is an efficient method for implementing wavelet transforms in-place, without requiring auxiliary memory buffers.

## Basic Concept

Traditional wavelet transforms use convolution with filter banks. Lifting decomposes this into simpler steps:

1. **Split**: Separate samples into even and odd indices
2. **Predict**: Predict odd samples from even (creates detail coefficients)
3. **Update**: Update even samples from predicted odd (creates approximation coefficients)

## Forward Transform (Analysis)

```
Original:  [e0, o0, e1, o1, e2, o2, ...]
           (e = even indices, o = odd indices)

Step 1 - Predict:
    odd[i] = odd[i] - predict(even[i-1], even[i], even[i+1], ...)

Step 2 - Update:
    even[i] = even[i] + update(odd[i-1], odd[i], odd[i+1], ...)

Result:   [approx0, detail0, approx1, detail1, ...]
```

## Inverse Transform (Synthesis)

The inverse is simply the reverse operations:

```
Step 1 - Undo Update:
    even[i] = even[i] - update(odd[i-1], odd[i], odd[i+1], ...)

Step 2 - Undo Predict:
    odd[i] = odd[i] + predict(even[i-1], even[i], even[i+1], ...)
```

## GFWX Filter Types

### FilterLinear (5/3 Wavelet)

The 5/3 wavelet uses linear interpolation:

```python
# Forward (predict step)
odd[i] -= (even[i] + even[i+1]) // 2

# Forward (update step)
even[i] += (odd[i-1] + odd[i]) // 4

# Inverse (undo update)
even[i] -= (odd[i-1] + odd[i]) // 4

# Inverse (undo predict)
odd[i] += (even[i] + even[i+1]) // 2
```

**Properties:**
- Simple integer arithmetic
- Exactly invertible (no rounding errors accumulate)
- Good for lossless compression
- Same as JPEG 2000 reversible 5/3 transform

### FilterCubic (9/7 Wavelet)

The 9/7 wavelet uses cubic interpolation with median clamping:

```python
def cubic(c0, c1, c2, c3):
    """Cubic interpolation with coefficients."""
    # Standard cubic: -c0/16 + 9*c1/16 + 9*c2/16 - c3/16
    # GFWX uses median clamping to [c1, c2] for robustness
    result = (-c0 + 9*c1 + 9*c2 - c3) // 16
    return clamp(result, min(c1, c2), max(c1, c2))

# Forward (predict step)
odd[i] -= cubic(even[i-1], even[i], even[i+1], even[i+2])

# Forward (update step)
even[i] += cubic(odd[i-1], odd[i], odd[i+1], odd[i+2]) // 2
```

**Properties:**
- Better frequency separation
- Preferred for lossy compression
- Median clamping prevents ringing artifacts
- Not exactly invertible (small rounding differences)

## 2D Transform

For 2D images, lifting is applied separably:

### Forward Transform Order
1. Horizontal predict (all rows)
2. Horizontal update (all rows)
3. Vertical predict (all columns)
4. Vertical update (all columns)

### Inverse Transform Order (reversed)
1. Vertical undo-update (all columns)
2. Vertical undo-predict (all columns)
3. Horizontal undo-update (all rows)
4. Horizontal undo-predict (all rows)

## Multi-Level Decomposition

For multi-level transforms, apply the 2D transform recursively to the approximation (LL) band:

```
Level 0: Full image → LL0, HL0, LH0, HH0
Level 1: LL0 → LL1, HL1, LH1, HH1
Level 2: LL1 → LL2, HL2, LH2, HH2
...
```

## Boundary Handling

At image boundaries, GFWX uses symmetric extension (reflection):

```python
def get_sample(image, x, size):
    """Get sample with boundary reflection."""
    if x < 0:
        x = -x
    elif x >= size:
        x = 2 * size - x - 2
    return image[x]
```

## In-Place Memory Layout

After lifting, the data is interleaved in memory:

```
Before:  [p0, p1, p2, p3, p4, p5, p6, p7]

After:   [L0, H0, L1, H1, L2, H2, L3, H3]
         (L = approximation, H = detail)
```

This allows the transform to work without allocating extra memory.

## 2D Interleaved Layout vs Quadrant Layout

GFWX stores wavelet coefficients **interleaved in-place**, which is different from the traditional **quadrant layout** used by codecs like CineForm for visualization.

### Traditional Quadrant Layout (CineForm-style)

Coefficients are stored in contiguous blocks:
```
+-------+-------+
|  LL   |  HL   |  ← Contiguous regions
+-------+-------+
|  LH   |  HH   |
+-------+-------+
```

### GFWX Interleaved Layout (In-Place)

Coefficients are scattered based on even/odd positions:
```
For a 4x4 image after 2D lifting:
         col 0   col 1   col 2   col 3
row 0:  [LL(0,0) HL(0,0) LL(0,1) HL(0,1)]
row 1:  [LH(0,0) HH(0,0) LH(0,1) HH(0,1)]
row 2:  [LL(1,0) HL(1,0) LL(1,1) HL(1,1)]
row 3:  [LH(1,0) HH(1,0) LH(1,1) HH(1,1)]
```

The bands are interleaved at every pixel position:
- **LL**: even row, even col → `data[0::2, 0::2]`
- **HL**: even row, odd col  → `data[0::2, 1::2]`
- **LH**: odd row, even col  → `data[1::2, 0::2]`
- **HH**: odd row, odd col   → `data[1::2, 1::2]`

### Multi-Level Interleaved Layout

After multiple levels, the pattern continues with larger strides:

```
After 3 levels on 64x64 image:
- Level 1: stride 2  → LL1 at [0::2, 0::2], HL1 at [0::2, 1::2], etc.
- Level 2: stride 4  → LL2 at [0::4, 0::4], HL2 at [0::4, 2::4], etc.
- Level 3: stride 8  → LL3 at [0::8, 0::8], HL3 at [0::8, 4::8], etc.
```

### Extracting Bands for Visualization

To visualize bands, extract them using numpy slicing:

```python
# After level 1 transform (step=1):
ll1 = data[0::2, 0::2]  # 32x32 approximation
hl1 = data[0::2, 1::2]  # 32x32 horizontal detail
lh1 = data[1::2, 0::2]  # 32x32 vertical detail
hh1 = data[1::2, 1::2]  # 32x32 diagonal detail

# After level 2 transform (step=2):
ll2 = data[0::4, 0::4]  # 16x16 approximation
hl2 = data[0::4, 2::4]  # 16x16 horizontal detail
lh2 = data[2::4, 0::4]  # 16x16 vertical detail
hh2 = data[2::4, 2::4]  # 16x16 diagonal detail
```

### Important: Extract LL Before Next Level

When visualizing multi-level decomposition, extract LL bands **before** applying the next level's transform. Otherwise, the LL positions will contain further-decomposed data (LL2, HL2, LH2, HH2 mixed together), creating a checkerboard appearance.

```python
# Correct approach for visualization:
_lift_horizontal(data, 0, 0, w, h, 1, filter)
_lift_vertical(data, 0, 0, w, h, 1, filter)
ll1 = data[0::2, 0::2].copy()  # Extract BEFORE level 2

_lift_horizontal(data, 0, 0, w, h, 2, filter)
_lift_vertical(data, 0, 0, w, h, 2, filter)
ll2 = data[0::4, 0::4].copy()  # Extract BEFORE level 3
```

### Converting to Quadrant Layout

To display in the traditional quadrant layout (for educational purposes):

```python
def interleaved_to_quadrant(data):
    """Convert GFWX interleaved layout to quadrant layout."""
    h, w = data.shape
    ll = data[0::2, 0::2]
    hl = data[0::2, 1::2]
    lh = data[1::2, 0::2]
    hh = data[1::2, 1::2]
    
    result = np.zeros_like(data)
    result[:h//2, :w//2] = ll
    result[:h//2, w//2:] = hl
    result[h//2:, :w//2] = lh
    result[h//2:, w//2:] = hh
    return result
```

## Integer Overflow Considerations

When implementing in Python, be aware:
- C++ code may overflow at 16-bit or 32-bit boundaries
- Python integers have arbitrary precision
- May need explicit masking to match SDK behavior exactly

```python
def to_int16(x):
    """Simulate C int16_t overflow."""
    x = x & 0xFFFF
    return x - 0x10000 if x >= 0x8000 else x
```

## Division Behavior (Critical for SDK Compatibility)

C++ integer division truncates toward zero, but Python's `//` operator floors toward negative infinity. This difference affects lifting calculations when sums are negative.

```python
# Example:
# C++: (-5) / 2 = -2  (truncate toward zero)
# Python: (-5) // 2 = -3  (floor toward negative infinity)

# To match C++ behavior, use:
def trunc_div(num, denom):
    """C-style integer division (truncate toward zero)."""
    return int(num / denom)

# In lifting:
# WRONG (Python floor division):
# odd[i] -= (left + right) // 2

# CORRECT (matches C++):
# odd[i] -= trunc_div(left + right, 2)
```

This applies to all `/ 2` and `/ 4` operations in the lifting scheme where the numerator can be negative (which is common with wavelet coefficients).

## References

- Sweldens, W. (1995). "The Lifting Scheme: A New Philosophy in Biorthogonal Wavelet Constructions"
- Daubechies, I., Sweldens, W. (1998). "Factoring Wavelet Transforms into Lifting Steps"
- JPEG 2000 Part 1 (ISO/IEC 15444-1) - Uses same 5/3 and 9/7 filters
