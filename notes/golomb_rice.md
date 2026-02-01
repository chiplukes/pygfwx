# Golomb-Rice Entropy Coding

This document explains the Golomb-Rice coding used in GFWX.

## Overview

Golomb-Rice coding is a variable-length entropy coding method optimized for geometric distributions (common in wavelet coefficients). It's simpler and faster than arithmetic coding while still providing good compression.

## Basic Concept

For a non-negative integer `x` and parameter `k` (power of two):

```
x = quotient * 2^k + remainder

where:
    quotient  = x >> k       (x divided by 2^k)
    remainder = x & (2^k-1)  (x modulo 2^k)
```

The code consists of:
1. **Unary part**: `quotient` ones followed by a zero (or zeros followed by a one)
2. **Binary part**: `k` bits for the remainder

## Example

Encode x = 13 with k = 2:
```
quotient  = 13 >> 2 = 3
remainder = 13 & 3  = 1

Unary part (3 zeros + 1 one): 0001
Binary part (2 bits):          01

Full code: 000101 (6 bits)
```

## GFWX Implementation Details

### Limited-Length Codes

GFWX uses a maximum unary length to bound worst-case code sizes:

```python
MAX_UNARY = 12  # Maximum quotient before escape

def unsigned_code(bits, x, pot):
    """Encode unsigned integer with power-of-two parameter."""
    quotient = x >> pot
    
    if quotient >= MAX_UNARY:
        # Escape: write MAX_UNARY zeros, then recurse with larger pot
        bits.put_bits(0, MAX_UNARY)
        unsigned_code(bits, x - (MAX_UNARY << pot), pot + 4)
    else:
        # Normal: unary quotient + binary remainder
        bits.put_bits(1, quotient + 1)  # quotient zeros + one
        bits.put_bits(x & ((1 << pot) - 1), pot)  # remainder

def unsigned_decode(bits, pot):
    """Decode unsigned integer."""
    zeros = bits.get_zeros(MAX_UNARY)
    
    if zeros >= MAX_UNARY:
        # Escape: recurse with larger pot
        return (MAX_UNARY << pot) + unsigned_decode(bits, pot + 4)
    else:
        # Normal: read remainder bits
        remainder = bits.get_bits(pot)
        return (zeros << pot) + remainder
```

### Signed Values

GFWX uses two methods for signed values:

#### Interleaved Coding
Maps signed to unsigned by interleaving positive and negative:
```
 0 → 0
+1 → 1
-1 → 2
+2 → 3
-2 → 4
...

signed_to_unsigned(x) = 2*|x| - (1 if x > 0 else 0)
unsigned_to_signed(u) = (u+1)//2 * (1 if u&1 else -1) if u > 0 else 0
```

#### Sign-Bit Coding
Encode magnitude, then sign bit (if non-zero):
```python
def signed_code(bits, x, pot):
    """Encode signed integer with trailing sign bit."""
    unsigned_code(bits, abs(x), pot)
    if x != 0:
        bits.put_bits(1 if x < 0 else 0, 1)
```

## Context-Adaptive Parameter Selection

GFWX selects the Golomb parameter `pot` based on local context (neighboring coefficients):

```python
def select_pot(context_sum, context_sum2, context_count):
    """Select optimal pot parameter from context statistics."""
    if context_count == 0:
        return 4  # Default
    
    # Estimate variance from context
    mean = context_sum / context_count
    variance = context_sum2 / context_count - mean * mean
    
    # Higher variance → larger pot
    # pot ≈ log2(sqrt(variance)) = log2(variance) / 2
    return optimal_pot_for_variance(variance)
```

## Zero-Run Encoding

Sparse wavelet coefficients have many zeros. GFWX encodes runs of zeros efficiently:

```python
def encode_with_runs(bits, coefficients, pot):
    """Encode coefficients with zero-run optimization."""
    i = 0
    while i < len(coefficients):
        if coefficients[i] == 0:
            # Count consecutive zeros
            run_length = count_zeros(coefficients, i)
            encode_zero_run(bits, run_length, pot)
            i += run_length
        else:
            # Encode non-zero coefficient
            signed_code(bits, coefficients[i], pot)
            i += 1
```

## Comparison with Other Codes

| Method | Complexity | Compression | Speed |
|--------|------------|-------------|-------|
| Golomb-Rice | O(1) per symbol | Good | Very fast |
| Huffman | O(log n) | Good | Fast |
| Arithmetic | O(1) amortized | Best | Slow |

Golomb-Rice is ideal for:
- Hardware implementation (simple bit operations)
- Real-time encoding
- Coefficients with geometric distribution

## Implementation Notes

### Bit Stream Interface

```python
class Bits:
    """Bit-level I/O for Golomb-Rice coding."""
    
    def put_bits(self, value, count):
        """Write `count` bits from `value`."""
        pass
    
    def get_bits(self, count):
        """Read `count` bits and return value."""
        pass
    
    def get_zeros(self, max_count):
        """Count consecutive zero bits (up to max_count)."""
        pass
```

### Efficiency Tips

1. Process bits in word-sized chunks when possible
2. Use lookup tables for small pot values
3. Maintain bit buffer to minimize memory accesses

## References

- Rice, R. F. (1979). "Some Practical Universal Noiseless Coding Techniques"
- Golomb, S. W. (1966). "Run-Length Encodings"
- Malvar, H. S. (2006). "Adaptive Run-Length / Golomb-Rice Encoding"
