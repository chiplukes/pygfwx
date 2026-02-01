"""Trace encoding at step=2 to find the difference."""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.pygfwx.core.bitstream import BitWriter
from src.pygfwx.core.context import Context, get_context
from src.pygfwx.core.header import Encoder


def trace_encode_step2(image, stream, x0, y0, x1, y1, step, scheme, quality, max_trace=20):
    """Trace coefficient encoding at step=2."""
    sizex = x1 - x0
    sizey = y1 - y0

    context = Context(sum=0, sum2=0)
    run = 0
    run_coder = 0

    trace_count = 0

    for y in range(0, sizey, step):
        x_step = step if (y & step) else step * 2

        for x in range(x_step - step, sizex, x_step):
            s = int(image[y0 + y, x0 + x])

            if trace_count < max_trace:
                # Get context for display
                ctx = get_context(image, x0, y0, x1, y1, x, y, step) if scheme == Encoder.CONTEXTUAL else context

                print(f"  [{y:2},{x:2}]: s={s:5}, ctx.sum={ctx.sum:5}, ctx.sum2={ctx.sum2:6}, run_coder={run_coder}")

                # Check run logic
                if run_coder and s == 0:
                    print(f"         -> continue run (run={run + 1})")
                elif run_coder:
                    print(f"         -> break run (had run={run}), encode run then s")
                else:
                    print(f"         -> encode s directly")

                trace_count += 1

            # Actually process (simplified - just track run)
            if run_coder and s == 0:
                run += 1
            else:
                if run_coder:
                    run = 0
                    if s < 0:
                        s += 1

                # Update for next iteration
                if scheme == Encoder.CONTEXTUAL:
                    ctx = get_context(image, x0, y0, x1, y1, x, y, step)
                    sum_sq = ctx.sum * ctx.sum
                    # Compute new run_coder
                    if quality == 1024:
                        run_coder = 1 if ctx.sum < 2 else 0
                    else:
                        if ctx.sum < 4 and ctx.sum2 < 2:
                            run_coder = 4
                        elif ctx.sum < 8 and ctx.sum2 < 4:
                            run_coder = 3
                        elif 2 * sum_sq < 3 * ctx.sum2 + 48:
                            run_coder = 2
                        elif 2 * sum_sq < 5 * ctx.sum2 + 32:
                            run_coder = 1
                        else:
                            run_coder = 0

                    # Only update run_coder if s and run_coder have same zero-ness
                    if bool(s) == bool(run_coder):
                        pass  # keep the computed value
                    else:
                        run_coder = 0  # Reset


def main():
    from src.pygfwx.core.lifting import lift

    np.random.seed(42)
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

    # Apply wavelet transform
    aux = img.astype(np.int32)
    boost = 8  # lossy mode boost
    aux = aux * boost
    lift(aux, 0, 0, 8, 8, 1, 7)  # filter=7 (cubic)

    # Apply quantization
    from src.pygfwx.core.quantization import quantize

    quality = 64
    max_q = 1024 * boost
    quantize(aux, 0, 0, 8, 8, 1, quality, 0, max_q)

    print("Quantized wavelet coefficients:")
    print(aux)
    print()

    print("=== Tracing step=2 encoding (CONTEXTUAL) ===")
    writer = BitWriter(100)
    trace_encode_step2(aux, writer, 0, 0, 8, 8, 2, Encoder.CONTEXTUAL, quality)


if __name__ == "__main__":
    main()
