"""Bit-level tracing of step=2 encoding."""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.pygfwx.core.bitstream import BitWriter
from src.pygfwx.core.context import Context, get_context
from src.pygfwx.core.golomb_rice import interleaved_encode, signed_encode
from src.pygfwx.core.header import Encoder


class TracingBitWriter:
    """BitWriter that traces all bits written."""

    def __init__(self, max_words: int):
        self._writer = BitWriter(max_words)
        self.bits_written = []
        self._pending_name = None

    def put_bits(self, n: int, bits: int) -> None:
        """Write n bits, tracing them."""
        # Record what we're writing
        bit_str = format(bits, f"0{n}b")
        self.bits_written.append((self._pending_name, n, bit_str))
        self._writer.put_bits(n, bits)

    def set_name(self, name: str):
        """Set name for next write."""
        self._pending_name = name

    def flush_write_word(self):
        self._writer.flush_write_word()

    def get_data(self):
        return self._writer.get_data()

    @property
    def buffer(self):
        return self._writer.buffer


def trace_encode_step2():
    from src.pygfwx.core.lifting import lift
    from src.pygfwx.core.quantization import quantize

    np.random.seed(42)
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

    aux = img.astype(np.int32)
    boost = 8
    aux = aux * boost
    lift(aux, 0, 0, 8, 8, 1, 7)

    quality = 64
    max_q = 1024 * boost
    quantize(aux, 0, 0, 8, 8, 1, quality, 0, max_q)

    print("Coefficients at step=2 positions:")
    step = 2
    positions = []
    for y in range(0, 8, step):
        x_step = step if (y & step) else step * 2
        for x in range(x_step - step, 8, x_step):
            positions.append((x, y, int(aux[y, x])))

    for x, y, v in positions:
        print(f"  [{y},{x}]: {v}")

    print()

    # Manual encoding with tracing
    writer = TracingBitWriter(100)
    sizex = 8
    sizey = 8
    x0, y0, x1, y1 = 0, 0, 8, 8
    has_dc = False
    is_chroma = False

    context = Context(sum=0, sum2=0)
    run = 0
    run_coder = 0

    for y in range(0, sizey, step):
        x_step = step if (y & step) else step * 2

        for x in range(x_step - step, sizex, x_step):
            s = int(aux[y0 + y, x0 + x])

            if run_coder and s == 0:
                run += 1
                print(f"[{y},{x}]: s={s}, run++ -> {run}")
            else:
                if run_coder:
                    print(f"[{y},{x}]: s={s}, break run={run}, encode run")
                    # unsigned_encode(run_coder, run, writer)
                    run = 0
                    if s < 0:
                        s += 1

                # Get context
                context = get_context(aux, x0, y0, x1, y1, x, y, step)

                # Encode
                sum_sq = context.sum * context.sum
                sum2 = context.sum2
                threshold = 250 if is_chroma else 100

                print(f"[{y},{x}]: s={s}, sum={context.sum}, sum2={sum2}, sumSq={sum_sq}")

                if sum_sq < 2 * sum2 + threshold:
                    print(f"  -> interleaved_encode(0, {s})")
                    # Trace interleaved_encode
                    _trace_interleaved(writer, 0, s)
                elif sum_sq < 2 * sum2 + 950:
                    print(f"  -> interleaved_encode(1, {s})")
                    _trace_interleaved(writer, 1, s)
                elif sum_sq < 3 * sum2 + 3000:
                    if sum_sq < 5 * sum2 + 400:
                        print(f"  -> signed_encode(1, {s})")
                        _trace_signed(writer, 1, s)
                    else:
                        print(f"  -> interleaved_encode(2, {s})")
                        _trace_interleaved(writer, 2, s)
                elif sum_sq < 3 * sum2 + 12000:
                    if sum_sq < 5 * sum2 + 3000:
                        print(f"  -> signed_encode(2, {s})")
                        _trace_signed(writer, 2, s)
                    else:
                        print(f"  -> interleaved_encode(3, {s})")
                        _trace_interleaved(writer, 3, s)
                elif sum_sq < 4 * sum2 + 44000:
                    if sum_sq < 6 * sum2 + 12000:
                        print(f"  -> signed_encode(3, {s})")
                        _trace_signed(writer, 3, s)
                    else:
                        print(f"  -> interleaved_encode(4, {s})")
                        _trace_interleaved(writer, 4, s)
                else:
                    print(f"  -> signed_encode(4, {s})")
                    _trace_signed(writer, 4, s)

                # Update run_coder
                if bool(s) == bool(run_coder):
                    sum_sq = context.sum * context.sum
                    if quality == 1024:
                        run_coder = 1 if context.sum < 2 else 0
                    else:
                        if context.sum < 4 and context.sum2 < 2:
                            run_coder = 4
                        elif context.sum < 8 and context.sum2 < 4:
                            run_coder = 3
                        elif 2 * sum_sq < 3 * context.sum2 + 48:
                            run_coder = 2
                        elif 2 * sum_sq < 5 * context.sum2 + 32:
                            run_coder = 1
                        else:
                            run_coder = 0

    if run > 0:
        print(f"Flush run={run}")
        # unsigned_encode(run_coder, run, writer)

    writer.flush_write_word()
    data = writer.get_data()
    print()
    print(f"Total: {len(data)} bytes")
    print(f"Bits written: {writer.bits_written}")


def _trace_interleaved(writer, pot, s):
    """Trace interleaved encoding."""
    # Convert signed to interleaved
    u = -2 * s - 1 if s < 0 else 2 * s

    q = u >> pot
    r = u & ((1 << pot) - 1) if pot > 0 else 0

    print(f"    interleaved: u={u}, q={q}, r={r}")
    print(f"    write {q + 1} ones then 0, then {pot} bits for r={r}")


def _trace_signed(writer, pot, s):
    """Trace signed encoding."""
    a = abs(s)
    q = a >> pot
    r = a & ((1 << pot) - 1) if pot > 0 else 0

    print(f"    signed: a={a}, q={q}, r={r}")
    print(f"    write {q + 1} ones then 0, then {pot} bits for r={r}, then sign bit")


if __name__ == "__main__":
    trace_encode_step2()
