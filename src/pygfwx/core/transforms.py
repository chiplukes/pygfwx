"""
GFWX Color Transforms.

Implements forward and inverse color transforms used by GFWX:
- UYV (YUV variant): Standard luma/chroma separation
- A710: Higher quality color transform

The transforms are specified as programs (arrays of integers) that
describe how to compute each output channel from combinations of
input channels.

Transform program format:
  [dest_channel, src_channel, factor, src_channel, factor, ..., -1, denom, is_chroma, ...]
  Ends with -1 to mark end of entire program

Reference: gfwx.h GFWX_TRANSFORM_UYV, GFWX_TRANSFORM_A710, transformTerm()
"""

import numpy as np

from pygfwx.core.bitstream import BitReader, BitWriter
from pygfwx.core.golomb_rice import signed_decode, signed_encode

# Pre-defined transform programs (matching SDK defines)

# UYV transform program: R -= G (chroma); B -= G (chroma); G += (R + B) / 4 (luma)
# Format: [dest, src, factor, ..., -1, denom, is_chroma] for each channel, ending with -1
# SDK: GFWX_TRANSFORM_UYV = { 0, 1, -1, -1, 1, 1, 2, 1, -1, -1, 1, 1, 1, 0, 1, 2, 1, -1, 4, 0, -1 }
TRANSFORM_UYV_PROGRAM = [  # cm:b2c3d4 — TRANSFORM_UYV_PROGRAM: UYV color transform program constant (SDK-compatible)
    0, 1, -1, -1, 1, 1,       # Channel 0 (R): subtract G*1, div by 1, chroma=1
    2, 1, -1, -1, 1, 1,       # Channel 2 (B): subtract G*1, div by 1, chroma=1
    1, 0, 1, 2, 1, -1, 4, 0,  # Channel 1 (G): add (R+B)/4, div by 4, chroma=0
    -1                         # End of program
]

# A710 transform for RGB order
# R -= G (chroma)
# B -= (G * 2 + R) / 2 (chroma)
# G += (B * 2 + R * 3) / 8 (luma)
TRANSFORM_A710_RGB = [
    0, 1, -1, -1, 1, 1,                   # R -= G, chroma
    2, 1, -2, 0, -1, -1, 2, 1,            # B -= (G*2 + R)/2, chroma
    1, 2, 2, 0, 3, -1, 8, 0,              # G += (B*2 + R*3)/8, luma
    -1
]


def forward_transform_uyv(  # cm:e5f6a7b — forward_transform_uyv(): RGB→YUV-like (R'=R-G, B'=B-G, G'=G+(R'+B')/4)
    image: np.ndarray,
    boost: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """
    Apply forward UYV color transform.

    Converts RGB to a YUV-like representation:
    - R' = R - G  (chroma)
    - B' = B - G  (chroma)
    - G' = G + (R' + B') / 4  (luma-like)

    Args:
        image: Input image shape (H, W, 3) or (H, W, C).
        boost: Scale factor (8 for lossy, 1 for lossless).

    Returns:
        Tuple of (transformed_image, transform_program).
        The program is needed for bitstream encoding.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("UYV transform requires at least 3 channels")

    height, width, channels = image.shape
    result = np.zeros((channels, height, width), dtype=np.int32)

    # Convert to working type and apply boost
    r = image[:, :, 0].astype(np.int32) * boost
    g = image[:, :, 1].astype(np.int32) * boost
    b = image[:, :, 2].astype(np.int32) * boost

    # Step 1: R -= G (store in result[0])
    result[0] = r - g

    # Step 2: B -= G (store in result[2])
    result[2] = b - g

    # Step 3: G += (R' + B') / 4 (where R', B' are the transformed values)
    result[1] = g + (result[0] + result[2]) // 4

    # Copy any additional channels unchanged
    for c in range(3, channels):
        result[c] = image[:, :, c].astype(np.int32) * boost

    return result, TRANSFORM_UYV_PROGRAM


def inverse_transform_uyv(
    aux_data: np.ndarray,
    boost: int = 8,
) -> np.ndarray:
    """
    Apply inverse UYV color transform.

    Converts from YUV-like back to RGB:
    - G = G' - (R' + B') / 4
    - R = R' + G
    - B = B' + G

    Args:
        aux_data: Transformed data shape (C, H, W).
        boost: Scale factor used during forward transform.

    Returns:
        Image data shape (H, W, C).
    """
    channels, height, width = aux_data.shape
    result = np.zeros((height, width, channels), dtype=np.int32)

    r_prime = aux_data[0]
    g_prime = aux_data[1]
    b_prime = aux_data[2]

    # Reverse step 3: G' = G + (R' + B') / 4  =>  G = G' - (R' + B') / 4
    g = g_prime - (r_prime + b_prime) // 4

    # Reverse step 1 & 2: R' = R - G, B' = B - G  =>  R = R' + G, B = B' + G
    r = r_prime + g
    b = b_prime + g

    result[:, :, 0] = r // boost
    result[:, :, 1] = g // boost
    result[:, :, 2] = b // boost

    # Copy additional channels
    for c in range(3, channels):
        result[:, :, c] = aux_data[c] // boost

    return result


def forward_transform_generic(  # cm:c8d9e0 — forward_transform_generic(): apply any SDK-format transform program
    image: np.ndarray,
    program: list[int],
    boost: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """
    Apply a generic forward color transform using a program.

    This matches the SDK's transform mechanism where a program describes
    how to compute each output channel.

    Args:
        image: Input image shape (H, W, C).
        program: Transform program array.
        boost: Scale factor.

    Returns:
        Tuple of (transformed_data shape (C, H, W), is_chroma per channel).
    """
    height, width, channels = image.shape
    buffer_size = height * width

    # Initialize aux_data
    aux_data = np.zeros((channels, height, width), dtype=np.int64)
    is_chroma = [-1] * channels  # -1 means not yet processed

    pc = 0  # Program counter

    while pc < len(program) and program[pc] >= 0:
        # Read destination channel
        dest_channel = program[pc]
        pc += 1

        # Accumulate transform term
        destination = np.zeros(buffer_size, dtype=np.int64)

        while pc < len(program) and program[pc] >= 0:
            src_channel = program[pc]
            pc += 1
            factor = program[pc]
            pc += 1

            if is_chroma[src_channel] == -1:
                # Source from original image
                src_data = image[:, :, src_channel].reshape(-1).astype(np.int64)
                destination += src_data * (boost * factor)
            else:
                # Source from already-transformed aux_data
                src_data = aux_data[src_channel].reshape(-1)
                destination += src_data * factor

        # Skip -1 marker
        pc += 1

        # Read denominator
        denom = program[pc]
        pc += 1

        # Apply division
        if denom == 2:
            destination >>= 1
        elif denom == 4:
            destination >>= 2
        elif denom == 8:
            destination >>= 3
        elif denom > 1:
            destination //= denom

        # Add original image contribution (dest channel)
        original = image[:, :, dest_channel].reshape(-1).astype(np.int64) * boost
        destination += original

        # Store result
        aux_data[dest_channel] = destination.reshape(height, width)

        # Read is_chroma flag
        is_chroma[dest_channel] = program[pc]
        pc += 1

    # Copy any channels without transforms
    for c in range(channels):
        if is_chroma[c] == -1:
            aux_data[c] = image[:, :, c].astype(np.int64) * boost
            is_chroma[c] = 0

    return aux_data.astype(np.int32), is_chroma


def inverse_transform_generic(
    aux_data: np.ndarray,
    program: list[int],
    boost: int = 8,
) -> np.ndarray:
    """
    Apply generic inverse color transform using a program.

    The inverse transform runs the program steps in reverse order,
    subtracting the computed terms instead of adding.

    Args:
        aux_data: Transformed data shape (C, H, W).
        program: Transform program array.
        boost: Scale factor used during forward transform.

    Returns:
        Reconstructed image shape (H, W, C).
    """
    channels, height, width = aux_data.shape
    buffer_size = height * width

    # Parse program to find step boundaries
    steps = []
    pc = 0
    while pc < len(program) and program[pc] >= 0:
        steps.append(pc)
        pc += 1  # dest

        # Skip source terms
        while pc < len(program) and program[pc] >= 0:
            pc += 2  # src, factor pairs
        pc += 1  # -1 marker
        pc += 1  # denom
        pc += 1  # is_chroma

    # Make a working copy to avoid modifying input
    working = aux_data.copy().astype(np.int64)

    # Process steps in reverse order (matching SDK's inverse transform)
    for s in range(len(steps) - 1, -1, -1):
        pc = steps[s]
        dest_channel = program[pc]
        pc += 1

        # Compute transform term to subtract
        transform_temp = np.zeros(buffer_size, dtype=np.int64)

        while program[pc] >= 0:
            src_channel = program[pc]
            pc += 1
            factor = program[pc]
            pc += 1

            src_data = working[src_channel].reshape(-1)
            transform_temp += src_data * factor

        pc += 1  # Skip -1 marker

        # Read and apply denominator
        denom = program[pc]
        if denom == 2:
            transform_temp >>= 1
        elif denom == 4:
            transform_temp >>= 2
        elif denom == 8:
            transform_temp >>= 3
        elif denom > 1:
            transform_temp //= denom

        # Subtract from destination channel and reshape back
        dest_flat = working[dest_channel].reshape(-1) - transform_temp
        working[dest_channel] = dest_flat.reshape(height, width)

    # Apply boost division and convert to output format
    result = np.zeros((height, width, channels), dtype=np.int32)
    for c in range(channels):
        result[:, :, c] = working[c] // boost

    return result


def write_transform_program(
    writer: BitWriter,
    program: list[int] | None,
) -> None:
    """
    Write transform program to bitstream.

    If program is None or empty, writes -1 to indicate no transform.

    Args:
        writer: BitWriter to write to.
        program: Transform program or None.
    """
    if program is None or len(program) == 0:
        # No transform
        signed_encode(2, -1, writer)
    else:
        # Write each integer in the program
        for value in program:
            signed_encode(2, value, writer)

    writer.flush_write_word()


def read_transform_program(reader: BitReader) -> list[int] | None:
    """
    Read transform program from bitstream.

    The format is a sequence of terms, each with:
      dest, [src, factor]*, -1, denom, is_chroma
    The program ends when we read -1 as a dest channel.

    Args:
        reader: BitReader to read from.

    Returns:
        Transform program list, or None if no transform.
    """
    program = []

    while True:
        # Read destination channel (or -1 to end program)
        dest = signed_decode(2, reader)
        program.append(dest)

        if dest == -1:
            # End of program
            if len(program) == 1:
                # First value is -1, no transform
                reader.flush_read_word()
                return None
            break

        # Read source/factor pairs until we hit -1
        while True:
            src = signed_decode(2, reader)
            program.append(src)

            if src == -1:
                # End of source/factor list
                break

            # Read the factor for this source
            factor = signed_decode(2, reader)
            program.append(factor)

        # Read denominator
        denom = signed_decode(2, reader)
        program.append(denom)

        # Read is_chroma flag
        is_chroma = signed_decode(2, reader)
        program.append(is_chroma)

    reader.flush_read_word()
    return program


def get_chroma_flags(
    channels: int,
    has_transform: bool,
    transform_program: list[int] | None = None,
) -> list[int]:
    """
    Get chroma flags for each channel.

    Chroma channels (U, V in YUV) get different quality settings.

    Args:
        channels: Number of channels.
        has_transform: Whether color transform is used.
        transform_program: The transform program (if used).

    Returns:
        List of chroma flags (0=luma, 1=chroma) per channel.
    """
    if not has_transform or transform_program is None:
        # No transform: all channels are luma-like
        return [0] * channels

    # Parse transform program to extract chroma flags
    is_chroma = [-1] * channels
    pc = 0

    while pc < len(transform_program) and transform_program[pc] >= 0:
        dest = transform_program[pc]
        pc += 1

        # Skip source terms
        while pc < len(transform_program) and transform_program[pc] >= 0:
            pc += 2  # Skip src, factor pairs

        pc += 1  # Skip -1 marker
        pc += 1  # Skip denom

        # Read chroma flag
        if pc < len(transform_program):
            is_chroma[dest] = transform_program[pc]
            pc += 1

    # Set remaining channels to 0 (luma)
    for c in range(channels):
        if is_chroma[c] == -1:
            is_chroma[c] = 0

    return is_chroma

# =============================================================================
# Custom Transform Building Utilities
# =============================================================================


def build_transform_step(
    dest_channel: int,
    sources: list[tuple[int, int]],
    denominator: int,
    is_chroma: bool,
) -> list[int]:
    """
    Build a single transform step as a program fragment.

    A transform step computes:
        dest += sum(src * factor for src, factor in sources) / denominator

    Args:
        dest_channel: Index of channel to modify.
        sources: List of (source_channel, factor) pairs.
        denominator: Division factor (1 for no division).
        is_chroma: Whether this channel is chroma (affects quality).

    Returns:
        Program fragment for this step.

    Example:
        # Create step: R -= G (i.e., R += G * -1)
        step = build_transform_step(0, [(1, -1)], 1, is_chroma=True)
        # Returns: [0, 1, -1, -1, 1, 1]
    """
    step = [dest_channel]

    for src, factor in sources:
        step.append(src)
        step.append(factor)

    step.append(-1)  # End of sources marker
    step.append(denominator)
    step.append(1 if is_chroma else 0)

    return step


def build_transform_program(*steps: list[int]) -> list[int]:
    """
    Build a complete transform program from individual steps.

    Args:
        *steps: Variable number of step fragments from build_transform_step().

    Returns:
        Complete program with end marker.

    Example:
        # UYV-like transform
        program = build_transform_program(
            build_transform_step(0, [(1, -1)], 1, is_chroma=True),   # R -= G
            build_transform_step(2, [(1, -1)], 1, is_chroma=True),   # B -= G
            build_transform_step(1, [(0, 1), (2, 1)], 4, is_chroma=False),  # G += (R+B)/4
        )
    """
    program = []
    for step in steps:
        program.extend(step)
    program.append(-1)  # End of program marker
    return program


def validate_transform_program(program: list[int], num_channels: int) -> bool:
    """
    Validate a transform program for correctness.

    Args:
        program: Transform program to validate.
        num_channels: Number of channels in the image.

    Returns:
        True if valid, raises ValueError if invalid.

    Raises:
        ValueError: If program is malformed or references invalid channels.
    """
    if not program:
        return True  # Empty program is valid (no transform)

    if program[-1] != -1:
        raise ValueError("Program must end with -1")

    pc = 0
    steps_seen = []

    while pc < len(program) and program[pc] >= 0:
        dest = program[pc]
        if dest < 0 or dest >= num_channels:
            raise ValueError(f"Invalid destination channel {dest}, must be 0-{num_channels - 1}")
        steps_seen.append(dest)
        pc += 1

        # Read source/factor pairs
        while pc < len(program) and program[pc] >= 0:
            src = program[pc]
            if src < 0 or src >= num_channels:
                raise ValueError(f"Invalid source channel {src}, must be 0-{num_channels - 1}")
            pc += 1

            if pc >= len(program):
                raise ValueError("Missing factor after source channel")
            # factor = program[pc]  # Any value is valid
            pc += 1

        # Check for -1 marker
        if pc >= len(program) or program[pc] != -1:
            raise ValueError("Missing -1 marker after source/factor pairs")
        pc += 1

        # Check denominator
        if pc >= len(program):
            raise ValueError("Missing denominator")
        denom = program[pc]
        if denom <= 0:
            raise ValueError(f"Invalid denominator {denom}, must be positive")
        pc += 1

        # Check is_chroma flag
        if pc >= len(program):
            raise ValueError("Missing is_chroma flag")
        chroma = program[pc]
        if chroma not in (0, 1):
            raise ValueError(f"Invalid is_chroma flag {chroma}, must be 0 or 1")
        pc += 1

    return True


def parse_transform_steps(program: list[int]) -> list[dict]:
    """
    Parse a transform program into a list of step dictionaries.

    Useful for debugging and understanding transform programs.

    Args:
        program: Transform program to parse.

    Returns:
        List of dictionaries, each with:
        - 'dest': Destination channel
        - 'sources': List of (src_channel, factor) tuples
        - 'denominator': Division factor
        - 'is_chroma': Boolean chroma flag
    """
    if not program or program[0] < 0:
        return []

    steps = []
    pc = 0

    while pc < len(program) and program[pc] >= 0:
        step = {"dest": program[pc]}
        pc += 1

        sources = []
        while pc < len(program) and program[pc] >= 0:
            src = program[pc]
            pc += 1
            factor = program[pc]
            pc += 1
            sources.append((src, factor))

        step["sources"] = sources
        pc += 1  # Skip -1 marker

        step["denominator"] = program[pc]
        pc += 1

        step["is_chroma"] = bool(program[pc])
        pc += 1

        steps.append(step)

    return steps


def describe_transform_program(program: list[int], channel_names: list[str] | None = None) -> str:
    """
    Generate a human-readable description of a transform program.

    Args:
        program: Transform program to describe.
        channel_names: Optional list of channel names (e.g., ['R', 'G', 'B']).

    Returns:
        Multi-line string describing each step.
    """
    if not program or program[0] < 0:
        return "No transform (identity)"

    steps = parse_transform_steps(program)
    lines = []

    for i, step in enumerate(steps):
        dest = step["dest"]
        dest_name = channel_names[dest] if channel_names and dest < len(channel_names) else f"Ch{dest}"

        # Build expression
        terms = []
        for src, factor in step["sources"]:
            src_name = channel_names[src] if channel_names and src < len(channel_names) else f"Ch{src}"
            if factor == 1:
                terms.append(src_name)
            elif factor == -1:
                terms.append(f"-{src_name}")
            else:
                terms.append(f"{src_name}*{factor}")

        expr = " + ".join(terms) if terms else "0"
        denom = step["denominator"]
        if denom > 1:
            expr = f"({expr}) / {denom}"

        chroma = "chroma" if step["is_chroma"] else "luma"
        lines.append(f"Step {i + 1}: {dest_name} += {expr}  [{chroma}]")

    return "\n".join(lines)
