"""
Pytest configuration and fixtures for PyGFWX tests.

This module provides:
- SDK availability detection
- Test image fixtures (using centralized reference_images module)
- Common fixtures for testing
"""

from pathlib import Path

import numpy as np
import pytest

from pygfwx.utils.reference_images import (
    create_impulse_image,
    create_reference_image,
    create_uniform_image,
)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def detect_sdk() -> Path | None:  # cm:c9d0e1b — detect_sdk(): locate GFWX SDK .dll/.so for cross-validation tests
    """Detect if the GFWX SDK shared library is available.

    Returns:
        Path to the SDK library if found, None otherwise.
    """
    # Possible SDK locations
    sdk_paths = [
        PROJECT_ROOT / "gfwx-sdk" / "build" / "Release" / "gfwx.dll",
        PROJECT_ROOT / "gfwx-sdk" / "build" / "Debug" / "gfwx.dll",
        PROJECT_ROOT / "gfwx-sdk" / "build" / "libgfwx.so",
        PROJECT_ROOT / "gfwx-sdk" / "build" / "libgfwx.dylib",
    ]

    for path in sdk_paths:
        if path.exists():
            return path

    return None


# Check SDK availability at module load time
SDK_PATH = detect_sdk()
SDK_AVAILABLE = SDK_PATH is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "sdk: tests that require the GFWX SDK")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip SDK tests if SDK is not available."""
    if not SDK_AVAILABLE:
        skip_sdk = pytest.mark.skip(reason="GFWX SDK not available")
        for item in items:
            if "sdk" in item.keywords:
                item.add_marker(skip_sdk)


# =============================================================================
# Reference Image Fixtures
# =============================================================================
# These use the centralized reference_images module to ensure consistency
# across all tests, examples, and validation code.


@pytest.fixture
def reference_image() -> np.ndarray:  # cm:f2a3b4c — reference_image fixture: 64×64 mono 8-bit with mixed frequency content
    """Primary reference image (64x64 mono) with mixed frequency content.

    This is the main test image that should be used for most tests.
    It has high and low frequency content in both H and V directions.
    """
    return create_reference_image(size=64, channels=1, bit_depth=8)


@pytest.fixture
def reference_image_rgb() -> np.ndarray:
    """Primary reference image as RGB (64x64x3).

    Each channel has distinct content to catch channel-mixing bugs.
    """
    return create_reference_image(size=64, channels=3, bit_depth=8)


@pytest.fixture
def reference_image_16bit() -> np.ndarray:
    """Primary reference image as 16-bit mono (64x64)."""
    return create_reference_image(size=64, channels=1, bit_depth=16)


@pytest.fixture
def uniform_image() -> np.ndarray:
    """Uniform gray image (64x64, value 128).

    Useful for testing DC handling and quantization edge cases.
    """
    return create_uniform_image(size=64, value=128, channels=1, bit_depth=8)


@pytest.fixture
def impulse_image() -> np.ndarray:
    """Image with single bright pixel at center (64x64).

    Useful for testing impulse response and boundary handling.
    """
    return create_impulse_image(size=64, channels=1, bit_depth=8)


# =============================================================================
# SDK Fixtures (only available when SDK is present)
# =============================================================================


@pytest.fixture
def sdk_wrapper():
    """Provide the SDK wrapper if available.

    Tests using this fixture will be skipped if SDK is not available.
    """
    if not SDK_AVAILABLE:
        pytest.skip("GFWX SDK not available")

    # Import here to avoid errors when SDK not available
    try:
        from cross_codec.gfwx_sdk import GFWXSDK

        return GFWXSDK()
    except ImportError:
        pytest.skip("GFWX SDK wrapper not implemented yet")


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_compressed_data() -> bytes | None:
    """Load sample compressed GFWX data if available.

    Returns None if no sample data exists yet.
    """
    sample_path = PROJECT_ROOT / "tests" / "data" / "sample.gfwx"
    if sample_path.exists():
        return sample_path.read_bytes()
    return None
