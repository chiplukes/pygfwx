"""
Multi-Layer Support for GFWX.

This module provides utilities for handling multi-layer GFWX files:
- Stereo image pairs (left/right views)
- Depth map + color image combinations
- Animation frames in a single file
- Any multi-frame/multi-view data

Data Layout:
GFWX stores multi-layer data in an interleaved format:
- For layers=L, channels=C, each pixel stores L*C values
- Values are organized as: [L0C0, L0C1, ..., L0C(C-1), L1C0, L1C1, ..., L1C(C-1), ...]
- Example for stereo RGB (layers=2, channels=3):
  [Left_R, Left_G, Left_B, Right_R, Right_G, Right_B] per pixel

Reference: gfwx.h compress() lines 728-731, decompress() lines 924-940
"""

from dataclasses import dataclass

import numpy as np

from pygfwx.core.header import GFWXHeader


@dataclass
class MultiLayerImage:
    """
    Container for multi-layer image data.

    Provides convenient access to individual layers while maintaining
    the interleaved format required by GFWX.

    Attributes:
        data: Raw interleaved data array shape (H, W, layers*channels)
        layers: Number of layers
        channels: Channels per layer
        height: Image height
        width: Image width
    """

    data: np.ndarray
    """Raw interleaved data (H, W) for mono or (H, W, layers*channels)."""

    layers: int
    """Number of layers (e.g., 2 for stereo)."""

    channels: int
    """Channels per layer (e.g., 3 for RGB)."""

    @property
    def height(self) -> int:
        """Image height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Image width."""
        return self.data.shape[1]

    @property
    def total_channels(self) -> int:
        """Total channels (layers * channels)."""
        return self.layers * self.channels

    @property
    def dtype(self) -> np.dtype:
        """Data type of the image."""
        return self.data.dtype

    def get_layer(self, layer_index: int) -> np.ndarray:
        """
        Extract a single layer from the multi-layer image.

        Args:
            layer_index: Layer to extract (0-based).

        Returns:
            Image array for that layer, shape (H, W) or (H, W, C).

        Raises:
            IndexError: If layer_index is out of range.
        """
        if layer_index < 0 or layer_index >= self.layers:
            raise IndexError(f"Layer index {layer_index} out of range [0, {self.layers})")

        if self.total_channels == 1:
            # Single layer, single channel
            return self.data

        start_ch = layer_index * self.channels
        end_ch = start_ch + self.channels

        if self.channels == 1:
            return self.data[:, :, start_ch]
        else:
            return self.data[:, :, start_ch:end_ch]

    def set_layer(self, layer_index: int, layer_data: np.ndarray) -> None:
        """
        Set a single layer in the multi-layer image.

        Args:
            layer_index: Layer to set (0-based).
            layer_data: Image array for that layer.

        Raises:
            IndexError: If layer_index is out of range.
            ValueError: If layer_data shape is incompatible.
        """
        if layer_index < 0 or layer_index >= self.layers:
            raise IndexError(f"Layer index {layer_index} out of range [0, {self.layers})")

        start_ch = layer_index * self.channels
        end_ch = start_ch + self.channels

        if self.channels == 1:
            if layer_data.ndim == 2:
                self.data[:, :, start_ch] = layer_data
            elif layer_data.ndim == 3 and layer_data.shape[2] == 1:
                self.data[:, :, start_ch] = layer_data[:, :, 0]
            else:
                raise ValueError(
                    f"Expected shape (H, W) or (H, W, 1), got {layer_data.shape}"
                )
        else:
            if layer_data.ndim == 3 and layer_data.shape[2] == self.channels:
                self.data[:, :, start_ch:end_ch] = layer_data
            else:
                raise ValueError(
                    f"Expected shape (H, W, {self.channels}), got {layer_data.shape}"
                )


def create_multi_layer(
    *layers: np.ndarray,
    dtype: np.dtype | None = None,
) -> MultiLayerImage:
    """
    Create a multi-layer image from individual layers.

    All layers must have the same dimensions and number of channels.
    The output will have interleaved data suitable for GFWX encoding.

    Args:
        *layers: One or more image arrays. Each should be (H, W) or (H, W, C).
        dtype: Output dtype. If None, uses the dtype of the first layer.

    Returns:
        MultiLayerImage with interleaved data.

    Raises:
        ValueError: If layers have incompatible shapes.

    Example:
        # Create stereo pair
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        stereo = create_multi_layer(left, right)
        # stereo.data has shape (480, 640, 6) - interleaved RGBRGB
    """
    if len(layers) == 0:
        raise ValueError("At least one layer is required")

    # Get reference shape from first layer
    ref = layers[0]
    if ref.ndim == 2:
        height, width = ref.shape
        channels = 1
    elif ref.ndim == 3:
        height, width, channels = ref.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got {ref.ndim}D")

    if dtype is None:
        dtype = ref.dtype

    num_layers = len(layers)

    # Validate all layers have same shape
    for i, layer in enumerate(layers):
        if layer.ndim == 2:
            lh, lw = layer.shape
            lc = 1
        elif layer.ndim == 3:
            lh, lw, lc = layer.shape
        else:
            raise ValueError(f"Layer {i}: Expected 2D or 3D array, got {layer.ndim}D")

        if (lh, lw, lc) != (height, width, channels):
            raise ValueError(
                f"Layer {i} shape ({lh}, {lw}, {lc}) doesn't match "
                f"reference ({height}, {width}, {channels})"
            )

    # Create interleaved output
    total_channels = num_layers * channels
    if total_channels == 1:
        output = layers[0].astype(dtype)
    else:
        output = np.zeros((height, width, total_channels), dtype=dtype)
        for i, layer in enumerate(layers):
            start_ch = i * channels
            end_ch = start_ch + channels
            if layer.ndim == 2:
                output[:, :, start_ch] = layer.astype(dtype)
            else:
                output[:, :, start_ch:end_ch] = layer.astype(dtype)

    return MultiLayerImage(data=output, layers=num_layers, channels=channels)


def split_layers(
    image: np.ndarray,
    layers: int,
    channels: int,
) -> list[np.ndarray]:
    """
    Split an interleaved multi-layer image into individual layers.

    Args:
        image: Interleaved image array (H, W) or (H, W, layers*channels).
        layers: Number of layers.
        channels: Channels per layer.

    Returns:
        List of layer arrays.

    Raises:
        ValueError: If image dimensions don't match layers*channels.
    """
    total = layers * channels

    if image.ndim == 2:
        if total != 1:
            raise ValueError(
                f"2D image implies 1 total channel, but layers*channels={total}"
            )
        return [image]

    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    if image.shape[2] != total:
        raise ValueError(
            f"Image has {image.shape[2]} channels, expected {total}"
        )

    result = []
    for i in range(layers):
        start_ch = i * channels
        end_ch = start_ch + channels
        if channels == 1:
            result.append(image[:, :, start_ch])
        else:
            result.append(image[:, :, start_ch:end_ch])

    return result


def decode_result_to_multi_layer(
    image: np.ndarray,
    header: GFWXHeader,
) -> MultiLayerImage:
    """
    Convert a decode result to a MultiLayerImage.

    Args:
        image: Decoded image array.
        header: Parsed GFWX header.

    Returns:
        MultiLayerImage wrapping the decoded data.
    """
    return MultiLayerImage(
        data=image,
        layers=header.layers,
        channels=header.channels,
    )


def validate_multi_layer_header(header: GFWXHeader) -> None:
    """
    Validate that a header has valid multi-layer parameters.

    Args:
        header: Header to validate.

    Raises:
        ValueError: If header is invalid.
    """
    if header.layers < 1:
        raise ValueError(f"Invalid layers: {header.layers}")
    if header.channels < 1:
        raise ValueError(f"Invalid channels: {header.channels}")
    if header.layers * header.channels > 65536:
        raise ValueError(
            f"Total channels ({header.layers * header.channels}) exceeds maximum (65536)"
        )
