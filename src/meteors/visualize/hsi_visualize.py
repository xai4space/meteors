from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from meteors import HSI
from meteors.attr import HSIAttributes


def visualize_hsi(image_or_attributes: HSI | HSIAttributes, ax: Axes | None) -> Axes:
    """Visualizes a Hyperspectral image object on the given axes. It uses either the object from HSI class or a field
    from the HSIAttributes class.

    Parameters:
        image_or_attributes (HSI | HSIAttributes): The hyperspectral image, or the attributes to be visualized.
        ax (matplotlib.axes.Axes | None): The axes on which the image will be plotted.
            If None, the current axes will be used.

    Returns:
        matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    if isinstance(image_or_attributes, HSIAttributes):
        image = image_or_attributes.hsi
    else:
        image = image_or_attributes

    rgb = image.get_rgb_image(output_channel_axis=2)
    ax = ax or plt.gca()
    ax.imshow(rgb)

    return ax
