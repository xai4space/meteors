from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from meteors import Image
from meteors.attr import ImageAttributes


def visualize_image(image: Image | ImageAttributes, ax: Axes | None) -> Axes:
    """Visualizes an LIME image object on the given axes.

    Parameters:
        image (Image | ImageAttributes): The image to be visualized.
        ax (matplotlib.axes.Axes | None): The axes on which the image will be plotted.
            If None, the current axes will be used.

    Returns:
        matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    if isinstance(image, ImageAttributes):
        image = image.image

    rgb = image.get_rgb_image(output_channel_axis=2)
    ax = ax or plt.gca()
    ax.imshow(rgb)

    return ax
