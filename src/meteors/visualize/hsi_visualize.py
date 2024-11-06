from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from meteors import HSI
from meteors.attr import HSIAttributes


def visualize_hsi(hsi_or_attributes: HSI | HSIAttributes, ax: Axes | None = None, use_mask: bool = True) -> Axes:
    """Visualizes a Hyperspectral image object on the given axes. It uses either the object from HSI class or a field
    from the HSIAttributes class.

    Parameters:
        hsi_or_attributes (HSI | HSIAttributes): The hyperspectral image, or the attributes to be visualized.
        ax (matplotlib.axes.Axes | None): The axes on which the image will be plotted.
            If None, the current axes will be used.
        use_mask (bool): Whether to use the image mask if provided for the visualization.


    Returns:
        matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    Raises:
        TypeError: If hsi_or_attributes is not an instance of HSI or HSIAttributes.
    """
    if isinstance(hsi_or_attributes, HSIAttributes):
        hsi = hsi_or_attributes.hsi
    else:
        hsi = hsi_or_attributes

    if not isinstance(hsi, HSI):
        raise TypeError("hsi_or_attributes must be an instance of HSI or HSIAttributes.")

    hsi = hsi.change_orientation("HWC", inplace=False)

    rgb = hsi.get_rgb_image(output_channel_axis=2, apply_mask=use_mask, normalize=True).cpu().numpy()
    ax = ax or plt.gca()
    ax.imshow(rgb)

    return ax
