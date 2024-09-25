from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from captum.attr import visualization as viz

from meteors.attr import HSIAttributes


def visualize_attributes(image_attributes: HSIAttributes, use_pyplot: bool = False) -> tuple[Figure, Axes] | None:
    """Visualizes the attributes of an image on the given axes.

    Parameters:
        image_attributes (HSIAttributes): The image attributes to be visualized.
        use_pyplot (bool): If True, uses pyplot to display the image. If False, returns the figure and axes objects.

    Returns:
        matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None .
            If all the attributions are zero, returns None.
    """

    if image_attributes.hsi.orientation != ("C", "H", "W"):
        raise ValueError(f"HSI orientation {image_attributes.hsi.orientation} is not supported yet")

    rotated_attributes_dataclass = image_attributes.change_orientation("HWC", inplace=False)
    rotated_attributes = rotated_attributes_dataclass.attributes.detach().cpu().numpy()
    if np.all(rotated_attributes == 0):
        warnings.warn("All the attributions are zero. There is nothing to visualize.")
        return None

    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0, 0].set_title("Attribution Heatmap")
    ax[0, 0].grid(False)
    ax[0, 0].axis("off")

    fig.suptitle(f"HSI Attributes of: {image_attributes.attribution_method}")

    _ = viz.visualize_image_attr(
        rotated_attributes,
        method="heat_map",
        sign="all",
        plt_fig_axis=(fig, ax[0, 0]),
        show_colorbar=True,
        use_pyplot=False,
    )

    ax[0, 1].set_title("Attribution Module Values")
    ax[0, 1].grid(False)
    ax[0, 1].axis("off")

    # Attributions module values
    _ = viz.visualize_image_attr(
        rotated_attributes,
        method="heat_map",
        sign="absolute_value",
        plt_fig_axis=(fig, ax[0, 1]),
        show_colorbar=True,
        use_pyplot=False,
    )

    sign_attr = np.sign(rotated_attributes).sum(axis=-1) / rotated_attributes.shape[-1]
    sns.heatmap(sign_attr, cmap="PiYG", vmin=-1, vmax=1, square=True, ax=ax[0, 2])
    ax[0, 2].set(title="Attribution Sign Values")
    ax[0, 2].grid(False)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])

    attr_all = rotated_attributes.sum(axis=(0, 1))
    ax[1, 0].scatter(image_attributes.hsi.wavelengths, attr_all, c="r")
    ax[1, 0].set_title("Spectral Attribution")
    ax[1, 0].set_xlabel("Wavelength")
    ax[1, 0].set_ylabel("Attribution")
    ax[1, 0].grid(True)

    attr_abs = np.abs(rotated_attributes).sum(axis=(0, 1))
    ax[1, 1].scatter(image_attributes.hsi.wavelengths, attr_abs, c="b")
    ax[1, 1].set_title("Spectral Attribution Absolute Values")
    ax[1, 1].set_xlabel("Wavelength")
    ax[1, 1].set_ylabel("Attribution Absolute Value")
    ax[1, 1].grid(True)

    # Sign values
    sign_attr = np.sign(rotated_attributes).sum(axis=(0, 1)) / rotated_attributes.shape[0] / rotated_attributes.shape[1]
    ax[1, 2].scatter(image_attributes.hsi.wavelengths, sign_attr, c="g")
    ax[1, 2].set_title("Spectral Attribution Sign Values")
    ax[1, 2].set_xlabel("Wavelength")
    ax[1, 2].set_ylabel("Attribution Sign Proportion")
    ax[1, 2].grid(True)
    ax[1, 2].set_yticks([-1, 0, 1])

    plt.tight_layout()

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover

    return fig, ax
