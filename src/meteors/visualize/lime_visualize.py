from __future__ import annotations

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from captum.attr import visualization as viz

from meteors import Image
from meteors.attr import ImageAttributes, ImageSpatialAttributes, ImageSpectralAttributes


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


def visualize_spatial_attributes(  # noqa: C901
    spatial_attributes: ImageSpatialAttributes, use_pyplot: bool = False
) -> tuple[Figure, Axes] | None:
    """Visualizes the spatial attributes of an image using Lime attribution.

    Args:
        spatial_attributes (ImageSpatialAttributes):
            The spatial attributes of the image object to visualize.
        use_pyplot (bool, optional):
            Whether to use pyplot for visualization. Defaults to False.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Spatial Attributes Visualization")
    ax[0].imshow(spatial_attributes.image.get_rgb_image(output_channel_axis=2).cpu())
    ax[0].set_title("Original image")

    viz.visualize_image_attr(
        spatial_attributes.attributes.cpu().numpy(),  # height, width, channels
        method="heat_map",
        sign="all",
        plt_fig_axis=(fig, ax[1]),
        show_colorbar=True,
        use_pyplot=False,
    )

    if spatial_attributes.segmentation_mask is not None:
        mask = spatial_attributes.segmentation_mask.cpu()
        if mask.ndim == 3:
            mask = mask[0]
        ax[2].imshow(mask / mask.max(), cmap="gray")
        ax[2].set_title("Mask")
        ax[2].grid(False)

    if use_pyplot:
        plt.show()
        return None
    else:
        return fig, ax


def visualize_spectral_attributes(
    spectral_attributes: ImageSpectralAttributes | list[ImageSpectralAttributes],
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
) -> tuple[Figure, Axes] | None:
    """Visualizes the spectral attributes of an image or a list of images.

    Args:
        spectral_attributes (ImageSpectralAttributes | list[ImageSpectralAttributes]):
            The spectral attributes of the image object to visualize.
        use_pyplot (bool, optional):
            If True, displays the visualization using pyplot.
            If False, returns the figure and axes objects. Defaults to False.
        color_palette (list[str] | None, optional):
            The color palette to use for visualizing different spectral bands.
            If None, a default color palette is used.
            Defaults to None.
        show_not_included (bool, optional):
            If True, includes the spectral bands that are not included in the visualization.
            If False, only includes the spectral bands that are included in the visualization.
            Defaults to True.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    band_names = (
        spectral_attributes.band_names
        if isinstance(spectral_attributes, ImageSpectralAttributes)
        else spectral_attributes[0].band_names
    )

    color_palette = color_palette or sns.color_palette("hsv", len(band_names.keys()))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Spectral Attributes Visualization")

    visualize_spectral_attributes_by_waveband(
        spectral_attributes,
        ax[0],
        color_palette=color_palette,
        show_not_included=show_not_included,
    )

    visualize_spectral_attributes_by_magnitude(
        spectral_attributes,
        ax[1],
        color_palette=color_palette,
        show_not_included=show_not_included,
    )

    if use_pyplot:
        plt.show()
        return None
    else:
        return fig, ax


def validate_consistent_band_and_wavelengths(
    band_names: dict[str, int], wavelengths: torch.Tensor, spectral_attributes: list[ImageSpectralAttributes]
) -> None:
    """Validates that all spectral attributes have consistent band names and wavelengths.

    Args:
        band_names (dict[str, int]): A dictionary mapping band names to their indices.
        wavelengths (torch.Tensor): A tensor containing the wavelengths of the image.
        spectral_attributes (list[ImageSpectralAttributes]): A list of spectral attributes.

    Raises:
        ValueError: If the band names or wavelengths of any spectral attribute are inconsistent.
    """
    for attr in spectral_attributes:
        if band_names != attr.band_names:
            raise ValueError("Band names are inconsistent among spectral attributes.")
        if not torch.equal(wavelengths, attr.image.wavelengths):
            raise ValueError("Wavelengths are inconsistent among spectral attributes.")


def setup_visualization(ax: Axes | None, title: str, xlabel: str, ylabel: str) -> Axes:
    """Set up the visualization by configuring the axes with the provided title, xlabel, and ylabel.

    Parameters:
        ax (Axes | None): The axes object to be configured. If None, the current axes will be used.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        Axes: The configured axes object.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def visualize_spectral_attributes_by_waveband(
    spectral_attributes: ImageSpectralAttributes | list[ImageSpectralAttributes],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
) -> Axes:
    """Visualizes spectral attributes by waveband.

    Args:
        spectral_attributes (ImageSpectralAttributes | list[ImageSpectralAttributes]):
            The spectral attributes to visualize.
        ax (Axes | None): The matplotlib axes to plot the visualization on.
            If None, a new axes will be created.
        color_palette (list[str] | None): The color palette to use for plotting.
            If None, a default color palette will be used.
        show_not_included (bool): Whether to show the "not_included" band in the visualization.
            Default is True.

    Returns:
        Axes: The matplotlib axes object containing the visualization.
    """
    if isinstance(spectral_attributes, ImageSpectralAttributes):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, ImageSpectralAttributes) for attr in spectral_attributes)
    ):
        raise ValueError(
            "spectral_attributes must be an ImageSpectralAttributes object or a list of ImageSpectralAttributes objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    wavelengths = spectral_attributes[0].image.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Waveband", "Wavelength (nm)", "Correlation with Output")

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    spectral_band_mask = spectral_attributes[0].spectral_band_mask.cpu()
    attribution_map = torch.stack([attr.flattened_attributes.cpu() for attr in spectral_attributes])

    for idx, (band_name, segment_id) in enumerate(band_names.items()):
        current_wavelengths = wavelengths[spectral_band_mask == segment_id]
        current_attribution_map = attribution_map[:, spectral_band_mask == segment_id]

        if aggregate_results:
            ax.errorbar(
                current_wavelengths,
                current_attribution_map.mean(dim=0),
                yerr=current_attribution_map.std(dim=0),
                label=band_name,
                color=color_palette[idx],
                markersize=100,
            )
        else:
            ax.scatter(
                current_wavelengths,
                current_attribution_map.mean(dim=0),
                label=band_name,
                s=50,
                color=color_palette[idx],
            )  # Increased marker size

    return ax


def calculate_average_magnitudes(
    band_names: dict[str, int], spectral_band_mask: torch.Tensor, attribution_map: torch.Tensor
) -> torch.Tensor:
    """Calculates the average magnitudes for each segment ID in the attribution map.

    Args:
        band_names (dict[str, int]): A dictionary mapping band names to segment IDs.
        spectral_band_mask (torch.Tensor): A tensor representing the spectral band mask.
        attribution_map (torch.Tensor): A tensor representing the attribution map.

    Returns:
        list[torch.Tensor]:
            2D tensor containing the average magnitudes for each segment ID.
    """
    avg_magnitudes = []
    for segment_id in band_names.values():
        band = attribution_map[:, spectral_band_mask == segment_id]
        if band.numel() != 0:
            avg_magnitude = band.mean(dim=1)
            avg_magnitudes.append(avg_magnitude)
        else:
            avg_magnitudes.append(torch.tensor([0]))
    return torch.stack(avg_magnitudes, dim=-1).squeeze(0)


def visualize_spectral_attributes_by_magnitude(
    spectral_attributes: ImageSpectralAttributes | list[ImageSpectralAttributes],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    annotate_bars: bool = True,
    show_not_included: bool = True,
) -> Axes:
    """Visualizes the spectral attributes by magnitude.

    Args:
        spectral_attributes (ImageSpectralAttributes | list[ImageSpectralAttributes]):
            The spectral attributes to visualize.
        ax (Axes | None): The matplotlib Axes object to plot the visualization on.
            If None, a new Axes object will be created.
        color_palette (list[str] | None): The color palette to use for the visualization.
            If None, a default color palette will be used.
        annotate_bars (bool): Whether to annotate the bars with their magnitudes.
            Defaults to True.
        show_not_included (bool): Whether to show the 'not_included' band in the visualization.
            Defaults to True.

    Returns:
        Axes: The matplotlib Axes object containing the visualization.
    """
    if isinstance(spectral_attributes, ImageSpectralAttributes):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, ImageSpectralAttributes) for attr in spectral_attributes)
    ):
        raise ValueError(
            "spectral_attributes must be an ImageSpectralAttributes object or a list of ImageSpectralAttributes objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    labels = list(band_names.keys())
    wavelengths = spectral_attributes[0].image.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Magnitude", "Group", "Average Attribution Magnitude")
    ax.tick_params(axis="x", rotation=45)

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")
        labels = list(band_names.keys())

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    spectral_band_mask = spectral_attributes[0].spectral_band_mask.cpu()
    attribution_map = torch.stack([attr.flattened_attributes.cpu() for attr in spectral_attributes])
    avg_magnitudes = calculate_average_magnitudes(band_names, spectral_band_mask, attribution_map)

    if aggregate_results:
        boxplot = ax.boxplot(avg_magnitudes, labels=labels, patch_artist=True)
        for patch, color in zip(boxplot["boxes"], color_palette):
            patch.set_facecolor(color)

    else:
        bars = ax.bar(labels, avg_magnitudes, color=color_palette)
        if annotate_bars:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
    return ax
