from __future__ import annotations
from typing import Callable

import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from captum.attr import visualization as viz
from mpl_toolkits.axes_grid1 import make_axes_locatable

from meteors import (
    HSI,
    HSIAttributes,
    HSISpectralAttributes,
    HSISpatialAttributes,
)
from meteors.lime import align_band_names_with_mask
from meteors.utils.utils import aggregate_by_mask, expand_spectral_mask


def visualize_hsi(hsi: HSI | HSIAttributes, ax: Axes | None, use_mask: bool = True) -> Axes:
    """Visualizes an LIME hsi object on the given axes.

    Parameters:
        hsi (HSI | HSIAttributes): The HSI object to visualize.
        ax (matplotlib.axes.Axes | None): The axes on which the image will be plotted.
            If None, the current axes will be used.
        use_mask (bool): Whether to use the image mask if provided for the visualization.

    Returns:
        matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    if isinstance(hsi, HSIAttributes):
        hsi = hsi.hsi

    rgb = hsi.get_rgb_image(output_channel_axis=2, apply_mask=use_mask).cpu().numpy()
    ax = ax or plt.gca()
    ax.imshow(rgb)

    return ax


def visualize_spatial_attributes(  # noqa: C901
    spatial_attributes: HSISpatialAttributes, use_pyplot: bool = False
) -> tuple[Figure, Axes] | None:
    """Visualizes the spatial attributes of an hsi using Lime attribution.

    Args:
        spatial_attributes (HSISpatialAttributes):
            The spatial attributes of the hsi object to visualize.
        use_pyplot (bool, optional):
            Whether to use pyplot for visualization. Defaults to False.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None:
            If use_pyplot is False, returns the figure and axes objects.
            If use_pyplot is True, returns None.
    """
    mask_enabled = spatial_attributes.segmentation_mask is not None
    fig, ax = plt.subplots(1, 3 if mask_enabled else 2, figsize=(15, 5))
    fig.suptitle("Spatial Attributes Visualization")
    spatial_attributes = spatial_attributes.change_orientation("HWC", inplace=False)

    if mask_enabled:
        mask = spatial_attributes.segmentation_mask.cpu()

        group_names = mask.unique().tolist()
        colors = sns.color_palette("hsv", len(group_names))
        color_map = dict(zip(group_names, colors))

        for unique in group_names:
            segment_indices = torch.argwhere(mask == unique)

            y_center, x_center = segment_indices.numpy().mean(axis=0).astype(int)
            ax[1].text(x_center, y_center, str(unique), color=color_map[unique], fontsize=8, ha="center", va="center")
            ax[2].text(x_center, y_center, str(unique), color=color_map[unique], fontsize=8, ha="center", va="center")

        ax[2].imshow(mask.numpy() / mask.max(), cmap="gray")
        ax[2].set_title("Mask")
        ax[2].grid(False)
        ax[2].axis("off")

    ax[0].imshow(spatial_attributes.hsi.get_rgb_image(output_channel_axis=2).cpu())
    ax[0].set_title("Original image")
    ax[0].grid(False)
    ax[0].axis("off")

    attrs = spatial_attributes.attributes.cpu().numpy()
    if np.all(attrs == 0):
        logger.warning("All spatial attributes are zero.")
        cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
        heat_map = ax[1].imshow(attrs.sum(axis=-1), cmap=cmap, vmin=-1, vmax=1)

        axis_separator = make_axes_locatable(ax[1])
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
    else:
        viz.visualize_image_attr(
            attrs,
            method="heat_map",
            sign="all",
            plt_fig_axis=(fig, ax[1]),
            show_colorbar=True,
            use_pyplot=False,
        )
    ax[1].set_title("Attribution Map")
    ax[1].axis("off")

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover
    else:
        return fig, ax


def visualize_spectral_attributes(
    spectral_attributes: HSISpectralAttributes | list[HSISpectralAttributes],
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
) -> tuple[Figure, Axes] | None:
    """Visualizes the spectral attributes of an hsi object or a list of hsi objects.

    Args:
        spectral_attributes (HSISpectralAttributes | list[HSISpectralAttributes]):
            The spectral attributes of the imhsiage object to visualize.
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
    agg = True if isinstance(spectral_attributes, list) else False
    band_names = spectral_attributes[0].band_names if agg else spectral_attributes.band_names  # type: ignore

    color_palette = color_palette or sns.color_palette("hsv", len(band_names.keys()))

    fig, ax = plt.subplots(1, 3 if agg else 2, figsize=(15, 5))
    fig.suptitle("Spectral Attributes Visualization")

    visualize_spectral_attributes_by_waveband(
        spectral_attributes,
        ax[0],
        color_palette=color_palette,
        show_not_included=show_not_included,
        show_legend=False,
    )

    visualize_spectral_attributes_by_magnitude(
        spectral_attributes,
        ax[1],
        color_palette=color_palette,
        show_not_included=show_not_included,
    )

    if agg:
        scores = [attr.score for attr in spectral_attributes]  # type: ignore
        mean_score = sum(scores) / len(scores)
        ax[2].hist(scores, bins=50, color="steelblue", alpha=0.7)
        ax[2].axvline(mean_score, color="darkred", linestyle="dashed")

        ax[2].set_title("Distribution of Score Values")
        ax[2].set_xlabel("Score")
        ax[2].set_ylabel("Frequency")

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover
    else:
        return fig, ax


def validate_consistent_band_and_wavelengths(
    band_names: dict[str, int], wavelengths: torch.Tensor, spectral_attributes: list[HSISpectralAttributes]
) -> None:
    """Validates that all spectral attributes have consistent band names and wavelengths.

    Args:
        band_names (dict[str, int]): A dictionary mapping band names to their indices.
        wavelengths (torch.Tensor): A tensor containing the wavelengths of the hsi.
        spectral_attributes (list[HSISpectralAttributes]): A list of spectral attributes.

    Raises:
        ValueError: If the band names or wavelengths of any spectral attribute are inconsistent.
    """
    for attr in spectral_attributes:
        if band_names != attr.band_names:
            raise ValueError("Band names are inconsistent among spectral attributes.")
        if not torch.equal(wavelengths, attr.hsi.wavelengths):
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
    spectral_attributes: HSISpectralAttributes | list[HSISpectralAttributes],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    show_legend: bool = True,
) -> Axes:
    """Visualizes spectral attributes by waveband.

    Args:
        spectral_attributes (HSISpectralAttributes | list[HSISpectralAttributes]):
            The spectral attributes to visualize.
        ax (Axes | None): The matplotlib axes to plot the visualization on.
            If None, a new axes will be created.
        color_palette (list[str] | None): The color palette to use for plotting.
            If None, a default color palette will be used.
        show_not_included (bool): Whether to show the "not_included" band in the visualization.
            Default is True.
        show_legend (bool): Whether to show the legend in the visualization.

    Returns:
        Axes: The matplotlib axes object containing the visualization.
    """
    if isinstance(spectral_attributes, HSISpectralAttributes):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, HSISpectralAttributes) for attr in spectral_attributes)
    ):
        raise ValueError(
            "spectral_attributes must be an HSISpectralAttributes object or a list of HSISpectralAttributes objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    wavelengths = spectral_attributes[0].hsi.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Waveband", "Wavelength (nm)", "Correlation with Output")

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    band_mask = spectral_attributes[0].band_mask.cpu()
    attribution_map = torch.stack([attr.flattened_attributes.cpu() for attr in spectral_attributes])

    for idx, (band_name, segment_id) in enumerate(band_names.items()):
        current_wavelengths = wavelengths[band_mask == segment_id]
        current_attribution_map = attribution_map[:, band_mask == segment_id]

        current_mean = current_attribution_map.numpy().mean(axis=0)
        if aggregate_results:
            lolims = current_attribution_map.numpy().min(axis=0)
            uplims = current_attribution_map.numpy().max(axis=0)

            ax.errorbar(
                current_wavelengths,
                current_mean,
                yerr=[current_mean - lolims, uplims - current_mean],
                label=band_name,
                color=color_palette[idx],
                linestyle="--",
                marker="o",
                markersize=5,
            )
        else:
            ax.scatter(
                current_wavelengths,
                current_mean,
                label=band_name,
                color=color_palette[idx],
            )

    if show_legend:
        ax.legend(title="SuperBand")

    return ax


def calculate_average_magnitudes(
    band_names: dict[str, int], band_mask: torch.Tensor, attribution_map: torch.Tensor
) -> torch.Tensor:
    """Calculates the average magnitudes for each segment ID in the attribution map.

    Args:
        band_names (dict[str, int]): A dictionary mapping band names to segment IDs.
        band_mask (torch.Tensor): A tensor representing the spectral band mask.
        attribution_map (torch.Tensor): A tensor representing the attribution map.

    Returns:
        list[torch.Tensor]:
            2D tensor containing the average magnitudes for each segment ID.
    """
    avg_magnitudes = []
    for segment_id in band_names.values():
        band = attribution_map[:, band_mask == segment_id]
        if band.numel() != 0:
            avg_magnitude = band.mean(dim=1)
            avg_magnitudes.append(avg_magnitude)
        else:
            avg_magnitudes.append(torch.tensor([0]))
    return torch.stack(avg_magnitudes, dim=-1).squeeze(0)


def visualize_spectral_attributes_by_magnitude(
    spectral_attributes: HSISpectralAttributes | list[HSISpectralAttributes],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    annotate_bars: bool = True,
    show_not_included: bool = True,
) -> Axes:
    """Visualizes the spectral attributes by magnitude.

    Args:
        spectral_attributes (HSISpectralAttributes | list[HSISpectralAttributes]):
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
    if isinstance(spectral_attributes, HSISpectralAttributes):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, HSISpectralAttributes) for attr in spectral_attributes)
    ):
        raise ValueError(
            "spectral_attributes must be an HSISpectralAttributes object or a list of HSISpectralAttributes objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    labels = list(band_names.keys())
    wavelengths = spectral_attributes[0].hsi.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Magnitude", "Group", "Average Attribution Magnitude")
    ax.tick_params(axis="x", rotation=45)

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")
        labels = list(band_names.keys())

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    band_mask = spectral_attributes[0].band_mask.cpu()
    attribution_map = torch.stack([attr.flattened_attributes.cpu() for attr in spectral_attributes])
    avg_magnitudes = calculate_average_magnitudes(band_names, band_mask, attribution_map)

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


def visualize_spatial_aggregated_attributes(
    attributes: HSIAttributes,
    aggregated_mask: torch.Tensor | np.ndarray,
    use_pyplot: bool = False,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> tuple[Figure, Axes] | None:
    """Visualizes the spatial attributes of an hsi object aggregated by a custom mask.

    Args:
        attributes (HSIAttributes): The spatial attributes of the hsi object to visualize.
        aggregated_mask (torch.Tensor | np.ndarray): The mask used to aggregate the spatial attributes.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If False, returns the figure and axes objects. Defaults to False.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.

    Raises:
        ValueError: If the shape of the aggregated mask does not match the shape of the spatial attributes.

    Returns:
        tuple[Figure, Axes] | None: If use_pyplot is False, returns the figure and axes objects.
    """
    if isinstance(aggregated_mask, np.ndarray):
        aggregated_mask = torch.from_numpy(aggregated_mask)

    if aggregated_mask.shape != attributes.hsi.image.shape:
        aggregated_mask = aggregated_mask.expand_as(attributes.attributes)

    new_attrs = aggregate_by_mask(attributes.attributes, aggregated_mask, aggregate_func)

    new_spatial_attributes = HSISpatialAttributes(
        hsi=attributes.hsi,
        attributes=new_attrs,
        mask=aggregated_mask,
        score=attributes.score,
    )

    fig, ax = visualize_spatial_attributes(new_spatial_attributes, use_pyplot=False)  # type: ignore
    fig.suptitle("Spatial Attributes Visualization Aggregated")

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover
    else:
        return fig, ax


def visualize_spectral_aggregated_attributes(
    attributes: HSIAttributes | list[HSIAttributes],
    band_names: dict[str, int],
    band_mask: torch.Tensor | np.ndarray,
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> tuple[Figure, Axes] | None:
    """Visualizes the spectral attributes of an hsi object aggregated by a custom band mask.

    Args:
        attributes (HSIAttributes | list[HSIAttributes]): The spectral attributes of the hsi object to visualize.
        band_names (dict[str, int]): A dictionary mapping band names to their indices.
        band_mask (torch.Tensor | np.ndarray): The mask used to aggregate the spectral attributes.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If False, returns the figure and axes objects. Defaults to False.
        color_palette (list[str] | None, optional): The color palette to use for visualizing different spectral bands.
            If None, a default color palette is used. Defaults to None.
        show_not_included (bool, optional): If True, includes the spectral bands that are not included in the visualization.
            If False, only includes the spectral bands that are included in the visualization. Defaults to True.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.

    Raises:
        ValueError: If the shape of the band mask does not match the shape of the spectral attributes.

    Returns:
        tuple[Figure, Axes] | None: If use_pyplot is False, returns the figure and axes objects.
    """
    attributes_example = attributes if isinstance(attributes, HSIAttributes) else attributes[0]
    if isinstance(band_mask, np.ndarray):
        band_mask = torch.from_numpy(band_mask)

    if band_mask.shape != attributes_example.hsi.image.shape:
        band_mask = expand_spectral_mask(attributes_example.hsi, band_mask, repeat_dimensions=True)

    band_names = align_band_names_with_mask(band_names, band_mask)

    new_attrs = aggregate_by_mask(attributes_example.attributes, band_mask, aggregate_func)

    new_spectral_attributes: HSISpectralAttributes | list[HSISpectralAttributes]
    if isinstance(attributes, HSIAttributes):
        new_spectral_attributes = HSISpectralAttributes(
            hsi=attributes.hsi,
            attributes=new_attrs,
            mask=band_mask,
            band_names=band_names,
            score=attributes.score,
        )
    else:
        new_spectral_attributes = [
            HSISpectralAttributes(
                hsi=attr.hsi,
                attributes=new_attrs,
                mask=band_mask,
                band_names=band_names,
                score=attr.score,
            )
            for attr in attributes
        ]

    fig, ax = visualize_spectral_attributes(  # type: ignore
        new_spectral_attributes, use_pyplot=False, color_palette=color_palette, show_not_included=show_not_included
    )

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover
    else:
        return fig, ax


def visualize_aggregated_attributes(
    attributes: HSIAttributes | list[HSIAttributes],
    mask: torch.Tensor | np.ndarray,
    band_names: dict[str, int] | None = None,
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> tuple[Figure, Axes] | None:
    """Visualizes the aggregated attributes of an hsi object.

    Args:
        attributes (HSIAttributes | list[HSIAttributes]): The attributes of the hsi object to visualize.
        mask (torch.Tensor | np.ndarray): The mask used to aggregate the attributes.
        band_names (dict[str, int] | None, optional): A dictionary mapping band names to their indices.
            If None, the visualization will be spatially aggregated. Defaults to None.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If False, returns the figure and axes objects. Defaults to False.
        color_palette (list[str] | None, optional): The color palette to use for visualizing different spectral bands.
            If None, a default color palette is used. Defaults to None.
        show_not_included (bool, optional): If True, includes the spectral bands that are not included in the visualization.
            If False, only includes the spectral bands that are included in the visualization. Defaults to True.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.

    Raises:
        ValueError: If the shape of the mask does not match the shape of the attributes.
        AssertionError: If band_names is None and attributes is a list of HSIAttributes objects.

    Returns:
        tuple[Figure, Axes] | None: If use_pyplot is False, returns the figure and axes objects.
    """
    agg = False if isinstance(attributes, HSIAttributes) else True
    if band_names is None:
        logger.info("Band names not provided. Using Spatial Analysis.")
        assert not agg, "In Spatial Analysis, attributes must be a single HSIAttributes object."
        return visualize_spatial_aggregated_attributes(attributes, mask, use_pyplot, aggregate_func)  # type: ignore
    else:
        logger.info("Band names provided. Using Spectral Analysis.")
        return visualize_spectral_aggregated_attributes(
            attributes, band_names, mask, use_pyplot, color_palette, show_not_included, aggregate_func
        )
