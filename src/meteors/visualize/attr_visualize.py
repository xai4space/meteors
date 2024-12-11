from __future__ import annotations
from typing import Callable

import torch
import warnings
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from captum.attr import visualization as viz
from mpl_toolkits.axes_grid1 import make_axes_locatable

from meteors.attr import HSIAttributesSpatial, HSIAttributesSpectral, HSIAttributes
from meteors.utils import expand_spectral_mask, aggregate_by_mask

from meteors.attr.attributes import align_band_names_with_mask


def visualize_attributes(
    image_attributes: HSIAttributes, ax: Axes | None = None, use_pyplot: bool = False, title: str | None = None
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the attributes of an image on the given axes.

    Parameters:
        image_attributes (HSIAttributes): The image attributes to be visualized.
        ax (Axes | None): The axes to visualize the image on. If None, creates a new figure and axes.
        use_pyplot (bool): If True, uses pyplot to display the image. If False, returns the figure and axes objects.
            if ax is not None, use_pyplot is ignored.
        title (str): The title of the chart. If it equals to None, will be replaced with title "HSI Attributes of {method used for explanations}". Defaults to None

    Returns:
        matplotlib.figure.Figure | matplotlib.axes.Axes | None: The figure and axes objects.
            If use_pyplot is False and ax is None, returns the figure and axes objects.
            If use_pyplot is True and ax is None, returns None, and displays the image using pyplot.
            if ax is not None, returns the axes object.
            If all the attributions are zero, returns None.

    Raises:
        ValueError: If the axes have less than 2 rows and 2 columns
        ValueError: If the axes object is not a list of axes objects
    """
    if image_attributes.hsi.orientation != ("H", "W", "C"):
        logger.info(
            f"The orientation of the image is not (H, W, C): {image_attributes.hsi.orientation}. "
            f"Changing it to (H, W, C) for visualization."
        )
        rotated_attributes_dataclass = image_attributes.change_orientation("HWC", inplace=False)
    else:
        rotated_attributes_dataclass = image_attributes

    rotated_attributes = rotated_attributes_dataclass.attributes.detach().cpu().numpy()
    if np.all(rotated_attributes == 0):
        warnings.warn("All the attributions are zero. There is nothing to visualize.")
        return None

    used_ax = True
    if ax is None:
        used_ax = False
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))

    if not hasattr(ax, "shape"):
        raise ValueError("Provided ax parameter is only one axes object, but it should be a list of axes objects")
    elif len(ax.shape) != 2 or ax.shape[0] < 2 or ax.shape[1] < 2:
        raise ValueError("The axes should have at least 2 rows and 2 columns.")
    else:
        fig = ax.ravel()[0].get_figure()

    ax[0, 0].set_title("Attribution Heatmap")
    ax[0, 0].grid(False)
    ax[0, 0].axis("off")

    if title is None:
        title = f"HSI Attributes of: {rotated_attributes_dataclass.attribution_method}"

    fig.suptitle(title)

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

    attr_all = rotated_attributes.sum(axis=(0, 1))
    ax[1, 0].scatter(rotated_attributes_dataclass.hsi.wavelengths, attr_all, c="r")
    ax[1, 0].set_title("Spectral Attribution")
    ax[1, 0].set_xlabel("Wavelength")
    ax[1, 0].set_ylabel("Attribution")
    ax[1, 0].grid(True)

    attr_abs = np.abs(rotated_attributes).sum(axis=(0, 1))
    ax[1, 1].scatter(rotated_attributes_dataclass.hsi.wavelengths, attr_abs, c="b")
    ax[1, 1].set_title("Spectral Attribution Absolute Values")
    ax[1, 1].set_xlabel("Wavelength")
    ax[1, 1].set_ylabel("Attribution Absolute Value")
    ax[1, 1].grid(True)

    plt.tight_layout()

    if used_ax:
        return ax

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover

    return fig, ax


def _merge_band_names_segments(band_names: dict[str | tuple[str, ...], int]) -> dict[str, int]:
    """Merges the band names with multiple segments into a single band name.

    Args:
        band_names (dict[str | tuple[str, ...], int]): A dictionary mapping band names to segment IDs.

    Returns:
        dict[str, int]: A dictionary mapping band names to segment IDs.
    """

    # simplify bands with multiple segments
    segments_to_be_updated = []

    for label in band_names.keys():
        if isinstance(label, tuple) or isinstance(label, list):
            new_key = ",".join(label)
            segments_to_be_updated.append((label, new_key))

    # update the dict
    for old_key, new_key in segments_to_be_updated:
        band_names[new_key] = band_names.pop(old_key)

    return band_names  # type: ignore


def visualize_spatial_attributes(
    spatial_attributes: HSIAttributesSpatial,
    ax: Axes | None = None,
    use_pyplot: bool = False,
    title: str | None = None,
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the spatial attributes of an hsi using Lime attribution.

    Args:
        spatial_attributes (HSIAttributesSpatial):
            The spatial attributes of the image object to visualize.
        ax (Axes | None, optional):
            The axes object to plot the visualization on. If None, a new axes will be created.
        use_pyplot (bool, optional):
            Whether to use pyplot for visualization. Defaults to False.
        title (str, optional): The title of the plot. If None will be set to "Spatial Attributes Visualization", Defaults to None.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | matplotlib.axes.Axes | None:
            If ax is not None, returns the axes object.
            If use_pyplot is True, returns None.
            If use_pyplot is False, returns the figure and axes objects.

    Raises:
        ValueError: If the axes have less 3 rows or 3 columns with mask
        ValueError: If the axes have less 2 rows or 2 columns without mask
        ValueError: If the axes is a 2d object not 1d
        ValueError: If the axes object is not a list of axes objects
    """
    mask_enabled = True if spatial_attributes.mask is not None else False
    use_ax = True
    if ax is None:
        use_ax = False
        fig, ax = plt.subplots(1, 3 if mask_enabled else 2, figsize=(15, 5) if mask_enabled else (10, 4))

    if not hasattr(ax, "shape"):
        raise ValueError("Provided as is one axes object, but it should be a list of axes objects")
    elif len(ax.shape) == 1:
        if mask_enabled and ax.shape[0] < 3:
            raise ValueError("The axes should have at least 3 rows or 3 columns")
        elif not mask_enabled and ax.shape[0] < 2:
            raise ValueError("The axes should have at least 2 rows or 2 columns")
        fig = ax[0].get_figure()
    elif len(ax.shape) != 1:
        raise ValueError("The axes are 2d please provide 1d axes")

    if title is None:
        title = "Spatial Attributes Visualization"

    fig.suptitle(title)

    spatial_attributes = spatial_attributes.change_orientation("HWC", inplace=False)

    if mask_enabled:
        mask = spatial_attributes.segmentation_mask.detach().cpu()

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

    ax[0].imshow(spatial_attributes.hsi.get_rgb_image(output_channel_axis=2).detach().cpu())
    ax[0].set_title("Original image")
    ax[0].grid(False)
    ax[0].axis("off")

    attrs = spatial_attributes.flattened_attributes.detach().cpu().unsqueeze(-1).numpy()
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

    if use_ax:
        return ax

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover
    else:
        return fig, ax


def visualize_spectral_attributes(
    spectral_attributes: HSIAttributesSpectral | list[HSIAttributesSpectral],
    ax: Axes | None = None,
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the spectral attributes of an hsi object or a list of hsi objects.

    Args:
        spectral_attributes (HSIAttributesSpectral | list[HSIAttributesSpectral]):
            The spectral attributes of the image object to visualize.
        ax (Axes | None, optional):
            The axes object to plot the visualization on. If None, a new axes will be created.
        use_pyplot (bool, optional):
            If ax is not None, use_pyplot is ignored.
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
        title (str, optional): The title of the plot. If None, the title will be set to "Spectral Attributes Visualization". Defaults to None.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | matplotlib.axes.Axes | None:
            If ax is not None, returns the axes object.
            If use_pyplot is True, returns None.
            If use_pyplot is False, returns the figure and axes objects.

    Raises:
        ValueError: If ax is provided as a single axes object and not a list of axes objects.
        ValueError: If agg is True and the axes have less than 3 rows or 3 columns.
        ValueError: If agg is False and the axes have less than 2 rows or 2 columns.
    """
    agg = True if isinstance(spectral_attributes, list) else False
    band_names = spectral_attributes[0].band_names if agg else spectral_attributes.band_names  # type: ignore

    if title is None:
        title = "Spectral Attributes Visualization"

    color_palette = color_palette or sns.color_palette("hsv", len(band_names.keys()))

    use_ax = True
    if ax is None:
        use_ax = False
        fig, ax = plt.subplots(1, 3 if agg else 2, figsize=(15, 5))
    else:
        if isinstance(ax, np.ndarray):
            fig = ax.ravel()[0].get_figure()
        else:
            fig = ax.get_figure()
    fig.suptitle(title)

    if not hasattr(ax, "shape"):
        raise ValueError("Provided as is one axes object, but it should be a list of axes objects")
    if agg and (len(ax.shape) != 1 or ax.shape[0] < 3):
        raise ValueError("The axes should have at least 3 rows or 3 columns if agg is True")
    if not agg and (len(ax.shape) != 1 or ax.shape[0] < 2):
        raise ValueError("The axes should have at least 2 rows or 2 columns if agg is False")

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
        mean_score = sum(scores) / len(scores)  # type: ignore
        ax[2].hist(scores, bins=50, color="steelblue", alpha=0.7)
        ax[2].axvline(mean_score, color="darkred", linestyle="dashed")

        ax[2].set_title("Distribution of Score Values")
        ax[2].set_xlabel("Score")
        ax[2].set_ylabel("Frequency")

    if use_ax:
        return ax

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover

    return fig, ax


def validate_consistent_band_and_wavelengths(
    band_names: dict[str | tuple[str, ...], int],
    wavelengths: torch.Tensor,
    spectral_attributes: list[HSIAttributesSpectral],
) -> None:
    """Validates that all spectral attributes have consistent band names and wavelengths.

    Args:
        band_names (dict[str | tuple[str, ...], int]): A dictionary mapping band names to their indices.
        wavelengths (torch.Tensor): A tensor containing the wavelengths of the hsi.
        spectral_attributes (list[HSIAttributesSpectral]): A list of spectral attributes.

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
    spectral_attributes: HSIAttributesSpectral | list[HSIAttributesSpectral],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    show_legend: bool = True,
) -> Axes:
    """Visualizes spectral attributes by waveband.

    Args:
        spectral_attributes (HSIAttributesSpectral | list[HSIAttributesSpectral]):
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
    Raises:
        TypeError: If the spectral attributes are not an HSIAttributesSpectral object or a list of HSIAttributesSpectral objects.
    """
    if isinstance(spectral_attributes, HSIAttributesSpectral):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, HSIAttributesSpectral) for attr in spectral_attributes)
    ):
        raise TypeError(
            "spectral_attributes parameter must be an HSIAttributesSpectral object or a list of HSIAttributesSpectral objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    wavelengths = spectral_attributes[0].hsi.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Waveband", "Wavelength (nm)", "Correlation with Output")

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")

    band_names = _merge_band_names_segments(band_names)  # type: ignore

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    band_mask = spectral_attributes[0].band_mask.detach().cpu()
    attribution_map = torch.stack([attr.flattened_attributes.detach().cpu() for attr in spectral_attributes])

    for idx, (band_name, segment_id) in enumerate(band_names.items()):
        current_wavelengths = wavelengths[band_mask == segment_id]
        current_attribution_map = attribution_map[:, band_mask == segment_id]

        current_mean = current_attribution_map.numpy().mean(axis=0)
        if aggregate_results:
            lolims = current_attribution_map.numpy().min(axis=0)
            uplims = current_attribution_map.numpy().max(axis=0)

            ax.errorbar(
                current_wavelengths.numpy(),
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
                current_wavelengths.numpy(),
                current_mean,
                label=band_name,
                color=color_palette[idx],
            )

    if show_legend:
        ax.legend(title="SuperBand")

    return ax


def calculate_average_magnitudes(
    band_names: dict[str | tuple[str, ...], int], band_mask: torch.Tensor, attribution_map: torch.Tensor
) -> torch.Tensor:
    """Calculates the average magnitudes for each segment ID in the attribution map.

    Args:
        band_names (dict[str | tuple[str, ...], int]): A dictionary mapping band names to segment IDs.
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
    spectral_attributes: HSIAttributesSpectral | list[HSIAttributesSpectral],
    ax: Axes | None,
    color_palette: list[str] | None = None,
    annotate_bars: bool = True,
    show_not_included: bool = True,
) -> Axes:
    """Visualizes the spectral attributes by magnitude.

    Args:
        spectral_attributes (HSIAttributesSpectral | list[HSIAttributesSpectral]):
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
    Raises:
        TypeError: If the spectral attributes are not an HSIAttributesSpectral object or a list of HSIAttributesSpectral objects.
    """
    if isinstance(spectral_attributes, HSIAttributesSpectral):
        spectral_attributes = [spectral_attributes]
    if not (
        isinstance(spectral_attributes, list)
        and all(isinstance(attr, HSIAttributesSpectral) for attr in spectral_attributes)
    ):
        raise TypeError(
            "spectral_attributes parameter must be an HSIAttributesSpectral object or a list of HSIAttributesSpectral objects."
        )

    aggregate_results = False if len(spectral_attributes) == 1 else True
    band_names = dict(spectral_attributes[0].band_names)
    wavelengths = spectral_attributes[0].hsi.wavelengths
    validate_consistent_band_and_wavelengths(band_names, wavelengths, spectral_attributes)

    ax = setup_visualization(ax, "Attributions by Magnitude", "Group", "Average Attribution Magnitude")
    ax.tick_params(axis="x", rotation=45)

    band_names = _merge_band_names_segments(band_names)  # type: ignore
    labels = list(band_names.keys())

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")
        labels = list(band_names.keys())

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    band_mask = spectral_attributes[0].band_mask.detach().cpu()
    attribution_map = torch.stack([attr.flattened_attributes.detach().cpu() for attr in spectral_attributes])
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
    ax: Axes | None = None,
    use_pyplot: bool = False,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    title: str | None = None,
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the spatial attributes of an hsi object aggregated by a custom mask.

    Args:
        attributes (HSIAttributes): The spatial attributes of the hsi object to visualize.
        aggregated_mask (torch.Tensor | np.ndarray): The mask used to aggregate the spatial attributes.
        ax (Axes | None, optional): The axes object to plot the visualization on. If None, a new axes will be created.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If ax is not None, use_pyplot is ignored.
            If False, returns the figure and axes objects. Defaults to False.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.
        title (str, optional): The title of the plot. If None will be replaced with "Spatial Attributes Visualization Aggregated". Defaults to None.

    Raises:
        ShapeMismatchError: If the shape of the aggregated mask does not match the shape of the spatial attributes.

    Returns:
        tuple[Figure, Axes] | Axes | None: If ax is not None, returns the axes object.
            If use_pyplot is True, returns None. If use_pyplot is False, returns the figure and axes objects.
    """
    if isinstance(aggregated_mask, np.ndarray):
        aggregated_mask = torch.from_numpy(aggregated_mask)

    if aggregated_mask.shape != attributes.hsi.image.shape:
        aggregated_mask = aggregated_mask.expand_as(attributes.attributes)

    if title is None:
        title = "Spatial Attributes Visualization Aggregated"

    new_attrs = aggregate_by_mask(attributes.attributes, aggregated_mask, aggregate_func)

    new_spatial_attributes = HSIAttributesSpatial(
        hsi=attributes.hsi,
        attributes=new_attrs,
        mask=aggregated_mask,
        score=attributes.score,
    )

    out = visualize_spatial_attributes(new_spatial_attributes, ax=ax, use_pyplot=False, title=title)  # type: ignore
    if ax is not None:
        return out

    fig, ax = out  # type: ignore
    fig.suptitle(title)

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover

    return fig, ax


def visualize_spectral_aggregated_attributes(
    attributes: HSIAttributes | list[HSIAttributes],
    band_names: dict[str | tuple[str, ...], int],
    band_mask: torch.Tensor | np.ndarray,
    ax: Axes | None = None,
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    title: str | None = None,
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the spectral attributes of an hsi object aggregated by a custom band mask.

    Args:
        attributes (HSIAttributes | list[HSIAttributes]): The spectral attributes of the hsi object to visualize.
        band_names (dict[str | tuple[str, ...], int]): A dictionary mapping band names to their indices.
        band_mask (torch.Tensor | np.ndarray): The mask used to aggregate the spectral attributes.
        ax (Axes | None, optional): The axes object to plot the visualization on. If None, a new axes will be created.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If ax is not None, use_pyplot is ignored. If False, returns the figure and axes objects. Defaults to False.
        color_palette (list[str] | None, optional): The color palette to use for visualizing different spectral bands.
            If None, a default color palette is used. Defaults to None.
        show_not_included (bool, optional): If True, includes the spectral bands that are not included in the visualization.
            If False, only includes the spectral bands that are included in the visualization. Defaults to True.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.
        title (str, optional): The title of the plot. If None, "Spectral Attributes Visualization" will be used. Defaults to None.

    Raises:
        ShapeMismatchError: If the shape of the band mask does not match the shape of the spectral attributes.

    Returns:
        tuple[Figure, Axes] | Axes | None: If ax is not None, returns the axes object.
            If use_pyplot is True, returns None. If use_pyplot is False, returns the figure and axes objects
    """
    attributes_example = attributes if isinstance(attributes, HSIAttributes) else attributes[0]
    if isinstance(band_mask, np.ndarray):
        band_mask = torch.from_numpy(band_mask)

    if band_mask.shape != attributes_example.hsi.image.shape:
        band_mask = expand_spectral_mask(attributes_example.hsi, band_mask, repeat_dimensions=True)

    band_names = align_band_names_with_mask(band_names, band_mask)

    new_attrs = aggregate_by_mask(attributes_example.attributes, band_mask, aggregate_func)

    new_spectral_attributes: HSIAttributesSpectral | list[HSIAttributesSpectral]
    if isinstance(attributes, HSIAttributes):
        new_spectral_attributes = HSIAttributesSpectral(
            hsi=attributes.hsi,
            attributes=new_attrs,
            mask=band_mask,
            band_names=band_names,
            score=attributes.score,
        )
    else:
        new_spectral_attributes = [
            HSIAttributesSpectral(
                hsi=attr.hsi,
                attributes=new_attrs,
                mask=band_mask,
                band_names=band_names,
                score=attr.score,
            )
            for attr in attributes
        ]

    out = visualize_spectral_attributes(
        new_spectral_attributes,
        ax=ax,
        use_pyplot=False,
        color_palette=color_palette,
        show_not_included=show_not_included,
        title=title,
    )  # type: ignore
    if ax is not None:
        return out

    if use_pyplot:
        plt.show()  # pragma: no cover
        return None  # pragma: no cover

    return out


def visualize_aggregated_attributes(
    attributes: HSIAttributes | list[HSIAttributes],
    mask: torch.Tensor | np.ndarray,
    band_names: dict[str | tuple[str, ...], int] | None = None,
    ax: Axes | None = None,
    use_pyplot: bool = False,
    color_palette: list[str] | None = None,
    show_not_included: bool = True,
    aggregate_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    title: str | None = None,
) -> tuple[Figure, Axes] | Axes | None:
    """Visualizes the aggregated attributes of an hsi object.

    Args:
        attributes (HSIAttributes | list[HSIAttributes]): The attributes of the hsi object to visualize.
        mask (torch.Tensor | np.ndarray): The mask used to aggregate the attributes.
        band_names (dict[str | tuple[str, ...], int] | None, optional): A dictionary mapping band names to their indices.
            If None, the visualization will be spatially aggregated. Defaults to None.
        ax (Axes | None, optional): The axes object to plot the visualization on. If None, a new axes will be created.
        use_pyplot (bool, optional): If True, displays the visualization using pyplot.
            If ax is not None, use_pyplot is ignored. If False, returns the figure and axes objects. Defaults to False.
        color_palette (list[str] | None, optional): The color palette to use for visualizing different spectral bands.
            If None, a default color palette is used. Defaults to None.
        show_not_included (bool, optional): If True, includes the spectral bands that are not included in the visualization.
            If False, only includes the spectral bands that are included in the visualization. Defaults to True.
        aggregate_func (Callable[[torch.Tensor], torch.Tensor], optional): The aggregation function to be applied.
            The function should take a tensor as input and return a tensor as output.
            We recommend using torch functions. Defaults to torch.mean.
        title (str, optional): The title of the plot. If None the default of the underlying figure will be used. Defaults to None.

    Raises:
        ValueError: If the shape of the mask does not match the shape of the attributes.
        AssertionError: If band_names is None and attributes is a list of HSIAttributes objects.

    Returns:
        tuple[Figure, Axes] | Axes | None: If ax is not None, returns the axes object.
            If use_pyplot is True, returns None. If use_pyplot is False, returns the figure and axes objects.
    """
    agg = False if isinstance(attributes, HSIAttributes) else True
    if band_names is None:
        logger.info("Band names not provided. Using Spatial Analysis.")
        assert not agg, "In Spatial Analysis, attributes must be a single HSIAttributes object."
        return visualize_spatial_aggregated_attributes(attributes, mask, ax, use_pyplot, aggregate_func, title=title)  # type: ignore
    else:
        logger.info("Band names provided. Using Spectral Analysis.")
        return visualize_spectral_aggregated_attributes(
            attributes, band_names, mask, ax, use_pyplot, color_palette, show_not_included, aggregate_func, title=title
        )


def visualize_bands_spatial_attributes(
    attributes: HSIAttributes,
    spectral_indexes: list[int] | None = None,
    spectral_wavelengths: list[float] | None = None,
    ax: Axes | None = None,
    use_pyplot: bool = False,
) -> tuple[Figure, Axes] | Axes | None:
    """
    Visualizes the spatial attributes for specific spectral bands, identified either by their indexes
    or wavelengths.

    Args:
        attributes (HSIAttributes): The HSI attributes object containing the spatial attributes to visualize.
        spectral_indexes (list[int] | None, optional): List of spectral band/wavelengths indexes to visualize.
            Takes precedence over spectral_wavelengths if both are provided. Defaults to None.
        spectral_wavelengths (list[float] | None, optional): List of band/wavelengths values to visualize.
            Only used if spectral_indexes is None. Wavelengths must match those in the HSI object.
            Defaults to None.
        ax (Axes | None, optional): Matplotlib axes for plotting. If None, creates new axes.
            Defaults to None.
        use_pyplot (bool, optional): Whether to display the plot using pyplot.
            Ignored if ax is provided. Defaults to False.

    Returns:
        tuple[Figure, Axes] | Axes | None:
            - If ax is provided: returns the modified axes
            - If ax is None and use_pyplot is False: returns (figure, axes)
            - If ax is None and use_pyplot is True: returns None and displays the plot

    Raises:
        ValueError: If some of the provided wavelengths are not present in the HSI object.
    """
    if spectral_indexes is None and spectral_wavelengths is None:
        spectral_indexes = list(range(attributes.hsi.image.shape[attributes.hsi.spectral_axis]))
    elif spectral_indexes is None and spectral_wavelengths is not None:
        try:
            spectral_indexes = [
                (attributes.hsi.wavelengths == w).nonzero(as_tuple=True)[0].item() for w in spectral_wavelengths
            ]
        except RuntimeError as e:
            if "a Tensor with 0 elements cannot be converted to Scalar" in str(e):
                raise ValueError("Some of the provided wavelengths are not present in the HSI object.") from e
            else:
                raise e

    spatial_attributes = attributes.attributes.index_select(
        dim=attributes.hsi.spectral_axis, index=torch.tensor(spectral_indexes)
    ).mean(dim=attributes.hsi.spectral_axis, keepdim=True)
    spatial_attributes_expanded = spatial_attributes.expand_as(attributes.hsi.image)
    assert spatial_attributes_expanded.shape == attributes.hsi.image.shape
    assert torch.equal(
        spatial_attributes_expanded.index_select(dim=attributes.hsi.spectral_axis, index=torch.tensor(0)),
        spatial_attributes,
    )

    spatial_attributes_object = HSIAttributesSpatial(
        hsi=attributes.hsi,
        attributes=spatial_attributes_expanded,
        mask=None,
        score=attributes.score,
        attribution_method=attributes.attribution_method,
    )

    return visualize_spatial_attributes(spatial_attributes_object, ax=ax, use_pyplot=use_pyplot)
