from typing import List

import matplotlib.figure

from meteors import (
    Image,
    ImageAttributes,
    ImageSpectralAttributes,
    ImageSpatialAttributes,
)
import torch
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

import seaborn as sns


def visualise_image(
    image: Image | ImageAttributes, ax: matplotlib.axes.Axes | None
) -> None | matplotlib.axes.Axes:
    if isinstance(image, ImageAttributes):
        image = image.image
    rgb = image.get_rgb_image(output_band_index=2)

    if ax is None:
        ax = plt.gca()

    ax.imshow(rgb)

    return ax


def visualise_spatial_attributes(
    spatial_attributes: ImageSpatialAttributes, use_pyplot=False
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Lime attribution")
    ax[0].imshow(spatial_attributes.image.get_rgb_image(output_band_index=2).cpu())
    ax[0].set_title("Original image")

    viz.visualize_image_attr(
        spatial_attributes.attributes.permute(1, 2, 0)
        .cpu()
        .numpy(),  # adjust shape to height, width, channels
        method="heat_map",
        sign="all",
        plt_fig_axis=(fig, ax[1]),
        show_colorbar=True,
        use_pyplot=False,
    )

    if spatial_attributes.segmentation_mask is not None:
        mask = spatial_attributes.segmentation_mask.cpu()
        highmask = torch.max(mask)
        if len(mask.shape) == 3:
            mask = mask[0]
        ax[2].imshow(mask / highmask, cmap="gray")
        ax[2].set_title("Mask")
        ax[2].grid(False)

    if use_pyplot:
        plt.show()
    else:
        return fig, ax


def visualise_spectral_attributes(
    spectral_attributes: ImageSpectralAttributes | List[ImageSpectralAttributes],
    use_pyplot=False,
    color_palette=None,
    show_not_included=False,
):
    if isinstance(spectral_attributes, ImageSpectralAttributes):
        band_names = spectral_attributes.band_names
    else:
        band_names = spectral_attributes[0].band_names

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Lime attribution")
    visualise_spectral_attributes_by_waveband(
        spectral_attributes,
        ax[0],
        color_palette=color_palette,
        show_not_included=show_not_included,
    )

    visualise_spectral_attributes_by_magnitude(
        spectral_attributes,
        ax[1],
        color_palette=color_palette,
        show_not_included=show_not_included,
    )

    if use_pyplot:
        plt.show()
    else:
        return fig, ax


def visualise_spectral_attributes_by_waveband(
    spectral_attributes: ImageSpectralAttributes | List[ImageSpectralAttributes],
    ax: matplotlib.axes.Axes | None,
    color_palette=None,
    show_not_included=True,
) -> matplotlib.axes.Axes:
    aggregate_results = False
    if not isinstance(spectral_attributes, ImageSpectralAttributes):
        aggregate_results = True

        band_names = dict(spectral_attributes[0].band_names)
        wavelengths = spectral_attributes[0].image.wavelengths
        for i in range(1, len(spectral_attributes)):
            if band_names != spectral_attributes[i].band_names:
                raise ValueError(
                    "All spectral attributes must have the same band names"
                )
                # if band names are all the same, then we can assume that also the masks are the same
            if (wavelengths != spectral_attributes[i].image.wavelengths).any():
                raise ValueError(
                    "All spectral attributes must have the same wavelengths"
                )

        flattened_band_mask = spectral_attributes[0].get_flattened_band_mask().cpu()
        wavelengths = spectral_attributes[0].image.wavelengths

        attribution_map = torch.zeros(
            len(spectral_attributes),
            len(spectral_attributes[0].get_flattened_attributes()),
        )
        for i in range(len(spectral_attributes)):
            attribution_map[i] = spectral_attributes[i].get_flattened_attributes().cpu()

    else:
        band_names = dict(spectral_attributes.band_names)
        flattened_band_mask = spectral_attributes.get_flattened_band_mask().cpu()
        wavelengths = spectral_attributes.image.wavelengths

        attribution_map = (
            spectral_attributes.get_flattened_attributes().unsqueeze(0).cpu()
        )

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    if ax is None:
        ax = plt.gca()

    for idx, band_name in enumerate(band_names.keys()):
        segment_id = band_names[band_name]

        if aggregate_results:
            ax.errorbar(
                wavelengths[flattened_band_mask == segment_id],
                attribution_map[:, flattened_band_mask == segment_id].mean(dim=0),
                yerr=attribution_map[:, flattened_band_mask == segment_id].std(dim=0),
                label=band_name,
                color=color_palette[idx],
                markersize=100,
            )
        else:
            ax.scatter(
                wavelengths[flattened_band_mask == segment_id],
                attribution_map[:, flattened_band_mask == segment_id].mean(dim=0),
                label=band_name,
                s=50,
                color=color_palette[idx],
            )  # Increased marker size

    ax.set_title("Attributions by Waveband", fontsize=14)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Correlation with Output", fontsize=12)
    ax.legend(fontsize=10)
    return ax


def visualise_spectral_attributes_by_magnitude(
    spectral_attributes: ImageSpectralAttributes | List[ImageSpectralAttributes],
    ax: matplotlib.axes.Axes | None,
    color_palette=None,
    annotate_bars=True,
    show_not_included=True,
) -> matplotlib.axes.Axes:
    aggregate_results = False
    if not isinstance(spectral_attributes, ImageSpectralAttributes):
        aggregate_results = True
        band_names = dict(spectral_attributes[0].band_names)
        wavelengths = spectral_attributes[0].image.wavelengths
        for i in range(1, len(spectral_attributes)):
            if band_names != spectral_attributes[i].band_names:
                raise ValueError(
                    "All spectral attributes must have the same band names"
                )
                # if band names are all the same, then we can assume that also the masks are the same
            if (wavelengths != spectral_attributes[i].image.wavelengths).any():
                raise ValueError(
                    "All spectral attributes must have the same wavelengths"
                )

        flattened_band_mask = spectral_attributes[0].get_flattened_band_mask().cpu()
        wavelengths = spectral_attributes[0].image.wavelengths

        attribution_map = torch.zeros(
            len(spectral_attributes),
            len(spectral_attributes[0].get_flattened_attributes()),
        )
        for i in range(len(spectral_attributes)):
            attribution_map[i] = spectral_attributes[i].get_flattened_attributes().cpu()

    else:
        band_names = dict(spectral_attributes.band_names)
        flattened_band_mask = spectral_attributes.get_flattened_band_mask().cpu()
        wavelengths = spectral_attributes.image.wavelengths

        attribution_map = (
            spectral_attributes.get_flattened_attributes().unsqueeze(0).cpu()
        )

    if not show_not_included and band_names.get("not_included") is not None:
        band_names.pop("not_included")

    if color_palette is None:
        color_palette = sns.color_palette("hsv", len(band_names.keys()))

    if ax is None:
        ax = plt.gca()

    avg_magnitudes = [0] * len(band_names)
    for idx, segment_id in enumerate(band_names.values()):
        band = attribution_map[:, flattened_band_mask == segment_id]
        if band.numel() != 0:
            avg_magnitude = band.mean(dim=1)
            avg_magnitudes[idx] = avg_magnitude

    labels = list(band_names.keys())
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

    ax.set_title("Average Attribution Magnitude by Group", fontsize=14)
    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel("Average Attribution Magnitude", fontsize=12)
    ax.tick_params(axis="x", rotation=45)

    return ax
