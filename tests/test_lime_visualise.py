import pytest

import torch
import numpy as np
import matplotlib.pyplot as plt

from meteors import (
    HSI,
    HSIAttributes,
    HSISpectralAttributes,
    HSISpatialAttributes,
)
from meteors.visualize import lime_visualize as visualize


# Temporary solution for wavelengths
wavelengths_main = [
    462.08,
    465.27,
    468.47,
    471.67,
    474.86,
    478.06,
    481.26,
    484.45,
    487.65,
    490.85,
    494.04,
    497.24,
    500.43,
    503.63,
    506.83,
    510.03,
    513.22,
    516.42,
    519.61,
    522.81,
    526.01,
    529.2,
    532.4,
    535.6,
    538.79,
    541.99,
    545.19,
    548.38,
    551.58,
    554.78,
    557.97,
    561.17,
    564.37,
    567.56,
    570.76,
    573.96,
    577.15,
    580.35,
    583.55,
    586.74,
    589.94,
    593.14,
    596.33,
    599.53,
    602.73,
    605.92,
    609.12,
    612.32,
    615.51,
    618.71,
    621.91,
    625.1,
    628.3,
    631.5,
    634.69,
    637.89,
    641.09,
    644.28,
    647.48,
    650.67,
    653.87,
    657.07,
    660.27,
    663.46,
    666.66,
    669.85,
    673.05,
    676.25,
    679.45,
    682.64,
    685.84,
    689.03,
    692.23,
    695.43,
    698.62,
]


def test_visualize_hsi_with_hsi_object():
    # Create an hsi object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)

    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi, None)

    # Check if the axes object is returned
    assert isinstance(ax, plt.Axes)


def test_visualize_hsi_with_hsi_attributes_object():
    # Create an HSIAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    hsi_attributes = HSIAttributes(hsi=hsi, attributes=attributes, score=score)
    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes, None)

    # Check if the axes object is returned
    assert isinstance(ax, plt.Axes)

    # Create an HSISpatialAttributes object
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(
        hsi=hsi, segmentation_mask=segmentation_mask, attributes=attributes, score=score
    )
    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes_spatial, None)

    # Check if the axes object is returned
    assert isinstance(ax, plt.Axes)

    # Create an HSISpectralAttributes object
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(hsi.image)
    band_mask[0] = 1
    band_mask[1] = 2
    hsi_attributes_spectral = HSISpectralAttributes(
        hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask
    )

    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes_spectral, None)

    # Check if the axes object is returned
    assert isinstance(ax, plt.Axes)


def test_visualize_hsi_with_hsi_object_and_ax():
    # Create an hsi object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)

    # Create an Axes object
    ax = plt.gca()

    # Call the visualize_hsi function
    returned_ax = visualize.visualize_hsi(hsi, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax


def test_visualize_hsi_with_hsi_attributes_object_and_ax():
    # Create an HSIAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    hsi_attributes = HSIAttributes(hsi=hsi, attributes=attributes, score=score)

    # Create an Axes object
    ax = plt.gca()

    # Call the visualize_image function
    returned_ax = visualize.visualize_hsi(hsi_attributes, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax

    # Create an HSISpatialAttributes object
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(
        hsi=hsi, segmentation_mask=segmentation_mask, attributes=attributes, score=score
    )
    # Call the visualize_image function
    returned_ax = visualize.visualize_hsi(hsi_attributes_spatial, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax

    # Create an HSISpectralAttributes object
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(hsi.image)
    band_mask[0] = 1
    band_mask[1] = 2
    hsi_attributes_spectral = HSISpectralAttributes(
        hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask
    )
    # Call the visualize_hsi function
    returned_ax = visualize.visualize_hsi(hsi_attributes_spectral, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax


def test_visualize_spatial_attributes():
    # Create an HSISpatialAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(
        hsi=hsi, segmentation_mask=segmentation_mask, attributes=attributes, score=score
    )

    # Call the visualize_spatial_attributes function
    fig, ax = visualize.visualize_spatial_attributes(hsi_attributes_spatial)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 3

    # Assert that the title is set correctly
    assert fig._suptitle.get_text() == "Spatial Attributes Visualization"

    # Assert that the first subplot shows the original image
    assert ax[0].get_title() == "Original image"

    # Assert that the third subplot shows the segmentation mask
    assert ax[2].get_title() == "Mask"

    # Cleanup
    plt.close(fig)


def test_validate_consistent_band_and_wavelengths():
    # Test case 1: Consistent band names and wavelengths
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(hsi.image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask),
        HSISpectralAttributes(hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask),
    ]

    # No exception should be raised
    visualize.validate_consistent_band_and_wavelengths(band_names, torch.tensor(wavelengths_main), spectral_attributes)

    # Test case 2: Inconsistent band names
    inconsistent_band_names = {"R": 0, "B": 1, "G": 2}
    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask),
        HSISpectralAttributes(
            hsi=hsi, attributes=attributes, score=score, band_names=inconsistent_band_names, band_mask=band_mask
        ),
    ]

    # ValueError should be raised for inconsistent band names
    with pytest.raises(ValueError):
        visualize.validate_consistent_band_and_wavelengths(
            inconsistent_band_names, torch.tensor(wavelengths_main), spectral_attributes
        )

    # Test case 3: Inconsistent wavelengths
    inconsistent_wavelengths = wavelengths_main + [1000.0]
    inconsistent_hsi = torch.ones((len(inconsistent_wavelengths), 240, 240))
    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, score=score, band_names=band_names, band_mask=band_mask),
        HSISpectralAttributes(
            hsi=HSI(image=inconsistent_hsi, wavelengths=inconsistent_wavelengths),  # noqa: E501
            attributes=torch.ones_like(inconsistent_hsi),
            score=score,
            band_names=band_names,
            band_mask=torch.zeros_like(inconsistent_hsi),
        ),
    ]

    # ValueError should be raised for inconsistent wavelengths
    with pytest.raises(ValueError):
        visualize.validate_consistent_band_and_wavelengths(
            band_names, torch.tensor(inconsistent_wavelengths), spectral_attributes
        )


def test_setup_visualization():
    # Test with existing axes
    fig, ax = plt.subplots()
    result_ax = visualize.setup_visualization(ax, "Test Title", "X Label", "Y Label")
    assert result_ax is ax
    assert result_ax.get_title() == "Test Title"
    assert result_ax.get_xlabel() == "X Label"
    assert result_ax.get_ylabel() == "Y Label"

    # Test with None axes
    result_ax = visualize.setup_visualization(
        None,
        "Test Title 2",
        "Y Label",
        "X Label",
    )
    assert isinstance(result_ax, plt.Axes)
    assert result_ax.get_title() == "Test Title 2"
    assert result_ax.get_xlabel() == "Y Label"
    assert result_ax.get_ylabel() == "X Label"

    # Cleanup
    plt.close(fig)


def test_visualize_spectral_attributes_by_waveband():
    # Create spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
    )

    # Call the function
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, None)

    # Assert the output
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax)

    # Assert the output
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"

    # Test multiple spectral attributes
    spectral_attributes = [
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            score=score,
            band_names=band_names,
            band_mask=band_mask,
        ),
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            score=score,
            band_names=band_names,
            band_mask=band_mask,
        ),
    ]

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax)

    # Assert the output
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"

    # Test invalid input
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes_by_waveband("invalid input", None)

    # Test empty list
    with pytest.raises(IndexError):
        visualize.visualize_spectral_attributes_by_waveband([], None)

    # Test custom color palette
    custom_palette = ["red", "green", "blue"]
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, color_palette=custom_palette)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Test show_not_included True
    with_not_included_band_names = {"R": 0, "G": 1, "B": 2, "not_included": 3}
    with_not_included_band_mask = band_mask.clone()
    with_not_included_band_mask[3] = 1
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=with_not_included_band_names,
        band_mask=with_not_included_band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, show_not_included=True)

    # Assert the output
    assert isinstance(ax, plt.Axes)
    # Test show_not_included True
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, show_not_included=False)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Cleanup
    ax.clear()
    fig.clear()
    plt.close("all")
    del ax, fig


def test_calculate_average_magnitudes():
    # Test case 1: Basic functionality
    band_names = {"band1": 0, "band2": 1}
    flattened_band_mask = torch.tensor([0, 0, 1, 1])
    attribution_map = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = visualize.calculate_average_magnitudes(band_names, flattened_band_mask, attribution_map)

    assert result.shape == (len(band_names),)
    assert torch.allclose(result, torch.tensor([1.5, 3.5]))

    # Test case 2: Multiple attributions maps
    band_names = {"band1": 0, "band2": 1}
    flattened_band_mask = torch.tensor([0, 0, 1, 1])
    attribution_map = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    result = visualize.calculate_average_magnitudes(band_names, flattened_band_mask, attribution_map)

    assert result.shape == (len(band_names), attribution_map.shape[0])
    assert torch.allclose(result, torch.tensor([[1.5, 3.5], [5.5, 7.5]]))

    # Test case 3: Empty input
    band_names = {"band1": 0, "band2": 1, "band3": 2}
    flattened_band_mask = torch.tensor([0, 0, 1, 1])
    attribution_map = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = visualize.calculate_average_magnitudes(band_names, flattened_band_mask, attribution_map)

    assert result.shape == (len(band_names),)
    assert torch.allclose(result, torch.tensor([1.5, 3.5, 0.0]))

    # Test case 4: Single Band
    band_names = {"band1": 0}
    flattened_band_mask = torch.tensor([0, 0, 0, 0])
    attribution_map = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = visualize.calculate_average_magnitudes(band_names, flattened_band_mask, attribution_map)

    assert result.shape == (len(band_names),)
    assert torch.allclose(result, torch.tensor([2.5]))

    # Test case 5: mismatched segment ids
    band_names = {"band1": 0, "band2": 2}  # Note: segment ID 2 is not in flattened_band_mask
    flattened_band_mask = torch.tensor([0, 0, 1, 1])
    attribution_map = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = visualize.calculate_average_magnitudes(band_names, flattened_band_mask, attribution_map)

    assert result.shape == (len(band_names),)
    assert torch.allclose(result, torch.tensor([1.5, 0.0]))


def test_visualize_spectral_attributes_by_magnitude():
    # Create spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
    )
    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, None)

    # Assert that the plot is correct
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Magnitude"
    assert ax.get_xlabel() == "Group"
    assert ax.get_ylabel() == "Average Attribution Magnitude"

    # Assert that the bars are correctly plotted
    bars = ax.patches
    assert len(bars) == len(band_names)
    for bar, (label, segment_id) in zip(bars, band_names.items()):
        assert ax.get_xticklabels()[segment_id].get_text() == label

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax

    # Test with axes
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax)

    # Assert that the plot is correct
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Magnitude"
    assert ax.get_xlabel() == "Group"
    assert ax.get_ylabel() == "Average Attribution Magnitude"

    # Assert that the bars are correctly plotted
    bars = ax.patches
    assert len(bars) == len(band_names)
    for bar, (label, segment_id) in zip(bars, band_names.items()):
        assert ax.get_xticklabels()[segment_id].get_text() == label

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax, fig

    # Test multiple spectral attributes
    spectral_attributes = [
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            score=score,
            band_names=band_names,
            band_mask=band_mask,
        ),
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            score=score,
            band_names=band_names,
            band_mask=band_mask,
        ),
    ]

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax)

    # Assert that the plot is correct
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Attributions by Magnitude"
    assert ax.get_xlabel() == "Group"
    assert ax.get_ylabel() == "Average Attribution Magnitude"

    # Assert that the bars are correctly plotted
    bars = ax.patches
    assert len(bars) == len(band_names)
    for bar, (label, segment_id) in zip(bars, band_names.items()):
        assert ax.get_xticklabels()[segment_id].get_text() == label

    # Cleanup
    ax.clear()
    plt.close("all")

    # Test invalid input
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes_by_magnitude("invalid input", None)

    # Test empty list
    with pytest.raises(IndexError):
        visualize.visualize_spectral_attributes_by_magnitude([], None)

    # Test custom color palette
    custom_palette = ["red", "green", "blue"]
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, color_palette=custom_palette)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Test annotate_bars False
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, annotate_bars=False)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Test show_not_included True
    with_not_included_band_names = {"R": 0, "G": 1, "B": 2, "not_included": 3}
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=with_not_included_band_names,
        band_mask=band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, show_not_included=True)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Test show_not_included True
    with_not_included_band_names = {"R": 0, "G": 1, "B": 2, "not_included": 3}
    with_not_included_band_mask = band_mask.clone()
    with_not_included_band_mask[3] = 3
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=with_not_included_band_mask,
        score=score,
        band_names=with_not_included_band_names,
        band_mask=band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, show_not_included=False)

    # Assert the output
    assert isinstance(ax, plt.Axes)

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax, fig


def test_visualize_spectral_attributes():
    # Create sample spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
    )

    # Call the function
    fig, ax = visualize.visualize_spectral_attributes(spectral_attributes, use_pyplot=False)

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 2

    # Assert that the title is set correctly
    assert fig._suptitle.get_text() == "Spectral Attributes Visualization"

    # Assert that the first subplot shows the visualization by waveband
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[0].get_xlabel() == "Wavelength (nm)"
    assert ax[0].get_ylabel() == "Correlation with Output"

    # Assert that the second subplot shows the visualization by magnitude
    assert ax[1].get_title() == "Attributions by Magnitude"
    assert ax[1].get_xlabel() == "Group"
    assert ax[1].get_ylabel() == "Average Attribution Magnitude"
