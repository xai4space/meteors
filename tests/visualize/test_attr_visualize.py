import pytest

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


from meteors import HSI
from meteors.attr import HSISpatialAttributes, HSISpectralAttributes, HSIAttributes, IntegratedGradients
from meteors.visualize import attr_visualize as visualize
from meteors.models import ExplainableModel


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


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.mean(input - 0.9).unsqueeze(0)


@pytest.fixture
def ig_model():
    model = ExplainableModel(forward_func=ToyModel(), problem_type="regression")
    ig = IntegratedGradients(model)
    return ig


def test_visualize_ig_attributes(ig_model):
    tensor_image = torch.rand((len(wavelengths_main), 240, 230))

    tensor_image[20:30, 20:30, 20:30] = 1000
    tensor_image[0:10, 0:10, 0:10] = -500
    tensor_image[50, 50:100, 40:60] = 500

    image = HSI(image=tensor_image, wavelengths=wavelengths_main, orientation="CHW")

    image_attributes = ig_model.attribute(image)

    fig, ax = visualize.visualize_attributes(image_attributes)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2, 2)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert fig.texts[0].get_text() == "HSI Attributes of: Integrated Gradients"
    assert ax[0, 0].get_title() == "Attribution Heatmap"
    assert ax[0, 1].get_title() == "Attribution Module Values"
    assert ax[1, 0].get_title() == "Spectral Attribution"
    assert ax[1, 0].get_xlabel() == "Wavelength"
    assert ax[1, 0].get_ylabel() == "Attribution"
    assert ax[1, 1].get_title() == "Spectral Attribution Absolute Values"
    assert ax[1, 1].get_xlabel() == "Wavelength"
    assert ax[1, 1].get_ylabel() == "Attribution Absolute Value"
    # Cleanup
    plt.close(fig)
    del ax, fig

    # test with given ax
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax = visualize.visualize_attributes(image_attributes, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2, 2)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert fig.texts[0].get_text() == "HSI Attributes of: Integrated Gradients"
    assert ax[0, 0].get_title() == "Attribution Heatmap"
    assert ax[0, 1].get_title() == "Attribution Module Values"
    assert ax[1, 0].get_title() == "Spectral Attribution"
    assert ax[1, 0].get_xlabel() == "Wavelength"
    assert ax[1, 0].get_ylabel() == "Attribution"
    assert ax[1, 1].get_title() == "Spectral Attribution Absolute Values"
    assert ax[1, 1].get_xlabel() == "Wavelength"
    assert ax[1, 1].get_ylabel() == "Attribution Absolute Value"

    # test with incorrect ax shape
    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    ax = visualize.visualize_attributes(image_attributes, ax=ax)
    assert isinstance(ax, np.ndarray)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_attributes(image_attributes, ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_attributes(image_attributes, ax=ax)
    plt.close(fig)
    del ax, fig


def test_incorrect_orientation(ig_model):
    tensor_image = torch.rand((len(wavelengths_main), 240, 230))

    tensor_image[20:30, 20:30, 20:30] = 1000
    tensor_image[0:10, 0:10, 0:10] = -500
    tensor_image[50, 50:100, 40:60] = 500

    image = HSI(image=tensor_image, wavelengths=wavelengths_main, orientation="CWH")

    image_attributes = ig_model.attribute(image)

    fig, ax = visualize.visualize_attributes(image_attributes)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2, 2)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert fig.texts[0].get_text() == "HSI Attributes of: Integrated Gradients"
    assert ax[0, 0].get_title() == "Attribution Heatmap"
    assert ax[0, 1].get_title() == "Attribution Module Values"
    assert ax[1, 0].get_title() == "Spectral Attribution"
    assert ax[1, 0].get_xlabel() == "Wavelength"
    assert ax[1, 0].get_ylabel() == "Attribution"
    assert ax[1, 1].get_title() == "Spectral Attribution Absolute Values"
    assert ax[1, 1].get_xlabel() == "Wavelength"
    assert ax[1, 1].get_ylabel() == "Attribution Absolute Value"
    # Cleanup
    plt.close(fig)


def test_empty_attributions(ig_model):
    tensor_image = torch.rand((len(wavelengths_main), 240, 240))
    image = HSI(image=tensor_image, wavelengths=wavelengths_main)

    image_attributes = ig_model.attribute(image)

    image_attributes.attributes = torch.zeros_like(image_attributes.attributes)

    response = visualize.visualize_attributes(image_attributes)

    assert response is None


def test__merge_band_names_segments():
    band_names = {
        "band1": 0,
        "band2": 1,
        ("band3", "segment1"): 2,
        ("band4", "segment2"): 3,
        ("band5", "segment2"): 4,
    }

    merged_band_names = visualize._merge_band_names_segments(band_names)

    expected_merged_band_names = {
        "band1": 0,
        "band2": 1,
        "band3,segment1": 2,
        "band4,segment2": 3,
        "band5,segment2": 4,
    }
    print(merged_band_names, expected_merged_band_names)
    assert merged_band_names == expected_merged_band_names


def test_visualize_spatial_attributes():
    # Create an HSISpatialAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(hsi=hsi, mask=segmentation_mask, attributes=attributes)

    # Call the visualize_spatial_attributes function
    fig, ax = visualize.visualize_spatial_attributes(hsi_attributes_spatial)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

    assert fig._suptitle.get_text() == "Spatial Attributes Visualization"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test with given ax
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_spatial_attributes(hsi_attributes_spatial, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 4, figsize=(9, 7))
    ax = visualize.visualize_spatial_attributes(hsi_attributes_spatial, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (4,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spatial_attributes(hsi_attributes_spatial, ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spatial_attributes(hsi_attributes_spatial, ax=ax)
    plt.close(fig)
    del ax, fig


def test_visualize_empty_spatial_attributes():
    # Create an HSISpatialAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.zeros_like(hsi.image)
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(hsi=hsi, mask=segmentation_mask, attributes=attributes)

    # Call the visualize_spatial_attributes function
    fig, ax = visualize.visualize_spatial_attributes(hsi_attributes_spatial)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)
    assert fig._suptitle.get_text() == "Spatial Attributes Visualization"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig


def test_validate_consistent_band_and_wavelengths():
    # Test case 1: Consistent band names and wavelengths
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(hsi.image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask),
        HSISpectralAttributes(hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask),
    ]

    # No exception should be raised
    visualize.validate_consistent_band_and_wavelengths(band_names, torch.tensor(wavelengths_main), spectral_attributes)

    # Test case 2: Inconsistent band names
    inconsistent_band_names = {"R": 0, "B": 1, "G": 2}
    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask),
        HSISpectralAttributes(hsi=hsi, attributes=attributes, band_names=inconsistent_band_names, mask=band_mask),
    ]

    # ValueError should be raised for inconsistent band names
    with pytest.raises(ValueError):
        visualize.validate_consistent_band_and_wavelengths(
            inconsistent_band_names, torch.tensor(wavelengths_main), spectral_attributes
        )

    # Test case 3: Inconsistent wavelengths
    inconsistent_wavelengths = wavelengths_main + [1000.0]
    inconsistent_hsi = torch.ones((len(inconsistent_wavelengths), 240, 240))
    inconsistent_mask = torch.zeros_like(inconsistent_hsi)
    inconsistent_mask[0] = 1
    inconsistent_mask[1] = 2
    spectral_attributes = [
        HSISpectralAttributes(hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask),
        HSISpectralAttributes(
            hsi=HSI(image=inconsistent_hsi, wavelengths=inconsistent_wavelengths),  # noqa: E501
            attributes=torch.ones_like(inconsistent_hsi),
            band_names=band_names,
            mask=inconsistent_mask,
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
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == "Test Title 2"
    assert result_ax.get_xlabel() == "Y Label"
    assert result_ax.get_ylabel() == "X Label"

    # Cleanup
    result_ax.clear()
    plt.close("all")
    del ax, fig


def test_visualize_spectral_attributes_by_waveband():
    # Create spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=band_names,
        mask=band_mask,
    )

    # Call the function
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, None)

    # Assert the output
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"
    assert ax.get_legend() is not None

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax)

    # Assert the output
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"
    assert ax.get_legend() is not None

    # Test multiple spectral attributes
    spectral_attributes = [
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            band_names=band_names,
            mask=band_mask,
        ),
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            band_names=band_names,
            mask=band_mask,
        ),
    ]

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax)

    # Assert the output
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Attributions by Waveband"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Correlation with Output"
    assert ax.get_legend() is not None

    # Test invalid input
    with pytest.raises(TypeError):
        visualize.visualize_spectral_attributes_by_waveband("invalid input", None)

    # Test empty list
    with pytest.raises(IndexError):
        visualize.visualize_spectral_attributes_by_waveband([], None)

    # Test custom color palette
    custom_palette = ["red", "green", "blue"]
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, color_palette=custom_palette)

    # Assert the output
    assert isinstance(ax, Axes)

    # Test show_not_included True
    with_not_included_band_names = {"not_included": 0, "R": 1, "G": 2, "B": 3}
    with_not_included_band_mask = band_mask.clone()
    with_not_included_band_mask[3] = 3
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=with_not_included_band_names,
        mask=with_not_included_band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, show_not_included=True)

    # Assert the output
    assert isinstance(ax, Axes)
    # Test show_not_included True
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, show_not_included=False)

    # Assert the output
    assert isinstance(ax, Axes)

    # Test show_legend False
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_waveband(spectral_attributes, ax, show_legend=False)

    # Assert the output
    assert isinstance(ax, Axes)
    assert ax.get_legend() is None

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
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=band_names,
        mask=band_mask,
    )
    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, None)

    # Assert that the plot is correct
    assert isinstance(ax, Axes)
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
    assert isinstance(ax, Axes)
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
            band_names=band_names,
            mask=band_mask,
        ),
        HSISpectralAttributes(
            hsi=HSI(image=image, wavelengths=wavelengths_main),
            attributes=attribution_map,
            band_names=band_names,
            mask=band_mask,
        ),
    ]

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax)

    # Assert that the plot is correct
    assert isinstance(ax, Axes)
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
    with pytest.raises(TypeError):
        visualize.visualize_spectral_attributes_by_magnitude("invalid input", None)

    # Test empty list
    with pytest.raises(IndexError):
        visualize.visualize_spectral_attributes_by_magnitude([], None)

    # Test custom color palette
    custom_palette = ["red", "green", "blue"]
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, color_palette=custom_palette)

    # Assert the output
    assert isinstance(ax, Axes)

    # Test annotate_bars False
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, annotate_bars=False)

    # Assert the output
    assert isinstance(ax, Axes)

    # Test show_not_included True
    with_not_included_band_names = {"not_included": 0, "R": 1, "G": 2, "B": 3}
    band_mask[3] = 3
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=with_not_included_band_names,
        mask=band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, show_not_included=True)

    # Assert the output
    assert isinstance(ax, Axes)

    # Test show_not_included True
    with_not_included_band_names = {"R": 0, "G": 1, "B": 2, "not_included": 3}
    with_not_included_band_mask = band_mask.clone()
    with_not_included_band_mask[3] = 3
    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=with_not_included_band_names,
        mask=with_not_included_band_mask,
    )

    # Call the function
    fig, ax = plt.subplots()
    ax = visualize.visualize_spectral_attributes_by_magnitude(spectral_attributes, ax, show_not_included=False)

    # Assert the output
    assert isinstance(ax, Axes)

    # Cleanup
    ax.clear()
    plt.close("all")
    del ax, fig


def test_visualize_spectral_attributes():
    # Create sample spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=band_names,
        mask=band_mask,
    )

    # Call the function
    fig, ax = visualize.visualize_spectral_attributes(spectral_attributes, use_pyplot=False)

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # test visualization with segment list as a band names

    band_names = {("R", "T"): 0, "G": 1, "B": 2}

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=0.2,
        band_names=band_names,
        mask=band_mask,
    )

    # Call the function
    fig, ax = visualize.visualize_spectral_attributes(spectral_attributes, use_pyplot=False)

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

    # Assert that the title is set correctly
    assert fig._suptitle.get_text() == "Spectral Attributes Visualization"

    # check if the labels are set correctly
    fig.gca().get_xticklabels()[2].get_text() == "R, T"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test ax passed
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    ax = visualize.visualize_spectral_attributes(spectral_attributes, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_spectral_attributes(spectral_attributes, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes(spectral_attributes, ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes(spectral_attributes, ax=ax)
    plt.close(fig)
    del ax, fig


def test_visualize_spectral_attributes_global():
    # Create sample spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes_1 = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        mask=band_mask,
    )

    spectral_attributes_2 = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        mask=band_mask,
    )

    # Call the function
    fig, ax = visualize.visualize_spectral_attributes([spectral_attributes_1, spectral_attributes_2], use_pyplot=False)

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert len(ax) == 3

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

    # Assert that the third subplot shows the distribution of score values
    assert ax[2].get_title() == "Distribution of Score Values"
    assert ax[2].get_xlabel() == "Score"
    assert ax[2].get_ylabel() == "Frequency"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test ax passed
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_spectral_attributes([spectral_attributes_1, spectral_attributes_2], ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    assert ax[2].get_title() == "Distribution of Score Values"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 4, figsize=(9, 7))
    ax = visualize.visualize_spectral_attributes([spectral_attributes_1, spectral_attributes_2], ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (4,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    assert ax[2].get_title() == "Distribution of Score Values"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes([spectral_attributes_1, spectral_attributes_2], ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_attributes([spectral_attributes_1, spectral_attributes_2], ax=ax)
    plt.close(fig)
    del ax, fig


def test_visualize_spectral_empty_attributes():
    # Create sample spectral attributes
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.zeros_like(image)
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        band_names=band_names,
        mask=band_mask,
    )

    # Call the function
    fig, ax = visualize.visualize_spectral_attributes(spectral_attributes, use_pyplot=False)

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # Cleanup
    plt.close("all")
    del ax, fig


def test_visualize_spatial_aggregated_attributes():
    # Create an HSISpatialAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(hsi=hsi, mask=segmentation_mask, attributes=attributes, score=score)
    new_segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    new_segmentation_mask[0, 0, 0] = 0

    # Call the visualize_spatial_attributes function
    fig, ax = visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

    assert fig._suptitle.get_text() == "Spatial Attributes Visualization Aggregated"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Numpy array
    new_segmentation_mask = new_segmentation_mask.numpy()
    fig, ax = visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

    assert fig._suptitle.get_text() == "Spatial Attributes Visualization Aggregated"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Smaller dimensions
    new_segmentation_mask = new_segmentation_mask[0]
    fig, ax = visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

    assert fig._suptitle.get_text() == "Spatial Attributes Visualization Aggregated"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test ax passed
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 4, figsize=(9, 7))
    ax = visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (4,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask, ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spatial_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask, ax=ax)
    plt.close(fig)
    del ax, fig


def test_visualize_spectral_aggregated_attributes():
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
        mask=band_mask,
    )
    new_band_mask = torch.zeros_like(image)
    new_band_mask[1] = 1
    new_band_mask[0] = 2

    # Call the function
    fig, ax = visualize.visualize_spectral_aggregated_attributes(
        spectral_attributes, band_names, new_band_mask, use_pyplot=False
    )

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Numpy array
    new_band_mask = new_band_mask.numpy()
    fig, ax = visualize.visualize_spectral_aggregated_attributes(
        spectral_attributes, band_names, new_band_mask, use_pyplot=False
    )

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Smaller dimensions
    new_band_mask = new_band_mask[:, 0, 0]
    fig, ax = visualize.visualize_spectral_aggregated_attributes(
        spectral_attributes, band_names, new_band_mask, use_pyplot=False
    )

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Test multiple spectral attributes
    spectral_attributes_new = HSISpectralAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        mask=band_mask,
    )

    fig, ax = visualize.visualize_spectral_aggregated_attributes(
        [spectral_attributes, spectral_attributes_new], band_names, new_band_mask, use_pyplot=False
    )

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

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

    # Assert that the third subplot shows the score distribution
    assert ax[2].get_title() == "Distribution of Score Values"
    assert ax[2].get_xlabel() == "Score"
    assert ax[2].get_ylabel() == "Frequency"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test ax passed
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_spectral_aggregated_attributes(
        [spectral_attributes, spectral_attributes_new], band_names, new_band_mask, ax=ax
    )
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    assert ax[2].get_title() == "Distribution of Score Values"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 4, figsize=(9, 7))
    ax = visualize.visualize_spectral_aggregated_attributes(
        [spectral_attributes, spectral_attributes_new], band_names, new_band_mask, ax=ax
    )
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (4,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    assert ax[2].get_title() == "Distribution of Score Values"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_aggregated_attributes(
            [spectral_attributes, spectral_attributes_new], band_names, new_band_mask, ax=ax
        )
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_spectral_aggregated_attributes(
            [spectral_attributes, spectral_attributes_new], band_names, new_band_mask, ax=ax
        )
    plt.close(fig)
    del ax, fig


def test_visualize_aggregated_attributes():
    # Test Spatial Analysis
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSIAttributes(hsi=hsi, mask=segmentation_mask, attributes=attributes, score=score)
    new_segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    new_segmentation_mask[0, 0, 0] = 0

    # Call the visualize_spatial_attributes function
    fig, ax = visualize.visualize_aggregated_attributes(hsi_attributes_spatial, new_segmentation_mask)

    # Check if the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 3
    assert isinstance(ax[0], Axes)

    assert fig._suptitle.get_text() == "Spatial Attributes Visualization Aggregated"
    assert ax[0].get_title() == "Original image"
    assert ax[1].get_title() == "Attribution Map"
    assert ax[2].get_title() == "Mask"

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # Test Spectral Analysis
    image = torch.ones((len(wavelengths_main), 240, 240))
    band_names = {"R": 0, "G": 1, "B": 2}
    attribution_map = torch.rand((image.shape))
    score = 0.5
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(image)
    band_mask[0] = 1
    band_mask[1] = 2

    spectral_attributes = HSIAttributes(
        hsi=HSI(image=image, wavelengths=wavelengths_main),
        attributes=attribution_map,
        score=score,
        band_names=band_names,
        mask=band_mask,
    )
    new_band_mask = torch.zeros_like(image)
    new_band_mask[1] = 1
    new_band_mask[0] = 2

    # Call the function
    fig, ax = visualize.visualize_aggregated_attributes(
        spectral_attributes, new_band_mask, band_names, use_pyplot=False
    )

    # Assert that the figure and axes objects are returned
    assert isinstance(fig, plt.Figure)
    assert len(ax) == 2
    assert isinstance(ax[0], Axes)

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

    # Cleanup
    for a in ax:
        a.clear()
    plt.close("all")
    del ax, fig

    # test ax passed
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    ax = visualize.visualize_aggregated_attributes(spectral_attributes, new_band_mask, band_names, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    plt.close(fig)
    del ax, fig

    # test with incorrect ax shape
    fig, ax = plt.subplots(1, 3, figsize=(9, 7))
    ax = visualize.visualize_aggregated_attributes(spectral_attributes, new_band_mask, band_names, ax=ax)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (3,)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert ax[0].get_title() == "Attributions by Waveband"
    assert ax[1].get_title() == "Attributions by Magnitude"
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_aggregated_attributes(spectral_attributes, new_band_mask, band_names, ax=ax)
    plt.close(fig)
    del ax, fig

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    with pytest.raises(ValueError):
        visualize.visualize_aggregated_attributes(spectral_attributes, new_band_mask, band_names, ax=ax)
    plt.close(fig)
    del ax, fig
