import pytest

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from meteors import HSI
from meteors.attr import HSIAttributes, HSISpatialAttributes, HSISpectralAttributes
import meteors.visualize as visualize

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


def test_visualize_hsi_with_incorrect_image():
    # Create incorrect image
    hsi = 5

    # Call the visualize_image function
    with pytest.raises(TypeError):
        visualize.visualize_hsi(hsi, None)


def test_visualize_hsi_with_hsi_object():
    # Create an hsi object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)

    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi, None)

    # Check if the axes object is returned
    assert isinstance(ax, Axes)


def test_visualize_hsi_with_axes_object():
    fig, ax = plt.subplots()
    # Create an hsi object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)

    # Call the visualize_image function
    returned_ax = visualize.visualize_hsi(hsi, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax

    # Wrong number of axes
    ax = [plt.gca(), plt.gca()]
    with pytest.raises(AttributeError):
        visualize.visualize_hsi(hsi, ax)


def test_visualize_hsi_with_hsi_attributes_object():
    # Create an HSIAttributes object
    hsi = HSI(image=torch.ones((len(wavelengths_main), 240, 240)), wavelengths=wavelengths_main)
    attributes = torch.ones_like(hsi.image)
    score = 0.5
    hsi_attributes = HSIAttributes(hsi=hsi, attributes=attributes, score=score, attribution_method="Lime")
    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes, None)

    # Check if the axes object is returned
    assert isinstance(ax, Axes)

    # Create an HSISpatialAttributes object
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(
        hsi=hsi, mask=segmentation_mask, attributes=attributes, score=score, attribution_method="Lime"
    )
    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes_spatial, None)

    # Check if the axes object is returned
    assert isinstance(ax, Axes)

    # Create an HSISpectralAttributes object
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.zeros_like(hsi.image)
    band_mask[0] = 1
    band_mask[1] = 2
    hsi_attributes_spectral = HSISpectralAttributes(
        hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask
    )

    # Call the visualize_image function
    ax = visualize.visualize_hsi(hsi_attributes_spectral, None)

    # Check if the axes object is returned
    assert isinstance(ax, Axes)


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
    hsi_attributes = HSIAttributes(hsi=hsi, attributes=attributes)

    # Create an Axes object
    ax = plt.gca()

    # Call the visualize_image function
    returned_ax = visualize.visualize_hsi(hsi_attributes, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax

    # Create an HSISpatialAttributes object
    segmentation_mask = torch.ones((len(wavelengths_main), 240, 240))
    hsi_attributes_spatial = HSISpatialAttributes(hsi=hsi, mask=segmentation_mask, attributes=attributes)
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
        hsi=hsi, attributes=attributes, band_names=band_names, mask=band_mask
    )
    # Call the visualize_hsi function
    returned_ax = visualize.visualize_hsi(hsi_attributes_spectral, ax)

    # Check if the same axes object is returned
    assert returned_ax is ax
