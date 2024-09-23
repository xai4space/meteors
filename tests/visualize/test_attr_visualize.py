import pytest

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import meteors as mt
from meteors.utils.models import ExplainableModel


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
    ig = mt.attr.IntegratedGradients(model)
    return ig


def test_visualize_ig_attributes(ig_model):
    tensor_image = torch.rand((len(wavelengths_main), 240, 230))

    tensor_image[20:30, 20:30, 20:30] = 1000
    tensor_image[0:10, 0:10, 0:10] = -500
    tensor_image[50, 50:100, 40:60] = 500

    image = mt.HSI(image=tensor_image, wavelengths=wavelengths_main)

    image_attributes = ig_model.attribute(image)

    fig, ax = mt.visualize.visualize_attributes(image_attributes)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == (2, 3)
    assert all([isinstance(a, Axes) for a in ax.ravel()])
    assert fig.texts[0].get_text() == "HSI Attributes of: Integrated Gradients"
    assert ax[0, 0].get_title() == "Attribution Heatmap"
    assert ax[0, 1].get_title() == "Attribution Module Values"
    assert ax[0, 2].get_title() == "Attribution Sign Values"
    assert ax[1, 0].get_title() == "Spectral Attribution"
    assert ax[1, 0].get_xlabel() == "Wavelength"
    assert ax[1, 0].get_ylabel() == "Attribution"
    assert ax[1, 1].get_title() == "Spectral Attribution Absolute Values"
    assert ax[1, 1].get_xlabel() == "Wavelength"
    assert ax[1, 1].get_ylabel() == "Attribution Absolute Value"
    assert ax[1, 2].get_title() == "Spectral Attribution Sign Values"
    assert ax[1, 2].get_xlabel() == "Wavelength"
    assert ax[1, 2].get_ylabel() == "Attribution Sign Proportion"
    assert ax[1, 2].get_yticks().tolist() == [-1, 0, 1]

    # Cleanup
    plt.close(fig)


model = ExplainableModel(forward_func=ToyModel(), problem_type="regression")
ig = mt.attr.IntegratedGradients(model)
test_visualize_ig_attributes(ig)


def test_validation_checks(ig_model):
    tensor_image = torch.rand((5, 5, len(wavelengths_main)))
    image = mt.HSI(image=tensor_image, wavelengths=wavelengths_main, orientation=("H", "W", "C"))
    image_attributes = ig_model.attribute(image)

    with pytest.raises(ValueError):
        mt.visualize.visualize_attributes(image_attributes)


def test_empty_attributions(ig_model):
    tensor_image = torch.rand((len(wavelengths_main), 240, 240))
    image = mt.HSI(image=tensor_image, wavelengths=wavelengths_main)

    image_attributes = ig_model.attribute(image)

    image_attributes.attributes = torch.zeros_like(image_attributes.attributes)

    response = mt.visualize.visualize_attributes(image_attributes)

    assert response is None
