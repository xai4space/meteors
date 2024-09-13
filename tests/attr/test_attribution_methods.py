import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.utils.models import ExplainableModel
from meteors.attr import IntegratedGradients, Saliency, InputXGradient, NoiseTunnel, Occlusion
from meteors import HSI

import pytest


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


class ExplainableToyModel(ExplainableModel):
    def __init__(self):
        super().__init__(problem_type="regression", forward_func=ToyModel())


def test_integrated_gradients():
    toy_model = ExplainableToyModel()
    ig = IntegratedGradients(toy_model)

    assert ig is not None
    assert ig._attribution_method is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert ig.has_convergence_delta()

    attributions = ig.attribute(image, return_convergence_delta=True)
    assert attributions.approximation_error is not None


def test_saliency():
    toy_model = ExplainableToyModel()
    saliency = Saliency(toy_model)

    assert saliency is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not saliency.has_convergence_delta()
    assert not saliency.multiplies_by_inputs


def test_input_x_gradient():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)

    assert input_x_gradient is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not input_x_gradient.has_convergence_delta()


def test_noise_tunnel():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)
    noise_tunnel = NoiseTunnel(input_x_gradient)

    assert noise_tunnel is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = noise_tunnel.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    # incorrect explainer class
    with pytest.raises(TypeError):
        NoiseTunnel(toy_model)

    with pytest.raises(TypeError):
        NoiseTunnel(InputXGradient)

    with pytest.raises(ValueError):
        NoiseTunnel(noise_tunnel)


def test_occlusion():
    toy_model = ExplainableToyModel()
    occlusion = Occlusion(toy_model)
    assert occlusion is not None

    image = HSI(image=torch.rand(3, 10, 10), wavelengths=[0, 100, 200])
    attributions = occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2))
    assert attributions.attributes.shape == image.image.shape
