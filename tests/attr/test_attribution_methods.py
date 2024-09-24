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


@pytest.fixture
def explainable_toy_model():
    return ExplainableModel(problem_type="regression", forward_func=ToyModel())


def test_integrated_gradients(explainable_toy_model):
    ig = IntegratedGradients(explainable_toy_model)

    assert ig is not None
    assert ig._attribution_method is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert ig.has_convergence_delta()

    attributions = ig.attribute(image, return_convergence_delta=True)
    assert attributions.score is not None

    with pytest.raises(ValueError):
        ig._attribution_method = None
        ig.attribute(image)


def test_saliency(explainable_toy_model):
    saliency = Saliency(explainable_toy_model)

    assert saliency is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not saliency.has_convergence_delta()
    assert not saliency.multiplies_by_inputs

    with pytest.raises(ValueError):
        saliency._attribution_method = None
        saliency.attribute(image)


def test_input_x_gradient(explainable_toy_model):
    input_x_gradient = InputXGradient(explainable_toy_model)

    assert input_x_gradient is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    assert not input_x_gradient.has_convergence_delta()

    with pytest.raises(ValueError):
        input_x_gradient._attribution_method = None
        input_x_gradient.attribute(image)


def test_noise_tunnel(explainable_toy_model):
    input_x_gradient = InputXGradient(explainable_toy_model)
    noise_tunnel = NoiseTunnel(input_x_gradient)

    assert noise_tunnel is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = noise_tunnel.attribute(image)
    assert attributions.attributes.shape == image.image.shape

    # incorrect explainer class
    with pytest.raises(TypeError):
        NoiseTunnel(explainable_toy_model)

    with pytest.raises(TypeError):
        NoiseTunnel(InputXGradient)

    with pytest.raises(ValueError):
        NoiseTunnel(noise_tunnel)

    with pytest.raises(ValueError):
        noise_tunnel._attribution_method = None
        noise_tunnel.attribute(image)

    with pytest.raises(ValueError):
        noise_tunnel = NoiseTunnel(input_x_gradient)
        noise_tunnel.chained_explainer = None
        noise_tunnel.attribute(image)


def test_occlusion(explainable_toy_model):
    occlusion = Occlusion(explainable_toy_model)
    assert occlusion is not None

    image = HSI(image=torch.rand(3, 10, 10), wavelengths=[0, 100, 200])
    attributions = occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2))
    assert attributions.attributes.shape == image.image.shape

    with pytest.raises(ValueError):
        occlusion._attribution_method = None
        occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2))
