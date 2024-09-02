import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.utils.models import ExplainableModel
from meteors.attr import IntegratedGradients, Saliency, InputXGradient, NoiseTunnel, Occlusion, Explainer
from meteors import Image

import pytest


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


class ExplainableToyModel(ExplainableModel):
    def __init__(self):
        super().__init__(problem_type="regression", forward_func=ToyModel())


def test_explainer():
    # Create mock objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")

    # Create a Explainer object
    explainer = Explainer(explainable_model)

    # Assert that the explainable_model is set correctly
    assert explainer.explainable_model == explainable_model

    # Test case 1: Valid input
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    explainable_model = ExplainableModel(forward_func=dumb_model, problem_type="regression")
    # Create a sample Explainer object
    explainer = Explainer(explainable_model)
    device = "cpu"

    # Call the to method
    explainer.to(device)

    # not implemented
    with pytest.raises(NotImplementedError):
        explainer.attribute(image=Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200]))

    # chained explainer - similar to the NoiseTunnel class
    chained_explainer = Explainer(explainer)

    # chained explainer with chained explainer - should raise ValueError
    with pytest.raises(ValueError):
        Explainer(chained_explainer)


def test_integrated_gradients():
    toy_model = ExplainableToyModel()
    ig = IntegratedGradients(toy_model)

    assert ig is not None
    assert ig._ig is not None

    image = Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_saliency():
    toy_model = ExplainableToyModel()
    saliency = Saliency(toy_model)

    assert saliency is not None

    image = Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_input_x_gradient():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)

    assert input_x_gradient is not None

    image = Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_noise_tunnel():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)
    noise_tunnel = NoiseTunnel(input_x_gradient)

    assert noise_tunnel is not None

    image = Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = noise_tunnel.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_occlusion():
    toy_model = ExplainableToyModel()
    occlusion = Occlusion(toy_model)

    assert occlusion is not None

    image = Image(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = occlusion.attribute(image)
    assert attributions.attributes.shape == image.image.shape


test_noise_tunnel()
