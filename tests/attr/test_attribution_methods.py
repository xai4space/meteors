import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.utils.models import ExplainableModel
from meteors.attr import IntegratedGradients, Saliency, InputXGradient, NoiseTunnel, Occlusion, Explainer
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

    # chained explainer - similar to the NoiseTunnel class
    chained_explainer = Explainer(explainer)

    # chained explainer with chained explainer - should raise ValueError
    with pytest.raises(ValueError):
        Explainer(chained_explainer)

    assert not explainer.has_convergence_delta()

    with pytest.raises(NotImplementedError):
        explainer.compute_convergence_delta()

    assert not explainer.multiplies_by_inputs


def test_explainer_validation():
    with pytest.raises(TypeError):
        explainer = Explainer(lambda x: x)

    from meteors.attr.explainer import validate_attribution_method_initialization

    explainer = None
    with pytest.raises(ValueError):
        validate_attribution_method_initialization(explainer)

    explainer = Explainer(ExplainableToyModel())
    validate_attribution_method_initialization(explainer)

    from meteors.attr.explainer import validate_and_transform_baseline

    baseline = 1
    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    baseline = validate_and_transform_baseline(baseline, image)
    assert baseline.shape == image.image.shape
    assert torch.all(baseline == 1)

    baseline = torch.rand(3, 224, 224)
    baseline = validate_and_transform_baseline(baseline, image)
    assert baseline.shape == image.image.shape

    baseline = torch.rand(3, 224, 225)
    with pytest.raises(ValueError):
        validate_and_transform_baseline(baseline, image)


def test_integrated_gradients():
    toy_model = ExplainableToyModel()
    ig = IntegratedGradients(toy_model)

    assert ig is not None
    assert ig._attribution_method is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_saliency():
    toy_model = ExplainableToyModel()
    saliency = Saliency(toy_model)

    assert saliency is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = saliency.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_input_x_gradient():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)

    assert input_x_gradient is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = input_x_gradient.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_noise_tunnel():
    toy_model = ExplainableToyModel()
    input_x_gradient = InputXGradient(toy_model)
    noise_tunnel = NoiseTunnel(input_x_gradient)

    assert noise_tunnel is not None

    image = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    attributions = noise_tunnel.attribute(image)
    assert attributions.attributes.shape == image.image.shape


def test_occlusion():
    toy_model = ExplainableToyModel()
    occlusion = Occlusion(toy_model)
    assert occlusion is not None

    image = HSI(image=torch.rand(3, 10, 10), wavelengths=[0, 100, 200])
    attributions = occlusion.attribute(image, sliding_window_shapes=(2, 2), strides=(2, 2))
    assert attributions.attributes.shape == image.image.shape


test_occlusion()
