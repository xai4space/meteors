import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.attr import HyperNoiseTunnel, IntegratedGradients
from meteors.utils.models.models import ExplainableModel

from meteors.attr.hyper_noise_tunnel import BaseHyperNoiseTunnel

import pytest

import meteors as mt


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


class ExplainableToyModel(ExplainableModel):
    def __init__(self):
        super().__init__(problem_type="regression", forward_func=ToyModel())


def test_torch_random_choice():
    from meteors.attr.hyper_noise_tunnel import torch_random_choice

    torch_random_choice(1, 1, 5)

    torch_random_choice(5, 1, 1)

    # both inputs should be integers
    with pytest.raises(TypeError):
        torch_random_choice(0.5, 0.5, 1)

    with pytest.raises(ValueError):
        torch_random_choice(1, 5, 3)


def test_perturb_input():
    input = torch.ones((5, 5, 5))
    baselines = torch.zeros((5, 5, 5))
    # base perturbation with given probability
    perturbed_input = BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=5, perturbation_prob=0.5)

    assert perturbed_input.shape == (5, 5, 5, 5)

    # no perturbation
    perturbed_input = BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=1, num_perturbed_bands=0)

    assert perturbed_input.shape == (1, 5, 5, 5)
    assert torch.all(perturbed_input.squeeze(0) == input)

    # full perturbation
    perturbed_input = BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=1, num_perturbed_bands=5)

    assert perturbed_input.shape == (1, 5, 5, 5)
    assert torch.all(perturbed_input.squeeze(0) == baselines)

    # test errors
    # try to perturb with incorrect probability
    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=5, perturbation_prob=1.5)

    # try to perturb too many bands
    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=5, num_perturbed_bands=6)

    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=5, num_perturbed_bands=-1)

    # try to perturb with incorrect number of samples
    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples=0)

    # try to perturb with incorrect input
    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(torch.ones((5, 5)), baselines, n_samples=5)


def test_attribute():
    model = ExplainableToyModel()

    ig = IntegratedGradients(model)
    hyper_noise_tunnel = HyperNoiseTunnel(ig)

    torch.manual_seed(0)

    tensor_image = torch.rand((5, 5, 5))
    image = mt.Image(image=tensor_image, wavelengths=[1, 2, 3, 4, 5], orientation=("C", "H", "W"))

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, method="smoothgrad_sq")

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, method="vargrad", steps_per_batch=3)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    # test validation
    # test with incorrect orientation
    image.orientation = ("H", "W", "C")
    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1)

    # test with incorrect method
    image.orientation = ("C", "H", "W")
    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1, method="incorrect")

    # incorrect explainer class
    with pytest.raises(TypeError):
        HyperNoiseTunnel(model)
