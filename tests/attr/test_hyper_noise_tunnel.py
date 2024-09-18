import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.attr import HyperNoiseTunnel, IntegratedGradients, Saliency
from meteors.utils.models.models import ExplainableModel

from meteors.attr.hyper_noise_tunnel import BaseHyperNoiseTunnel

import pytest

import meteors as mt


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


@pytest.fixture
def model():
    return ExplainableModel(problem_type="regression", forward_func=ToyModel())


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

    # try to perturb with incorrect baselines
    with pytest.raises(ValueError):
        BaseHyperNoiseTunnel.perturb_input(input, torch.ones((5, 5)), n_samples=5)


def test_attribute(model):
    ig = IntegratedGradients(model)
    hyper_noise_tunnel = HyperNoiseTunnel(ig)

    torch.manual_seed(0)

    tensor_image = torch.rand((5, 5, 5))
    image = mt.HSI(image=tensor_image, wavelengths=[1, 2, 3, 4, 5], orientation=("C", "H", "W"))

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, method="smoothgrad_sq")

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, method="vargrad", steps_per_batch=3)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    # test changing the image orientation
    image.orientation = ("W", "C", "H")
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)
    assert attributes.orientation == ("W", "C", "H")

    # test validation

    # test with incorrect method
    image.orientation = ("C", "H", "W")
    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1, method="incorrect")

    # incorrect explainer class
    with pytest.raises(TypeError):
        HyperNoiseTunnel(model)

    # different explanation method
    saliency = Saliency(model)

    hyper_noise_tunnel = HyperNoiseTunnel(saliency)
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)


def test_base_hyper_noise_tunnel_class(model):
    tensor_image = torch.rand((5, 5, 5))
    baselines = torch.zeros((1, 5, 5, 5))

    ig = IntegratedGradients(model)
    base_hyper_noise_tunnel = BaseHyperNoiseTunnel(ig._attribution_method)

    tensor_attributes = base_hyper_noise_tunnel.attribute(tensor_image, baselines=baselines, n_samples=1)

    assert tensor_attributes is not None
    assert tensor_attributes.shape == (1, 5, 5, 5)

    tensor_image_with_incorrect_shape = torch.rand((5, 5))

    with pytest.raises(ValueError):
        base_hyper_noise_tunnel.attribute(tensor_image_with_incorrect_shape, baselines=baselines, n_samples=1)

    integer_baselines = 0
    base_hyper_noise_tunnel.attribute(tensor_image, baselines=integer_baselines, n_samples=1)

    incorrect_baselines = torch.zeros((1, 5))
    with pytest.raises(ValueError):
        base_hyper_noise_tunnel.attribute(tensor_image, baselines=incorrect_baselines, n_samples=1)
