import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.attr import HyperNoiseTunnel, IntegratedGradients, Saliency, NoiseTunnel
from meteors.utils.models.models import ExplainableModel
from meteors.utils.utils import agg_segmentation_postprocessing

from meteors.attr.noise_tunnel import BaseHyperNoiseTunnel, BaseNoiseTunnel

import pytest

import meteors as mt


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input), dim=(1, 2, 3))


class SegmentationToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        adjusted_input = F.relu(input)
        return torch.round(adjusted_input)


@pytest.fixture
def explainable_toy_model():
    return ExplainableModel(problem_type="regression", forward_func=ToyModel())


@pytest.fixture
def explainable_segmentation_toy_model():
    return ExplainableModel(problem_type="segmentation", forward_func=SegmentationToyModel())


def test_torch_random_choice():
    from meteors.attr.noise_tunnel import torch_random_choice

    torch_random_choice(1, 1, 5)

    torch_random_choice(5, 1, 1)

    # both inputs should be integers
    with pytest.raises(TypeError):
        torch_random_choice(0.5, 0.5, 1)

    with pytest.raises(ValueError):
        torch_random_choice(1, 5, 3)


def test_noise_perturb_input():
    input = torch.ones((5, 5, 5))
    # base perturbation with standard deviation
    perturbed_input = BaseNoiseTunnel.perturb_input(input, n_samples=5, stdevs=0.1)

    assert perturbed_input.shape == (5, 5, 5, 5)

    # no standard deviation
    perturbed_input = BaseNoiseTunnel.perturb_input(input, n_samples=5, stdevs=0.0)

    assert perturbed_input.shape == (5, 5, 5, 5)
    assert torch.all(perturbed_input.squeeze(0) == input)

    # perturbation_axis specified
    indexes = (0, 0, slice(1, 4))
    perturbed_input = BaseNoiseTunnel.perturb_input(input, n_samples=5, perturbation_axis=indexes)

    assert perturbed_input.shape == (5, 5, 5, 5)
    assert torch.all(perturbed_input[0, 0, 0, slice(1, 4)] != input[indexes])
    assert torch.all(perturbed_input[0, 1:, 1:, 0] == input[1:, 1:, 0])

    # test errors
    # try to perturb with incorrect number of samples
    with pytest.raises(ValueError):
        BaseNoiseTunnel.perturb_input(input, n_samples=0)

    # try to perturb with incorrect perturbation axis
    with pytest.raises(IndexError):
        BaseNoiseTunnel.perturb_input(input, n_samples=5, perturbation_axis=(0, 0, 0, 0))


def test_hyper_noise_perturb_input():
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


def test_noise_attribute(explainable_toy_model, explainable_segmentation_toy_model):
    ig = IntegratedGradients(explainable_toy_model)
    noise_tunnel = NoiseTunnel(ig)

    torch.manual_seed(0)

    tensor_image = torch.rand((5, 5, 5))
    image = mt.HSI(image=tensor_image, wavelengths=[1, 2, 3, 4, 5], orientation=("C", "H", "W"))

    attributes = noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute(image, n_samples=1, method="smoothgrad_sq")

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute(image, n_samples=1, method="vargrad", steps_per_batch=3)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute(image, n_samples=1, stdevs=0.1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute(image, n_samples=1, stdevs=[0.1])

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    # test changing the image orientation
    image.orientation = ("W", "C", "H")
    attributes = noise_tunnel.attribute(image, n_samples=1)
    assert attributes.orientation == ("W", "C", "H")

    # test validation

    # test with incorrect method
    image.orientation = ("C", "H", "W")
    with pytest.raises(ValueError):
        noise_tunnel.attribute(image, n_samples=1, method="incorrect")

    # incorrect explainer class
    with pytest.raises(TypeError):
        NoiseTunnel(explainable_toy_model)

    # test incorrect stdevs
    # This one will pass but will informt that first stdev will be used
    noise_tunnel.attribute(image, n_samples=1, stdevs=[0.1, 0.2])

    # different explanation method
    saliency = Saliency(explainable_toy_model)

    noise_tunnel = NoiseTunnel(saliency)
    attributes = noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    # Test multiple images
    attributes = noise_tunnel.attribute([image, image], n_samples=1)

    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute([image, image], n_samples=1, stdevs=[0.1, 0.2])

    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    attributes = noise_tunnel.attribute([image, image], n_samples=1, target=[None, None])

    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    # Test segmentation output
    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    ig = IntegratedGradients(
        explainable_segmentation_toy_model, postprocessing_segmentation_output=postprocessing_segmentation_output
    )
    hyper_noise_tunnel = HyperNoiseTunnel(ig)
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, target=0)
    assert attributes.attributes.shape == image.image.shape

    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, target=0)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == image.image.shape
    assert attributes[1].attributes.shape == image.image.shape

    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, target=[0, 0])
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == image.image.shape
    assert attributes[1].attributes.shape == image.image.shape


def test_hyper_attribute(explainable_toy_model, explainable_segmentation_toy_model):
    ig = IntegratedGradients(explainable_toy_model)
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

    # test various baselines
    baselines = torch.zeros((5, 5, 5))
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, baselines=baselines)
    assert attributes.attributes.shape == (5, 5, 5)

    baselines = 0
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, baselines=baselines)
    assert attributes.attributes.shape == (5, 5, 5)

    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1, baselines=torch.ones((5, 5)))

    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1, baselines=torch.ones((1, 5, 5, 5)))

    # test validation

    # test with incorrect method
    image.orientation = ("C", "H", "W")
    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute(image, n_samples=1, method="incorrect")

    # incorrect explainer class
    with pytest.raises(TypeError):
        HyperNoiseTunnel(explainable_toy_model)

    # different explanation method
    saliency = Saliency(explainable_toy_model)

    hyper_noise_tunnel = HyperNoiseTunnel(saliency)
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)

    # Test multiple images
    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    baselines = torch.zeros((5, 5, 5))
    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, baselines=baselines)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    baselines = 0
    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, baselines=baselines)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)

    baselines = torch.zeros((2, 5, 5, 5))
    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, baselines=baselines)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, target=[None, None])
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == (5, 5, 5)
    assert attributes[1].attributes.shape == (5, 5, 5)

    with pytest.raises(ValueError):
        hyper_noise_tunnel.attribute([image, image], n_samples=1, baselines=torch.ones((1, 5, 5, 5)))

    # Test segmentation output
    postprocessing_segmentation_output = agg_segmentation_postprocessing(classes_numb=3)
    ig = IntegratedGradients(
        explainable_segmentation_toy_model, postprocessing_segmentation_output=postprocessing_segmentation_output
    )
    hyper_noise_tunnel = HyperNoiseTunnel(ig)
    attributes = hyper_noise_tunnel.attribute(image, n_samples=1, target=0)
    assert attributes.attributes.shape == image.image.shape

    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, target=0)
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == image.image.shape
    assert attributes[1].attributes.shape == image.image.shape

    attributes = hyper_noise_tunnel.attribute([image, image], n_samples=1, target=[0, 0])
    assert len(attributes) == 2
    assert attributes[0].attributes.shape == image.image.shape
    assert attributes[1].attributes.shape == image.image.shape
