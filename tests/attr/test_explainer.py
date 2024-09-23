import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.utils.models import ExplainableModel
import meteors.attr.explainer as explainer_module
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


def test_validate_attribution_method_initialization(explainable_toy_model):
    # Test case 1: Valid attribution method
    attribution_method = explainer_module.Explainer(explainable_toy_model)
    explainer_module.validate_attribution_method_initialization(attribution_method)

    # Test case 2: None attribution method
    attribution_method = None
    with pytest.raises(ValueError):
        explainer_module.validate_attribution_method_initialization(attribution_method)

    # Test case 3: Invalid attribution method
    attribution_method = "invalid"
    with pytest.raises(TypeError):
        explainer_module.validate_attribution_method_initialization(attribution_method)

    # Test case 4: None forward function in ExplainableModel
    not_working_model = explainer_module.Explainer(ExplainableModel(problem_type="regression", forward_func=None))
    with pytest.raises(ValueError):
        explainer_module.validate_attribution_method_initialization(not_working_model)

    # test case 5: None explainable_model in Explainer


def test_validate_and_transform_baseline():
    # Test case 1: None baseline
    baseline = None
    hsi = HSI(image=torch.rand(3, 224, 224), wavelengths=[0, 100, 200])
    transformed_baseline = explainer_module.validate_and_transform_baseline(baseline, hsi)
    assert transformed_baseline.shape == hsi.image.shape
    assert torch.all(transformed_baseline == 0)
    assert transformed_baseline.device == hsi.image.device

    # Test case 2: Integer baseline
    baseline = 1
    transformed_baseline = explainer_module.validate_and_transform_baseline(baseline, hsi)
    assert transformed_baseline.shape == hsi.image.shape
    assert torch.all(transformed_baseline == 1)
    assert transformed_baseline.device == hsi.image.device

    # Test case 3: Float baseline
    baseline = 0.5
    transformed_baseline = explainer_module.validate_and_transform_baseline(baseline, hsi)
    assert transformed_baseline.shape == hsi.image.shape
    assert torch.all(transformed_baseline == 0.5)
    assert transformed_baseline.device == hsi.image.device

    # Test case 4: Tensor baseline with matching shape
    baseline = torch.rand(3, 224, 224)
    transformed_baseline = explainer_module.validate_and_transform_baseline(baseline, hsi)
    assert transformed_baseline.shape == hsi.image.shape
    assert torch.all(transformed_baseline == baseline)
    assert transformed_baseline.device == hsi.image.device

    # Test case 5: Tensor baseline with mismatching shape
    baseline = torch.rand(3, 224, 225)
    with pytest.raises(ValueError):
        explainer_module.validate_and_transform_baseline(baseline, hsi)


def test_explainer():
    # Create mock objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")

    # Create a Explainer object
    explainer = explainer_module.Explainer(explainable_model)

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
    explainer = explainer_module.Explainer(explainable_model)
    device = "cpu"

    # Call the to method
    explainer.to(device)

    # chained explainer - similar to the NoiseTunnel class
    chained_explainer = explainer_module.Explainer(explainer)

    # chained explainer with chained explainer - should raise ValueError
    with pytest.raises(ValueError):
        explainer_module.Explainer(chained_explainer)

    assert not explainer.has_convergence_delta()

    with pytest.raises(NotImplementedError):
        explainer.compute_convergence_delta()

    assert not explainer.multiplies_by_inputs
