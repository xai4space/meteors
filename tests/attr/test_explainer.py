import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.models import ExplainableModel
import meteors.attr.explainer as explainer_module
from meteors import HSI
from meteors.exceptions import ShapeMismatchError

import pytest


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


@pytest.fixture
def explainable_toy_model():
    return ExplainableModel(problem_type="regression", forward_func=ToyModel())


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
    with pytest.raises(ShapeMismatchError):
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


def test_explainer_segmentation():
    # Create objects for ExplainableModel and InterpretableModel
    with pytest.raises(ValueError):
        explainable_model = ExplainableModel(forward_func=lambda x: x + 1, problem_type="segmentation")

    # Postprocessing function
    def postprocessing_segmentation_output(x):
        return x.mean(dim=(1, 2, 3))

    # Create objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(
        forward_func=lambda x: x + 1,
        problem_type="segmentation",
        postprocessing_output=postprocessing_segmentation_output,
    )
    assert explainable_model.postprocessing_segmentation_output

    # Test output
    input = torch.rand(1, 3, 224, 224)
    output = explainable_model.forward_func(input)
    assert output.shape == (1,)

    # Test postprocessing function with non-segmentation problem type
    explainable_model = ExplainableModel(
        forward_func=lambda x: x + 1,
        problem_type="regression",
        postprocessing_output=postprocessing_segmentation_output,
    )

    # Test output
    input = torch.rand(1, 3, 224, 224)
    output = explainable_model.forward_func(input)
    assert output.shape == (1,)
