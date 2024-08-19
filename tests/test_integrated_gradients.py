import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.utils.models import ExplainableModel
from meteors.attr.integrated_gradients import IntegratedGradients
from meteors import Image


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input))


class ExplainableToyModel(ExplainableModel):
    def __init__(self):
        super().__init__(problem_type="regression", forward_func=ToyModel())


def test_integrated_gradients():
    toy_model = ExplainableToyModel()
    ig = IntegratedGradients(toy_model)

    assert ig is not None
    assert ig._ig is not None

    image = Image(image=torch.rand(1, 3, 224, 224))
    attributions = ig.attribute(image)
    assert attributions.attributes.shape == image.image.shape
