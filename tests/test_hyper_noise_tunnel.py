import torch
import torch.nn as nn
import torch.nn.functional as F
from meteors.attr import HyperNoiseTunnel, IntegratedGradients
from meteors.utils.models.models import ExplainableModel

import meteors as mt


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(F.relu(input)).unsqueeze(0)


class ExplainableToyModel(ExplainableModel):
    def __init__(self):
        super().__init__(problem_type="regression", forward_func=ToyModel())


def test_hyper_noise_tunnel():
    model = ExplainableToyModel()

    ig = IntegratedGradients(model)
    hyper_noise_tunnel = HyperNoiseTunnel(ig)

    torch.manual_seed(0)

    tensor_image = torch.rand((5, 5, 5))
    image = mt.Image(image=tensor_image, wavelengths=[1, 2, 3, 4, 5], orientation=("C", "H", "W"))

    attributes = hyper_noise_tunnel.attribute(image, n_samples=1)

    assert attributes is not None
    assert attributes.attributes.shape == (5, 5, 5)
