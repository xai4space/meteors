from __future__ import annotations

from functools import partial
import inspect

from typing import Any, Union

import torch
from torch import Tensor

from captum.attr._utils.attribution import GradientAttribution
from captum.attr import Attribution


class BaseHyperNoiseTunnel(Attribution):
    def __init__(self, model: GradientAttribution):
        self.attribute_main = model.attribute
        sig = inspect.signature(self.attribute_main)
        if "abs" in sig.parameters:
            self.attribute_main = partial(self.attribute_main, abs=False)

    @staticmethod
    def perturb_input(input, baseline, samples):
        # expand input size by the number of drawn samples
        input_expanded_size = (samples,) + (input.shape[0],)
        output_expanded_size = (samples,) + input.shape

        mask_inputs = torch.rand(input_expanded_size, device=input.device).uniform_(0, 1)
        masks = torch.bernoulli(mask_inputs).bool()
        masks_expanded = masks.unsqueeze(-1).unsqueeze(-1).expand(*output_expanded_size)
        input = input.unsqueeze(0).repeat_interleave(samples, dim=0)
        baseline = baseline.unsqueeze(0).repeat_interleave(samples, dim=0)
        assert input.shape == baseline.shape
        assert masks_expanded.shape == baseline.shape
        result = torch.where(masks_expanded, baseline, input)
        result.requires_grad_(True)
        return result

    def attribute(
        self,
        inputs: Tensor,
        baselines: Union[Tensor, int, float],
        target: int | None = None,
        additional_forward_args: Any = None,
        n_samples: int = 5,
        steps_per_batch: int = 1,
        method: str = "smoothgrad",
    ) -> Tensor:
        if method not in ["smoothgrad", "smoothgrad_sq", "vargrad"]:
            raise ValueError("Method must be one of 'smoothgrad', 'smoothgrad_sq', 'vargrad'")

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        elif inputs.dim() != 4:
            raise ValueError("Input must be in the format (N, C, H, W)")

        if isinstance(baselines, (int, float)):
            baselines = torch.zeros_like(inputs, device=inputs.device) + baselines
        elif baselines.dim() == 4:
            baselines = baselines.squeeze(0)
        elif baselines.dim() != 3:
            raise ValueError("Baselines must be in the format (C, H, W)")

        attributions = torch.empty((n_samples,) + inputs.shape, device=inputs.device)

        for batch in range(0, inputs.shape[0]):
            input = inputs[batch]
            perturbed_input = BaseHyperNoiseTunnel.perturb_input(input, baselines, n_samples)
            for i in range(0, n_samples, steps_per_batch):
                perturbed_batch = perturbed_input[i : i + steps_per_batch]
                attributions[i : i + steps_per_batch, batch] = self.attribute_main(
                    perturbed_batch, target=target, additional_forward_args=additional_forward_args
                )
            else:
                steps_left = n_samples % steps_per_batch
                if steps_left:
                    perturbed_batch = perturbed_input[-steps_left:]
                    attributions[-steps_left:, batch] = self.attribute_main(
                        perturbed_batch, target=target, additional_forward_args=additional_forward_args
                    )

        if method == "smoothgrad":
            return attributions.mean(dim=0)
        elif method == "smoothgrad_sq":
            return (attributions**2).mean(dim=0)
        else:
            return (attributions**2 - attributions.mean(dim=0) ** 2).mean(dim=0)


class HyperNoiseTunnel:
    def __init__(self, attribution_method):
        self._attribution_method = BaseHyperNoiseTunnel(attribution_method._attribution_method)

    def attribute(self, image, target):
        return self._attribution_method.attribute(image, target=target)
