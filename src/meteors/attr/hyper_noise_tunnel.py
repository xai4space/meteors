from __future__ import annotations

from functools import partial
import inspect

from typing import Any, Union, Literal

from meteors.attr import Explainer, ImageAttributes
from meteors.attr.explainer import validate_and_transform_baseline

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


class HyperNoiseTunnel(Explainer):
    def __init__(self, attribution_method):
        super().__init__(attribution_method)
        if not isinstance(attribution_method, Explainer):
            raise TypeError(f"Expected Explainer as attribution_method, but got {type(attribution_method)}")
        if not attribution_method._attribution_method:
            raise ValueError("Attribution method is not initialized")
        self._attribution_method: Attribution = BaseHyperNoiseTunnel(attribution_method._attribution_method)

    def attribute(
        self,
        image,
        baselines=None,
        target=None,
        n_samples: int = 5,
        steps_per_batch: int = 1,
        method: Literal["smoothgrad", "smoothgrad_sq", "vargrad"] = "smoothgrad",
    ):
        baselines = validate_and_transform_baseline(baselines, image)

        attributes = self._attribution_method.attribute(
            image.image,
            baselines=baselines,
            target=target,
            n_samples=n_samples,
            steps_per_batch=steps_per_batch,
            method=method,
        )
        attributes = attributes.squeeze(0)

        image_attributes = ImageAttributes(image=image, attributes=attributes, attribution_method=self.get_name())
        return image_attributes
