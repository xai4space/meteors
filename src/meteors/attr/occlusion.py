from __future__ import annotations

from typing import Any

import torch
from captum.attr import Occlusion as CaptumOcclusion

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes
from meteors.attr import Explainer
from meteors.attr.explainer import validate_and_transform_baseline


class Occlusion(Explainer):
    """
    Occlusion explainer class for generating attributions using the Occlusion method.
    This attribution method perturbs the input by replacing the contiguous rectangular region
    with a given baseline and computing the difference in output.
    In our case, features are located in multiple regions, and attribution from different hyper-rectangles is averaged.
    The implementation of this method is also based on the [`captum` repository](https://captum.ai/api/occlusion.html).
    More details about this approach can be found in the [original paper](https://arxiv.org/abs/1311.2901)

    Attributes:
        _attribution_method (CaptumOcclusion): The Occlusion method from the `captum` library.
    """

    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)
        self._attribution_method = CaptumOcclusion(explainable_model.forward_func)

    def attribute(
        self,
        hsi: HSI,
        sliding_window_shapes,
        strides=None,  # TODO add default value
        target: int | None = None,
        baseline: int | float | torch.Tensor = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSIAttributes:
        if self._attribution_method is None:
            raise ValueError("Occlusion explainer is not initialized")

        baseline = validate_and_transform_baseline(baseline, hsi)

        occlusion_attributions = self._attribution_method.attribute(
            hsi.get_image().unsqueeze(0),
            sliding_window_shapes=(1,)
            + sliding_window_shapes,  # I'am not sure about this scaling method - need to check how exactly occlusion modifies the image shape
            strides=(1,) + strides,
            target=target,
            baselines=baseline.unsqueeze(0),
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
        occlusion_attributions = occlusion_attributions.squeeze(0)
        attributes = HSIAttributes(hsi=hsi, attributes=occlusion_attributions, attribution_method=self.get_name())

        return attributes
