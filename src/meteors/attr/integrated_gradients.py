from __future__ import annotations

from typing import Literal

import torch
from loguru import logger
from captum.attr import IntegratedGradients as CaptumIntegratedGradients

from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes, Explainer

from meteors.attr.explainer import validate_and_transform_baseline


class IntegratedGradients(Explainer):
    def __init__(self, explainable_model: ExplainableModel, multiply_by_inputs: bool = True):
        super().__init__(explainable_model)
        self._attribution_method = CaptumIntegratedGradients(
            explainable_model.forward_func, multiply_by_inputs=multiply_by_inputs
        )

    def attribute(
        self,
        image: Image,
        baseline: int | float | torch.Tensor = None,
        target: int | None = None,
        method: Literal[
            "riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"
        ] = "gausslegendre",
        return_convergence_delta: bool = False,
    ) -> ImageAttributes:
        if self._attribution_method is None:
            raise ValueError("IntegratedGradients explainer is not initialized")

        baseline = validate_and_transform_baseline(baseline, image)

        logger.debug("Applying IntegratedGradients on the image")

        ig_attributions = self._attribution_method.attribute(
            image.image,
            baselines=baseline,
            target=target,
            method=method,
            return_convergence_delta=return_convergence_delta,
        )
        if return_convergence_delta:
            attributions, approximation_error = ig_attributions
        else:
            attributions, approximation_error = ig_attributions, None

        attributes = ImageAttributes(
            image=image,
            attributes=attributions,
            approximation_error=approximation_error,
            attribution_method=self.get_name(),
        )

        return attributes
