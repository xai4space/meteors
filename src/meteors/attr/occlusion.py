from __future__ import annotations

from typing import Any

from loguru import logger
import torch
from captum.attr import Occlusion as CaptumOcclusion

from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes
from meteors.attr import Explainer

from .integrated_gradients import validate_and_transform_baseline

## VALIDATORS


class Occlusion(Explainer):
    def __init__(self, explainable_model: ExplainableModel, multiply_by_inputs: bool = True):
        super().__init__(explainable_model)
        self._occlusion = CaptumOcclusion(explainable_model.forward_func)

    def attribute(
        self,
        image: Image,
        sliding_window_shapes,
        strides=None,  # TODO add default value
        baseline: int | float | torch.Tensor = None,
        target: int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ):
        if self._occlusion is None:
            raise ValueError("Occlusion explainer is not initialized")

        baseline = validate_and_transform_baseline(baseline, image)

        logger.debug("Applying Occlusion on the image")

        occlusion_attributions = self._occlusion.attribute(
            image,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baseline=baseline,
            target=target,
            abs=abs,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )
        attributes = ImageAttributes(image=image, attributes=occlusion_attributions, attribution_method="occlusion")

        return attributes
