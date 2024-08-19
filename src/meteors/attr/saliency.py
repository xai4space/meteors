from __future__ import annotations

from typing import Any

from loguru import logger
from captum.attr import Saliency as CaptumSaliency

from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes

## VALIDATORS


class Saliency:
    def __init__(self, explainable_model: ExplainableModel, multiply_by_inputs: bool = True):
        if not isinstance(explainable_model, ExplainableModel):
            raise TypeError(f"Expected ExplainableModel, but got {type(explainable_model)}")

        logger.debug("Initializing IntegratedGradients explainer on model {explainable_model}")

        self.model = explainable_model
        self._saliency = CaptumSaliency(explainable_model.forward_func)

    def attribute(
        self,
        image: Image,
        target: int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ):
        if self._saliency is None:
            raise ValueError("Saliency explainer is not initialized")

        logger.debug("Applying Saliency on the image")

        saliency_attributions = self._saliency.attribute(
            image, target=target, abs=abs, additional_forward_args=additional_forward_args
        )
        attributes = ImageAttributes(image=image, attributes=saliency_attributions, attribution_method="saliency")

        return attributes
