from __future__ import annotations

from typing import Any

from loguru import logger

from captum.attr import InputXGradient as CaptumInputXGradient
from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes


class InputXGradient:
    def __init__(self, explainable_model: ExplainableModel):
        if not isinstance(explainable_model, ExplainableModel):
            raise TypeError(f"Expected ExplainableModel, but got {type(explainable_model)}")

        logger.debug("Initializing InputXGradient explainer on model {explainable_model}")

        self.model = explainable_model
        self._saliency = CaptumInputXGradient(explainable_model.forward_func)

    def attribute(
        self,
        image: Image,
        target: int | None = None,
        additional_forward_args: Any = None,
    ):
        if self._saliency is None:
            raise ValueError("InputXGradient explainer is not initialized")

        logger.debug("Applying InputXGradient on the image")

        gradient_attribution = self._saliency.attribute(
            image, target=target, additional_forward_args=additional_forward_args
        )
        attributes = ImageAttributes(
            image=image, attributes=gradient_attribution, attribution_method="input x gradient"
        )

        return attributes
