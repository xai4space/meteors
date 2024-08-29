from __future__ import annotations

from typing import Any

from loguru import logger

from captum.attr import InputXGradient as CaptumInputXGradient
from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes, Explainer


class InputXGradient(Explainer):
    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)
        self._inputxgradients = CaptumInputXGradient(explainable_model.forward_func)

    def attribute(
        self,
        image: Image,
        target: int | None = None,
        additional_forward_args: Any = None,
    ):
        if self._inputxgradients is None:
            raise ValueError("InputXGradient explainer is not initialized")

        logger.debug("Applying InputXGradient on the image")

        gradient_attribution = self._inputxgradients.attribute(
            image, target=target, additional_forward_args=additional_forward_args
        )
        attributes = ImageAttributes(
            image=image, attributes=gradient_attribution, attribution_method="input x gradient"
        )

        return attributes
