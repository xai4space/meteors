from __future__ import annotations

from typing import Any

from loguru import logger
from captum.attr import Saliency as CaptumSaliency

from meteors.utils.models import ExplainableModel
from meteors import Image
from meteors.attr import ImageAttributes

from meteors.attr import Explainer

## VALIDATORS


class Saliency(Explainer):
    def __init__(
        self,
        explainable_model: ExplainableModel,
    ):
        super().__init__(explainable_model)

        self._attribution_method = CaptumSaliency(explainable_model.forward_func)

    def attribute(
        self,
        image: Image,
        target: int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> ImageAttributes:
        if self._attribution_method is None:
            raise ValueError("Saliency explainer is not initialized")

        logger.debug("Applying Saliency on the image")

        saliency_attributions = self._attribution_method.attribute(
            image.image, target=target, abs=abs, additional_forward_args=additional_forward_args
        )
        attributes = ImageAttributes(image=image, attributes=saliency_attributions, attribution_method="saliency")

        return attributes
