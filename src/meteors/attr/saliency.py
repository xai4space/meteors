from __future__ import annotations

from typing import Any

from loguru import logger
from captum.attr import Saliency as CaptumSaliency

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes

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
        hsi: HSI,
        target: int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> HSIAttributes:
        if self._attribution_method is None:
            raise ValueError("Saliency explainer is not initialized")

        logger.debug("Applying Saliency on the image")

        saliency_attributions = self._attribution_method.attribute(
            hsi.get_image().unsqueeze(0), target=target, abs=abs, additional_forward_args=additional_forward_args
        )
        attributes = HSIAttributes(
            hsi=hsi, attributes=saliency_attributions.squeeze(0), attribution_method=self.get_name()
        )

        return attributes
