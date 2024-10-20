from __future__ import annotations

from typing import Any, Callable

import torch
from captum.attr import Saliency as CaptumSaliency

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes

from meteors.attr import Explainer

## VALIDATORS


class Saliency(Explainer):
    """
    Saliency explainer class for generating attributions using the Saliency method.
    This baseline method for computing input attribution calculates gradients with respect to inputs.
    It also has an option to return the absolute value of the gradients, which is the default behaviour.
    Implementation of this method is based on the [`captum` repository](https://captum.ai/api/saliency.html)

    Attributes:
        _attribution_method (CaptumSaliency): The Saliency method from the `captum` library.
    """

    def __init__(
        self,
        explainable_model: ExplainableModel,
    ):
        super().__init__(explainable_model)

        self._attribution_method = CaptumSaliency(explainable_model.forward_func)

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        if self._attribution_method is None:
            raise ValueError("Saliency explainer is not initialized")

        # Ensure that the input tensor requires gradients
        if isinstance(hsi, list):
            input_tensor = torch.stack([hsi_image.get_image().requires_grad_(True) for hsi_image in hsi], dim=0)
        else:
            input_tensor = hsi.get_image().unsqueeze(0).requires_grad_(True)

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumSaliency(adjusted_forward_func)

            saliency_attributions = segmentation_attribution_method.attribute(
                input_tensor, target=target, abs=abs, additional_forward_args=additional_forward_args
            )
        else:
            saliency_attributions = self._attribution_method.attribute(
                input_tensor, target=target, abs=abs, additional_forward_args=additional_forward_args
            )

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, saliency_attributions)
            ]
        else:
            attributes = HSIAttributes(
                hsi=hsi, attributes=saliency_attributions.squeeze(0), attribution_method=self.get_name()
            )
        return attributes
