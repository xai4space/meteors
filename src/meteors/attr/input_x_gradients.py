from __future__ import annotations

from typing import Any, Callable


import torch
from captum.attr import InputXGradient as CaptumInputXGradient

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, Explainer


class InputXGradient(Explainer):
    """
    Initializes the InputXGradient explainer. The InputXGradients method is a straightforward approach to
    computing attribution. It simply multiplies the input image with the gradient with respect to the input.
    This method is based on the [`captum` implementation](https://captum.ai/api/input_x_gradient.html)

    Attributes:
        _attribution_method (CaptumIntegratedGradients): The InputXGradient method from the `captum` library.
    """

    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)
        self._attribution_method = CaptumInputXGradient(explainable_model.forward_func)

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        if self._attribution_method is None:
            raise ValueError("InputXGradient explainer is not initialized")

        # Ensure that the input tensor requires gradients
        if isinstance(hsi, list):
            input_tensor = torch.stack([hsi_image.get_image().requires_grad_(True) for hsi_image in hsi], dim=0)
        else:
            input_tensor = hsi.get_image().unsqueeze(0).requires_grad_(True)

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumInputXGradient(adjusted_forward_func)

            gradient_attribution = segmentation_attribution_method.attribute(
                input_tensor, target=target, additional_forward_args=additional_forward_args
            )
        else:
            gradient_attribution = self._attribution_method.attribute(
                input_tensor, target=target, additional_forward_args=additional_forward_args
            )

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, gradient_attribution)
            ]
        else:
            attributes = HSIAttributes(
                hsi=hsi, attributes=gradient_attribution.squeeze(0), attribution_method=self.get_name()
            )

        return attributes
