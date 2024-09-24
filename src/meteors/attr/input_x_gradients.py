from __future__ import annotations

from typing import Any


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
        hsi: HSI,
        target: int | None = None,
        additional_forward_args: Any = None,
    ) -> HSIAttributes:
        if self._attribution_method is None:
            raise ValueError("InputXGradient explainer is not initialized")

        gradient_attribution = self._attribution_method.attribute(
            hsi.get_image().unsqueeze(0), target=target, additional_forward_args=additional_forward_args
        )
        attributes = HSIAttributes(
            hsi=hsi, attributes=gradient_attribution.squeeze(0), attribution_method=self.get_name()
        )

        return attributes
