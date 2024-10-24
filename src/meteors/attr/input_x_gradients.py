from __future__ import annotations

from typing import Any


from captum.attr import InputXGradient as CaptumInputXGradient
from meteors.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, Explainer
from meteors.exceptions import HSIAttributesError


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
        """
        Raises:
            RuntimeError: If the explainer is not initialized.
            HSIAttributesError: If an error occurs during the generation of the attributions.
        """
        if self._attribution_method is None:
            raise RuntimeError("InputXGradient explainer is not initialized, INITIALIZATION ERROR")

        gradient_attribution = self._attribution_method.attribute(
            hsi.get_image().unsqueeze(0), target=target, additional_forward_args=additional_forward_args
        )
        try:
            attributes = HSIAttributes(
                hsi=hsi, attributes=gradient_attribution.squeeze(0), attribution_method=self.get_name()
            )
        except Exception as e:
            raise HSIAttributesError(f"Error in generating InputXGradient attributions: {e}") from e

        return attributes
