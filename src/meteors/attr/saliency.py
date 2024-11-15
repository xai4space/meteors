from __future__ import annotations

from typing import Any

import torch
from captum.attr import Saliency as CaptumSaliency

from meteors.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, Explainer

from meteors.exceptions import HSIAttributesError

## VALIDATORS


class Saliency(Explainer):
    """
    Saliency explainer class for generating attributions using the Saliency method.
    This baseline method for computing input attribution calculates gradients with respect to inputs.
    It also has an option to return the absolute value of the gradients, which is the default behaviour.
    Implementation of this method is based on the [`captum` repository](https://captum.ai/api/saliency.html)

    Attributes:
        _attribution_method (CaptumSaliency): The Saliency method from the `captum` library.

    Args:
        explainable_model (ExplainableModel | Explainer): The explainable model to be explained.
    """

    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)

        self._attribution_method = CaptumSaliency(explainable_model.forward_func)

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Saliency method.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            abs (bool, optional): Returns absolute value of gradients if set to True,
                otherwise returns the (signed) gradients if False. Default: True
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None

        Returns:
            HSIAttributes | list[HSIAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Raises:
            RuntimeError: If the explainer is not initialized.
            HSIAttributesError: If an error occurs during the generation of the attributions

        Examples:
            >>> saliency = Saliency(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = saliency.attribute(hsi)
            >>> attributions = saliency.attribute([hsi, hsi])
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise RuntimeError("Saliency explainer is not initialized, INITIALIZATION ERROR")

        if not isinstance(hsi, list):
            hsi = [hsi]

        if not all(isinstance(hsi_image, HSI) for hsi_image in hsi):
            raise TypeError("All of the input hyperspectral images must be of type HSI")

        input_tensor = torch.stack(
            [hsi_image.get_image().requires_grad_(True).to(hsi_image.device) for hsi_image in hsi], dim=0
        )

        saliency_attributions = self._attribution_method.attribute(
            input_tensor, target=target, abs=abs, additional_forward_args=additional_forward_args
        )

        try:
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, saliency_attributions)
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error in generating Saliency attributions: {e}") from e

        return attributes[0] if len(attributes) == 1 else attributes
