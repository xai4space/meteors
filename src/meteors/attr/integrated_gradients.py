from __future__ import annotations

from typing import Literal, Any

import torch
from captum.attr import IntegratedGradients as CaptumIntegratedGradients

from meteors.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, Explainer

from meteors.attr.explainer import validate_and_transform_baseline

from meteors.exceptions import HSIAttributesError


class IntegratedGradients(Explainer):
    """
    IntegratedGradients explainer class for generating attributions using the Integrated Gradients method.
    The Integrated Gradients method is based on the [`captum` implementation](https://captum.ai/api/integrated_gradients.html)
    and is an implementation of an idea coming from the [original paper on Integrated Gradients](https://arxiv.org/pdf/1703.01365),
    where more details about this method can be found.

    Attributes:
        _attribution_method (CaptumIntegratedGradients): The Integrated Gradients method from the `captum` library.
        multiply_by_inputs: Indicates whether to factor model inputs’ multiplier in the final attribution scores.
            In the literature this is also known as local vs global attribution. If inputs’ multiplier isn’t factored
            in, then that type of attribution method is also called local attribution. If it is, then that type of
            attribution method is called global. More detailed can be found in this [paper](https://arxiv.org/abs/1711.06104).
            In case of integrated gradients, if multiply_by_inputs is set to True,
            final sensitivity scores are being multiplied by (inputs - baselines).

    Args:
        explainable_model (ExplainableModel | Explainer): The explainable model to be explained.
    """

    def __init__(self, explainable_model: ExplainableModel, multiply_by_inputs: bool = True):
        super().__init__(explainable_model)
        self.multiply_by_inputs = multiply_by_inputs

        self._attribution_method = CaptumIntegratedGradients(
            explainable_model.forward_func, multiply_by_inputs=self.multiply_by_inputs
        )

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        baseline: int | float | torch.Tensor | list[int | float | torch.Tensor] = None,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: Literal[
            "riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"
        ] = "gausslegendre",
        return_convergence_delta: bool = False,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Integrated Gradients method.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            baseline (int | float | torch.Tensor | list[int | float | torch.Tensor, optional): Baselines define the
                starting point from which integral is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
                    - list of integers, floats or tensors with the same shape as the input tensor, providing a baseline
                        for each input pixel. If the input is a list of HSI objects, the baseline can be a list of
                        tensors with the same shape as the input tensor for each HSI object. Defaults to None.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            n_steps (int, optional): The number of steps to approximate the integral. Default: 50.
            method (Literal["riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"],
                optional): Method for approximating the integral, one of riemann_right, riemann_left, riemann_middle,
                riemann_trapezoid or gausslegendre. Default: gausslegendre if no method is provided.
            return_convergence_delta (bool, optional): Indicates whether to return convergence delta or not.
                If return_convergence_delta is set to True convergence delta will be returned in a tuple following
                attributions. Default: False

        Returns:
            HSIAttributes | list[HSIAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Raises:
            RuntimeError: If the explainer is not initialized.
            HSIAttributesError: If an error occurs during the generation of the attributions.


        Examples:
            >>> integrated_gradients = IntegratedGradients(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = integrated_gradients.attribute(hsi, method="riemann_right", baseline=0.0)
            >>> attributions, approximation_error = integrated_gradients.attribute(hsi, return_convergence_delta=True)
            >>> approximation_error
            0.5
            >>> attributions = integrated_gradients.attribute([hsi, hsi])
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise RuntimeError("IntegratedGradients explainer is not initialized, INITIALIZATION ERROR")

        if not isinstance(hsi, list):
            hsi = [hsi]

        if not all(isinstance(hsi_image, HSI) for hsi_image in hsi):
            raise TypeError("All of the input hyperspectral images must be of type HSI")

        if not isinstance(baseline, list):
            baseline = [baseline] * len(hsi)

        baseline = torch.stack(
            [
                validate_and_transform_baseline(base, hsi_image).to(hsi_image.device)
                for hsi_image, base in zip(hsi, baseline)
            ],
            dim=0,
        )
        input_tensor = torch.stack(
            [hsi_image.get_image().requires_grad_(True).to(hsi_image.device) for hsi_image in hsi], dim=0
        )

        ig_attributions = self._attribution_method.attribute(
            input_tensor,
            baselines=baseline,
            target=target,
            n_steps=n_steps,
            additional_forward_args=additional_forward_args,
            method=method,
            return_convergence_delta=return_convergence_delta,
        )

        if return_convergence_delta:
            attributions, approximation_error = ig_attributions
        else:
            attributions, approximation_error = ig_attributions, [None] * len(hsi)

        try:
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, score=error, attribution_method=self.get_name())
                for hsi_image, attribution, error in zip(hsi, attributions, approximation_error)
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error while creating HSIAttributes: {e}") from e

        return attributes[0] if len(attributes) == 1 else attributes
