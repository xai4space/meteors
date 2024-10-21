from __future__ import annotations

from typing import Literal, Callable, Any

import torch
from captum.attr import IntegratedGradients as CaptumIntegratedGradients

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, Explainer

from meteors.attr.explainer import validate_and_transform_baseline


class IntegratedGradients(Explainer):
    """
    IntegratedGradients explainer class for generating attributions using the Integrated Gradients method.
    The Integrated Gradients method is based on the [`captum` implementation](https://captum.ai/docs/extension/integrated_gradients)
    and is an implementation of an idea coming from the [original paper on Integrated Gradients](https://arxiv.org/pdf/1703.01365),
    where more details about this method can be found.

    Attributes:
        _attribution_method (CaptumIntegratedGradients): The Integrated Gradients method from the `captum` library.
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
        baseline: int | float | torch.Tensor = None,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        method: Literal[
            "riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"
        ] = "gausslegendre",
        return_convergence_delta: bool = False,
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Integrated Gradients method.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            baseline (int | float | torch.Tensor, optional): Baselines define the starting point from which integral
                is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
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
            method (Literal["riemann_right", "riemann_left", "riemann_middle", "riemann_trapezoid", "gausslegendre"],
                optional): Method for approximating the integral, one of riemann_right, riemann_left, riemann_middle,
                riemann_trapezoid or gausslegendre. Default: gausslegendre if no method is provided.
            return_convergence_delta (bool, optional): Indicates whether to return convergence delta or not.
                If return_convergence_delta is set to True convergence delta will be returned in a tuple following
                attributions. Default: False
            postprocessing_segmentation_output (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional):
                A segmentation postprocessing function for segmentation problem type. This is required for segmentation
                problem type as attribution methods needs to have 1d output. Defaults to None, which means that the
                attribution method is not used.

        Returns:
            HSIAttributes | list[HSIAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

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
            raise ValueError("IntegratedGradients explainer is not initialized")

        if isinstance(hsi, list):
            if isinstance(baseline, torch.Tensor) and baseline.ndim == 4:
                baseline = torch.stack(
                    [validate_and_transform_baseline(baseline[i], hsi[i]) for i in range(len(hsi))], dim=0
                )
            else:
                baseline = torch.stack(
                    [validate_and_transform_baseline(baseline, hsi[i]) for i in range(len(hsi))], dim=0
                )
            input_tensor = torch.stack([hsi_image.get_image().requires_grad_(True) for hsi_image in hsi], dim=0)
        else:
            baseline = validate_and_transform_baseline(baseline, hsi).unsqueeze(0)
            input_tensor = hsi.get_image().unsqueeze(0).requires_grad_(True)

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumIntegratedGradients(
                adjusted_forward_func, multiply_by_inputs=self.multiply_by_inputs
            )

            ig_attributions = segmentation_attribution_method.attribute(
                input_tensor,
                baselines=baseline,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
                return_convergence_delta=return_convergence_delta,
            )
        else:
            ig_attributions = self._attribution_method.attribute(
                input_tensor,
                baselines=baseline,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
                return_convergence_delta=return_convergence_delta,
            )

        if return_convergence_delta:
            attributions, approximation_error_tensor = ig_attributions

            if isinstance(hsi, list):
                approximation_error = approximation_error_tensor.reshape(-1).tolist()
                assert len(approximation_error) == len(hsi), "Approximation error should be returned for each HSI"
            else:
                approximation_error = approximation_error_tensor.item()
        else:
            if isinstance(hsi, list):
                attributions, approximation_error = ig_attributions, [None] * len(hsi)
            else:
                attributions, approximation_error = ig_attributions, None

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, score=error, attribution_method=self.get_name())
                for hsi_image, attribution, error in zip(hsi, attributions, approximation_error)
            ]
        else:
            attributes = HSIAttributes(
                hsi=hsi,
                attributes=attributions.squeeze(0),
                score=approximation_error,
                attribution_method=self.get_name(),
            )

        return attributes
