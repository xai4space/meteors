from __future__ import annotations

from typing import Any, Callable

import torch
from captum.attr import Occlusion as CaptumOcclusion

from meteors.utils.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes
from meteors.attr import Explainer
from meteors.attr.explainer import validate_and_transform_baseline


class Occlusion(Explainer):
    """
    Occlusion explainer class for generating attributions using the Occlusion method.
    This attribution method perturbs the input by replacing the contiguous rectangular region
    with a given baseline and computing the difference in output.
    In our case, features are located in multiple regions, and attribution from different hyper-rectangles is averaged.
    The implementation of this method is also based on the [`captum` repository](https://captum.ai/api/occlusion.html).
    More details about this approach can be found in the [original paper](https://arxiv.org/abs/1311.2901)

    Attributes:
        _attribution_method (CaptumOcclusion): The Occlusion method from the `captum` library.

    Args:
        explainable_model (ExplainableModel | Explainer): The explainable model to be explained.
        postprocessing_segmentation_output (Callable[[torch.Tensor], torch.Tensor] | None):
            A segmentation postprocessing function for segmentation problem type. This is required for segmentation
            problem type as attribution methods needs to have 1d output. Defaults to None, which means that the
            attribution method is not used.
    """

    def __init__(
        self,
        explainable_model: ExplainableModel,
        postprocessing_segmentation_output: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(explainable_model, postprocessing_segmentation_output=postprocessing_segmentation_output)

        self._attribution_method = CaptumOcclusion(explainable_model.forward_func)

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int, int, int] = (1, 1, 1),
        strides: int | tuple[int, int, int] = (1, 1, 1),
        baseline: int | float | torch.Tensor = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Occlusion method.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            sliding_window_shapes (int | tuple[int, int, int]):
                The shape of the sliding window. If an integer is provided, it will be used for all dimensions.
                Defaults to (1, 1, 1).
            strides (int | tuple[int, int, int], optional): The stride of the sliding window. Defaults to (1, 1, 1).
                Simply put, the stride is the number of pixels by which the sliding window is moved in each dimension.
            baseline (int | float | torch.Tensor, optional): Baselines define reference value which replaces each
                feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            perturbations_per_eval (int, optional): Allows multiple occlusions to be included in one batch
                (one call to forward_fn). By default, perturbations_per_eval is 1, so each occlusion is processed
                individually. Each forward pass will contain a maximum of perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the available devices, so evaluations on each
                available device contain at most (perturbations_per_eval * #examples) / num_devices samples. When
                working with multiple examples, the number of perturbations per evaluation should be set to at least
                the number of examples. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed attributions.

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = occlusion.attribute(hsi, baseline=0, sliding_window_shapes=(4, 3, 3), strides=(1, 1, 1))
            >>> attributions = occlusion.attribute([hsi, hsi], baseline=0, sliding_window_shapes=(4, 3, 3), strides=(1, 2, 2))
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise ValueError("Occlusion explainer is not initialized")

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

        if isinstance(sliding_window_shapes, int):
            sliding_window_shapes = (sliding_window_shapes, sliding_window_shapes, sliding_window_shapes)
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        assert len(strides) == 3, "Strides must be a tuple of three integers"
        assert len(sliding_window_shapes) == 3, "Sliding window shapes must be a tuple of three integers"

        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
            show_progress=show_progress,
        )

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, occlusion_attributions)
            ]
        else:
            attributes = HSIAttributes(
                hsi=hsi, attributes=occlusion_attributions.squeeze(0), attribution_method=self.get_name()
            )

        return attributes

    def get_spatial_attributes(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int, int] = (1, 1),
        strides: int | tuple[int, int] = 1,
        baseline: int | float | torch.Tensor = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSIAttributes | list[HSIAttributes]:
        """Compute spatial attributions for the input HSI using the Occlusion method. In this case, the sliding window
        is applied to the spatial dimensions only.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            sliding_window_shapes (int | tuple[int, int]): The shape of the sliding window for spatial dimensions.
                If an integer is provided, it will be used for both spatial dimensions. Defaults to (1, 1).
            strides (int | tuple[int, int], optional): The stride of the sliding window for spatial dimensions.
                Defaults to 1. Simply put, the stride is the number of pixels by which the sliding window is moved
                in each spatial dimension.
            baseline (int | float | torch.Tensor, optional): Baselines define reference value which replaces each
                feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            perturbations_per_eval (int, optional): Allows multiple occlusions to be included in one batch
                (one call to forward_fn). By default, perturbations_per_eval is 1, so each occlusion is processed
                individually. Each forward pass will contain a maximum of perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the available devices, so evaluations on each
                available device contain at most (perturbations_per_eval * #examples) / num_devices samples. When
                working with multiple examples, the number of perturbations per evaluation should be set to at least
                the number of examples. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed spatial attributions.

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = occlusion.get_spatial_attributes(hsi, baseline=0, sliding_window_shapes=(3, 3), strides=(1, 1))
            >>> attributions = occlusion.get_spatial_attributes([hsi, hsi], baseline=0, sliding_window_shapes=(3, 3), strides=(2, 2))
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise ValueError("Occlusion explainer is not initialized")

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

        if isinstance(sliding_window_shapes, int):
            sliding_window_shapes = (sliding_window_shapes, sliding_window_shapes)
        if isinstance(strides, int):
            strides = (strides, strides)

        assert len(strides) == 2, "Strides must be a tuple of two integers or a single integer"
        assert (
            len(sliding_window_shapes) == 2
        ), "Sliding window shapes must be a tuple of two integers or a single integer"

        list_sliding_window_shapes = list(sliding_window_shapes)
        list_strides = list(strides)
        if isinstance(hsi, list):
            list_sliding_window_shapes.insert(hsi[0].spectral_axis, hsi[0].image.shape[hsi[0].spectral_axis])
            list_strides.insert(hsi[0].spectral_axis, hsi[0].image.shape[hsi[0].spectral_axis])
        else:
            list_sliding_window_shapes.insert(hsi.spectral_axis, hsi.image.shape[hsi.spectral_axis])
            list_strides.insert(hsi.spectral_axis, hsi.image.shape[hsi.spectral_axis])
        sliding_window_shapes = tuple(list_sliding_window_shapes)  # type: ignore
        strides = tuple(list_strides)  # type: ignore

        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
            show_progress=show_progress,
        )

        spatial_attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            spatial_attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, occlusion_attributions)
            ]
        else:
            spatial_attributes = HSIAttributes(
                hsi=hsi, attributes=occlusion_attributions.squeeze(0), attribution_method=self.get_name()
            )

        return spatial_attributes

    def get_spectral_attributes(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int] = 1,
        strides: int | tuple[int] = 1,
        baseline: int | float | torch.Tensor = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSIAttributes | list[HSIAttributes]:
        """Compute spectral attributions for the input HSI using the Occlusion method. In this case, the sliding window
        is applied to the spectral dimension only.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            sliding_window_shapes (int | tuple[int]): The size of the sliding window for the spectral dimension.
                Defaults to 1.
            strides (int | tuple[int], optional): The stride of the sliding window for the spectral dimension.
                Defaults to 1. Simply put, the stride is the number of pixels by which the sliding window is moved
                in spectral dimension.
            baseline (int | float | torch.Tensor, optional): Baselines define reference value which replaces each
                feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            perturbations_per_eval (int, optional): Allows multiple occlusions to be included in one batch
                (one call to forward_fn). By default, perturbations_per_eval is 1, so each occlusion is processed
                individually. Each forward pass will contain a maximum of perturbations_per_eval * #examples samples.
                For DataParallel models, each batch is split among the available devices, so evaluations on each
                available device contain at most (perturbations_per_eval * #examples) / num_devices samples. When
                working with multiple examples, the number of perturbations per evaluation should be set to at least
                the number of examples. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed spectral attributions.

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((10, 240, 240)), wavelengths=torch.arange(10))
            >>> attributions = occlusion.get_spectral_attributes(hsi, baseline=0, sliding_window_shapes=3, strides=1)
            >>> attributions = occlusion.get_spectral_attributes([hsi, hsi], baseline=0, sliding_window_shapes=3, strides=2)
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise ValueError("Occlusion explainer is not initialized")

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

        if isinstance(sliding_window_shapes, tuple):
            assert (
                len(sliding_window_shapes) == 1
            ), "Sliding window shapes must be a single integer or a tuple of a single integer"
            sliding_window_shapes = sliding_window_shapes[0]
        if isinstance(strides, tuple):
            assert len(strides) == 1, "Strides must be a single integer or a tuple of a single integer"
            strides = strides[0]

        assert isinstance(sliding_window_shapes, int), "Sliding window shapes must be a single integer"
        assert isinstance(strides, int), "Strides must be a single integer"

        if isinstance(hsi, list):
            full_sliding_window_shapes = list(hsi[0].image.shape)
            full_sliding_window_shapes[hsi[0].spectral_axis] = sliding_window_shapes
            full_strides = list(hsi[0].image.shape)
            full_strides[hsi[0].spectral_axis] = strides
        else:
            full_sliding_window_shapes = list(hsi.image.shape)
            full_sliding_window_shapes[hsi.spectral_axis] = sliding_window_shapes
            full_strides = list(hsi.image.shape)
            full_strides[hsi.spectral_axis] = strides

        sliding_window_shapes = tuple(full_sliding_window_shapes)
        strides = tuple(full_strides)

        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
            show_progress=show_progress,
        )

        spectral_attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            spectral_attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, occlusion_attributions)
            ]
        else:
            spectral_attributes = HSIAttributes(
                hsi=hsi, attributes=occlusion_attributions.squeeze(0), attribution_method=self.get_name()
            )

        return spectral_attributes
