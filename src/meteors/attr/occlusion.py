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
    """

    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)
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
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """Compute attributions for the input HSI using the Occlusion method.

        Args:
            hsi (HSI): The input hyperspectral image.
            target (int | None, optional): Target for attribution. Defaults to None.
            sliding_window_shapes (int | tuple[int, int, int]):
                The shape of the sliding window. If an integer is provided, it will be used for all dimensions.
                Defaults to (1, 1, 1).
            strides (int | tuple[int, int, int], optional): The stride of the sliding window. Defaults to (1, 1, 1).
            baseline (int | float | torch.Tensor, optional): The baseline value(s) for the input. Defaults to None.
            additional_forward_args (Any, optional):
                Additional arguments to be passed to the forward function. Defaults to None.
            perturbations_per_eval (int, optional):
                Number of perturbations to be evaluated per call to forward function. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed attributions.

        Example:
            >>> model = lambda x: torch.sum(x, dim=1)
            >>> explainable_model = ExplainableModel(model, problem_type="regression")
            >>> hsi = HSI(image=torch.rand(10, 20, 30), wavelengths=torch.arange(10))
            >>> occlusion = Occlusion(explainable_model)
            >>> result = occlusion.attribute(hsi, sliding_window_shapes=(3, 3, 3), baseline=0)
            >>> result.attributes.shape
            torch.Size([10, 20, 30])
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

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumOcclusion(adjusted_forward_func)

            occlusion_attributions = segmentation_attribution_method.attribute(
                input_tensor,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                target=target,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
                show_progress=show_progress,
            )
        else:
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
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """Compute spatial attributions for the input HSI using the Occlusion method. In this case, the sliding window
        is applied to the spatial dimensions only.

        Args:
            hsi (HSI): The input hyperspectral image.
            target (int | None, optional): Target for attribution. Defaults to None.
            sliding_window_shapes (int | tuple[int, int]): The shape of the sliding window for spatial dimensions.
                If an integer is provided, it will be used for both spatial dimensions. Defaults to (1, 1).
            strides (int | tuple[int, int], optional): The stride of the sliding window for spatial dimensions.
                Defaults to 1.
            baseline (int | float | torch.Tensor, optional): The baseline value(s) for the input. Defaults to None.
            additional_forward_args (Any, optional): Additional arguments to be passed to the forward function.
                Defaults to None.
            perturbations_per_eval (int, optional):
                Number of perturbations to be evaluated per call to forward function. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed spatial attributions.

        Example:
            >>> model = lambda x: torch.sum(x, dim=1)
            >>> explainable_model = ExplainableModel(model, problem_type="regression")
            >>> hsi = HSI(image=torch.rand(10, 20, 30), wavelengths=torch.arange(10))
            >>> occlusion = Occlusion(explainable_model)
            >>> result = occlusion.get_spatial_attributes(hsi, sliding_window_shapes=(3, 3), baseline=0)
            >>> result.attributes.shape
            torch.Size([10, 20, 30])
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

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumOcclusion(adjusted_forward_func)

            occlusion_attributions = segmentation_attribution_method.attribute(
                input_tensor,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                target=target,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
                show_progress=show_progress,
            )
        else:
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
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """Compute spectral attributions for the input HSI using the Occlusion method. In this case, the sliding window
        is applied to the spectral dimension only.

        Args:
            hsi (HSI): The input hyperspectral image.
            target (int | None, optional): Target for attribution. Defaults to None.
            sliding_window_shapes (int | tuple[int]): The size of the sliding window for the spectral dimension.
                Defaults to 1.
            strides (int | tuple[int], optional): The stride of the sliding window for the spectral dimension. Defaults to 1.
            baseline (int | float | torch.Tensor, optional): The baseline value(s) for the input. Defaults to None.
            additional_forward_args (Any, optional): Additional arguments to be passed to the forward function.
                Defaults to None.
            perturbations_per_eval (int, optional):
                Number of perturbations to be evaluated per call to forward function. Defaults to 1.
            show_progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
            HSIAttributes: An object containing the computed spectral attributions.

        Example:
            >>> model = lambda x: torch.sum(x, dim=1)
            >>> explainable_model = ExplainableModel(model, problem_type="regression")
            >>> hsi = HSI(image=torch.rand(10, 20, 30), wavelengths=torch.arange(10))
            >>> occlusion = Occlusion(explainable_model)
            >>> result = occlusion.get_spectral_attributes(hsi, sliding_window_shapes=3, baseline=0)
            >>> result.attributes.shape
            torch.Size([10, 20, 30])
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

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = CaptumOcclusion(adjusted_forward_func)

            occlusion_attributions = segmentation_attribution_method.attribute(
                input_tensor,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                target=target,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=min(perturbations_per_eval, len(hsi) if isinstance(hsi, list) else 1),
                show_progress=show_progress,
            )
        else:
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
