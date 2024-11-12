from __future__ import annotations

from typing import Any

import itertools
import torch
from captum.attr import Occlusion as CaptumOcclusion

from meteors.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes, HSISpatialAttributes, HSISpectralAttributes
from meteors.attr import Explainer
from meteors.attr.explainer import validate_and_transform_baseline


from meteors.exceptions import HSIAttributesError


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

    def __init__(self, explainable_model: ExplainableModel):
        super().__init__(explainable_model)

        self._attribution_method = CaptumOcclusion(explainable_model.forward_func)

    @staticmethod
    def _create_segmentation_mask(
        input_shape: tuple[int, int, int], sliding_window_shapes: tuple[int, int, int], strides: tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Create a binary segmentation mask based on sliding windows.

        Args:
            input_shape (Tuple[int, int, int]): Shape of the input tensor (e.g., (H, W, C))
            sliding_window_shapes (Tuple[int, int, int]): Shape of the sliding window (e.g., (h, w, c))
            strides (Tuple[int, int, int]): Strides for the sliding window (e.g., (s_h, s_w, s_c))

        Returns:
            torch.Tensor: Binary mask tensor with ones where windows are placed
        """
        # Initialize empty mask
        mask = torch.zeros(input_shape, dtype=torch.int32)

        # Calculate number of windows in each dimension
        windows = []
        for dim_size, window_size, stride in zip(input_shape, sliding_window_shapes, strides):
            if stride == 0:
                raise ValueError("Stride cannot be zero.")
            n_windows = dim_size // stride if (dim_size - window_size) % stride == 0 else dim_size // stride + 1
            # 1 + (dim_size - window_size) // stride
            windows.append(n_windows)

        # Generate all possible indices using itertools.product
        for i, indices in enumerate(itertools.product(*[range(w) for w in windows])):
            # Calculate start position for each dimension
            starts = [idx * stride for idx, stride in zip(indices, strides)]

            # Calculate end position for each dimension
            ends = [start + window for start, window in zip(starts, sliding_window_shapes)]

            # Create slice objects for each dimension
            slices = tuple(slice(start, end) for start, end in zip(starts, ends))

            # Mark window position in mask
            mask[slices] = i + 1

        return mask

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int, int, int] = (1, 1, 1),
        strides: int | tuple[int, int, int] = (1, 1, 1),
        baseline: int | float | torch.Tensor | list[int | float | torch.Tensor] = None,
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
            baseline (int | float | torch.Tensor | list[int | float | torch.Tensor], optional): Baselines define
                reference value which replaces each feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
                    - list of integers, floats or tensors with the same shape as the input tensor, providing a baseline
                        for each input pixel. If the input is a list of HSI objects, the baseline can be a list of
                        tensors with the same shape as the input tensor for each HSI object. Defaults to None.
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
            HSIAttributes: The computed attributions for the input hyperspectral image(s). if a list of HSI objects
                is provided, the attributions are computed for each HSI object in the list.

        Raises:
            RuntimeError: If the explainer is not initialized.
            ValueError: If the sliding window shapes or strides are not a tuple of three integers.
            HSIAttributesError: If an error occurs during the generation of the attributions.

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = occlusion.attribute(hsi, baseline=0, sliding_window_shapes=(4, 3, 3), strides=(1, 1, 1))
            >>> attributions = occlusion.attribute([hsi, hsi], baseline=0, sliding_window_shapes=(4, 3, 3), strides=(1, 2, 2))
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise RuntimeError("Occlusion explainer is not initialized, INITIALIZATION ERROR")

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

        if isinstance(sliding_window_shapes, int):
            sliding_window_shapes = (sliding_window_shapes, sliding_window_shapes, sliding_window_shapes)
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        if len(strides) != 3:
            raise ValueError("Strides must be a tuple of three integers")
        if len(sliding_window_shapes) != 3:
            raise ValueError("Sliding window shapes must be a tuple of three integers")

        assert len(sliding_window_shapes) == len(strides) == 3
        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi)),
            show_progress=show_progress,
        )

        try:
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, occlusion_attributions)
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error in generating Occlusion attributions: {e}") from e

        return attributes[0] if len(attributes) == 1 else attributes

    def get_spatial_attributes(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int, int] = (1, 1),
        strides: int | tuple[int, int] = 1,
        baseline: int | float | torch.Tensor | list[int | float | torch.Tensor] = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSISpatialAttributes | list[HSISpatialAttributes]:
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
            baseline (int | float | torch.Tensor | list[int | float | torch.Tensor], optional): Baselines define
                reference value which replaces each feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
                    - list of integers, floats or tensors with the same shape as the input tensor, providing a baseline
                      for each input pixel. If the input is a list of HSI objects, the baseline can be a list of
                      tensors with the same shape as the input tensor for each HSI object. Defaults to None.
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
            HSISpatialAttributes | list[HSISpatialAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Raises:
            RuntimeError: If the explainer is not initialized.
            ValueError: If the sliding window shapes or strides are not a tuple of two integers.
            HSIAttributesError: If an error occurs during the generation of the attributions

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = occlusion.get_spatial_attributes(hsi, baseline=0, sliding_window_shapes=(3, 3), strides=(1, 1))
            >>> attributions = occlusion.get_spatial_attributes([hsi, hsi], baseline=0, sliding_window_shapes=(3, 3), strides=(2, 2))
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise RuntimeError("Occlusion explainer is not initialized, INITIALIZATION ERROR")

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

        if isinstance(sliding_window_shapes, int):
            sliding_window_shapes = (sliding_window_shapes, sliding_window_shapes)
        if isinstance(strides, int):
            strides = (strides, strides)

        if len(strides) != 2:
            raise ValueError("Strides must be a tuple of two integers")
        if len(sliding_window_shapes) != 2:
            raise ValueError("Sliding window shapes must be a tuple of two integers")

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

        assert len(sliding_window_shapes) == len(strides) == 3
        segment_mask = [
            self._create_segmentation_mask(hsi_image.image.shape, sliding_window_shapes, strides) for hsi_image in hsi
        ]

        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi)),
            show_progress=show_progress,
        )

        try:
            spatial_attributes = [
                HSISpatialAttributes(
                    hsi=hsi_image, attributes=attribution, attribution_method=self.get_name(), mask=mask
                )
                for hsi_image, attribution, mask in zip(hsi, occlusion_attributions, segment_mask)
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error in generating Occlusion attributions: {e}") from e

        return spatial_attributes[0] if len(spatial_attributes) == 1 else spatial_attributes

    def get_spectral_attributes(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        sliding_window_shapes: int | tuple[int] = 1,
        strides: int | tuple[int] = 1,
        baseline: int | float | torch.Tensor | list[int | float | torch.Tensor] = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> HSISpectralAttributes | list[HSISpectralAttributes]:
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
            baseline (int | float | torch.Tensor | list[int | float | torch.Tensor], optional): Baselines define
                reference value which replaces each feature when occluded is computed and can be provided as:
                    - integer or float representing a constant value used as the baseline for all input pixels.
                    - tensor with the same shape as the input tensor, providing a baseline for each input pixel.
                        if the input is a list of HSI objects, the baseline can be a tensor with the same shape as
                        the input tensor for each HSI object.
                    - list of integers, floats or tensors with the same shape as the input tensor, providing a baseline
                      for each input pixel. If the input is a list of HSI objects, the baseline can be a list of
                      tensors with the same shape as the input tensor for each HSI object. Defaults to None.
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
            HSISpectralAttributes | list[HSISpectralAttributes]: The computed attributions for the input hyperspectral
                image(s). if a list of HSI objects is provided, the attributions are computed for each HSI object in
                the list.

        Raises:
            RuntimeError: If the explainer is not initialized.
            ValueError: If the sliding window shapes or strides are not a tuple of a single integer.
            TypeError: If the sliding window shapes or strides are not a single integer.
            HSIAttributesError: If an error occurs during the generation of the attributions

        Example:
            >>> occlusion = Occlusion(explainable_model)
            >>> hsi = HSI(image=torch.ones((10, 240, 240)), wavelengths=torch.arange(10))
            >>> attributions = occlusion.get_spectral_attributes(hsi, baseline=0, sliding_window_shapes=3, strides=1)
            >>> attributions = occlusion.get_spectral_attributes([hsi, hsi], baseline=0, sliding_window_shapes=3, strides=2)
            >>> len(attributions)
            2
        """
        if self._attribution_method is None:
            raise RuntimeError("Occlusion explainer is not initialized, INITIALIZATION ERROR")

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

        if isinstance(sliding_window_shapes, tuple):
            if len(sliding_window_shapes) != 1:
                raise ValueError("Sliding window shapes must be a single integer or a tuple of a single integer")
            sliding_window_shapes = sliding_window_shapes[0]
        if isinstance(strides, tuple):
            if len(strides) != 1:
                raise ValueError("Strides must be a single integer or a tuple of a single integer")
            strides = strides[0]

        if not isinstance(sliding_window_shapes, int):
            raise TypeError("Sliding window shapes must be a single integer")
        if not isinstance(strides, int):
            raise TypeError("Strides must be a single integer")

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

        assert len(sliding_window_shapes) == len(strides) == 3
        band_mask = [
            self._create_segmentation_mask(hsi_image.image.shape, sliding_window_shapes, strides) for hsi_image in hsi
        ]
        band_names = {str(ui.item()): ui.item() for ui in torch.unique(band_mask[0])}

        occlusion_attributions = self._attribution_method.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=min(perturbations_per_eval, len(hsi)),
            show_progress=show_progress,
        )

        try:
            spectral_attributes = [
                HSISpectralAttributes(
                    hsi=hsi_image,
                    attributes=attribution,
                    attribution_method=self.get_name(),
                    mask=mask,
                    band_names=band_names,
                )
                for hsi_image, attribution, mask in zip(hsi, occlusion_attributions, band_mask)
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error in generating Occlusion attributions: {e}") from e

        return spectral_attributes[0] if len(spectral_attributes) == 1 else spectral_attributes
