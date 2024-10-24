from __future__ import annotations
from typing import Any, Literal, Callable

from functools import partial
import warnings
import inspect
from enum import Enum
from copy import deepcopy

from loguru import logger

import torch
from captum.attr._utils.attribution import GradientAttribution
from captum.attr import Attribution

from meteors.attr import Explainer, HSIAttributes
from meteors.attr.explainer import validate_and_transform_baseline, validate_attribution_method_initialization
from meteors import HSI


class NoiseTunnelType(Enum):
    smoothgrad = "smoothgrad"
    smoothgrad_sq = "smoothgrad_sq"
    vargrad = "vargrad"


def torch_random_choice(n: int, k: int, n_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Randomly selects `k` elements from the range [0, n) without replacement.

    Args:
        n (int): The range of the selection.
        k (int): The number of elements to select.
        n_samples (int): The number of samples to be drawn.
        device (torch.device | str | None): The device to which the tensor will be moved.
    Returns:
        torch.Tensor: A tensor of shape (n_samples,n) containing True for the selected elements and False for the rest. Each row contains k True values.
    """
    if k > n:
        raise ValueError(f"Cannot select {k} elements from the range [0, {n})")
    if k == n:
        return torch.ones((n_samples, n), device=device).bool()
    result = torch.zeros((n_samples, n), device=device).bool()
    for i in range(n_samples):
        result[i, torch.randperm(n)[:k]] = True
    return result


class BaseNoiseTunnel(Attribution):
    """
    Compute the attribution of the given inputs using the noise tunnel method.
    This class performs the attribution on the tensor input and is inspired by the captum library.

    Except for the standard perturbation using the random gaussian noise applied for the image, this class also supports the custom perturbation that masks specified bands in the input tensor.
    """

    def __init__(self, model: GradientAttribution, perturbation_method: Literal["gaussian", "mask"] = "gaussian"):
        """_summary_

        Args:
            model (GradientAttribution): a gradient attribution method that will be used to compute the attributions.
            perturbation_method (Literal[&quot;gaussian&quot;, &quot;mask&quot;], optional): a type of perturbation method to be used. If `gaussian` using the standard perturbation method, if `mask` using perturbation method designed for hyperspectral images that masks random bands.
            Defaults to "gaussian".

        Raises:
            ValueError: _description_
        """
        self.attribute_model = model
        self.attribute_main = model.attribute
        sig = inspect.signature(self.attribute_main)
        if "abs" in sig.parameters:
            self.attribute_main = partial(self.attribute_main, abs=False)

        self.perturbation_method = perturbation_method
        if self.perturbation_method not in ["gaussian", "mask"]:
            raise ValueError(f"Noise tunnel with the {perturbation_method} perturbation method is not supported")

    @staticmethod
    def gaussian_perturb_input(
        input, n_samples: int = 1, perturbation_axis: None | tuple[int | slice] = None, stdevs: float = 1
    ) -> torch.Tensor:
        """The perturbation function used in the noise tunnel. It randomly adds noise to the input tensor from a normal
        distribution with a given standard deviation. The noise is added to the selected bands (channels) of the input
        tensor. The bands to be perturbed are selected based on the `perturbation_axis` parameter.

        Args:
            input (torch.Tensor): An input tensor to be perturbed. It should have the shape (C, H, W).
            n_samples (int): A number of samples to be drawn - number of perturbed inputs to be generated.
            perturbation_axis (None | tuple[int | slice]): The indices of the bands to be perturbed.
                If set to None, all bands are perturbed. Defaults to None.
            stdevs (float): The standard deviation of gaussian noise with zero mean that is added to each input
                in the batch. Defaults to 1.0.

        Returns:
            torch.Tensor: A perturbed tensor, which contains `n_samples` perturbed inputs.
        """
        if n_samples < 1:
            raise ValueError("Number of perturbated samples to be generated must be greater than 0")

        # the perturbation
        perturbed_input = input.clone().unsqueeze(0)
        # repeat the perturbed_input on the first dimension n_samples times
        perturbed_input = perturbed_input.repeat_interleave(n_samples, dim=0)

        # the perturbation shape
        if perturbation_axis is None:
            perturbation_shape = perturbed_input.shape
        else:
            perturbation_axis = (slice(None),) + perturbation_axis  # type: ignore
            perturbation_shape = perturbed_input[perturbation_axis].shape

        # the noise
        noise = torch.normal(0, stdevs, size=perturbation_shape).to(input.device)

        # add the noise to the perturbed_input
        if perturbation_axis is None:
            perturbed_input += noise
        else:
            perturbed_input[perturbation_axis] += noise

        perturbed_input.requires_grad_(True)

        return perturbed_input

    @staticmethod
    def mask_perturb_input(
        input, baseline, n_samples: int = 1, perturbation_prob: float = 0.5, num_perturbed_bands: int | None = None
    ) -> torch.Tensor:
        """The perturbation function used in the hyper noise tunnel. It randomly selects a subset of the input bands
        that will be masked out and replaced with the baseline. The parameters `num_perturbed_bands` and
        `perturbation_prob` control the number of bands that will be perturbed (masked). If `num_perturbed_bands` is
        set, it will be used as the number of bands to perturb, which will be randomly selected. Otherwise, the number
        of bands will be drawn from a binomial distribution with `perturbation_prob` as the probability of success.

        Args:
            input (torch.Tensor): An input tensor to be perturbed. It should have the shape (C, H, W).
            baseline (torch.Tensor): A baseline tensor to replace the perturbed bands.
            n_samples (int): A number of samples to be drawn - number of perturbed inputs to be generated.
            perturbation_prob (float, optional): A probability that each band will be perturbed intependently. Defaults to 0.5.
            num_perturbed_bands (int | None, optional): A number of perturbed bands in the whole image. If set to None, the bands are perturbed with probability `perturbation_prob` each. Defaults to None.

        Returns:
            torch.Tensor: A perturbed tensor, which contains `n_samples` perturbed inputs.
        """
        # validate the baseline against the input
        if baseline.shape != input.shape:
            raise ValueError(f"Baseline shape {baseline.shape} does not match input shape {input.shape}")

        if n_samples < 1:
            raise ValueError("Number of perturbated samples to be generated must be greater than 0")

        if perturbation_prob < 0 or perturbation_prob > 1:
            raise ValueError("Perturbation probability must be in the range [0, 1]")

        # the perturbation
        perturbed_input = input.clone().unsqueeze(0)
        # repeat the perturbed_input on the first dimension n_samples times
        perturbed_input = perturbed_input.repeat_interleave(n_samples, dim=0)

        n_samples_x_channels_shape = (
            n_samples,
            input.shape[0],
        )  # shape of the tensor containing the perturbed channels for each sample

        channels_to_be_perturbed: torch.Tensor = torch.zeros(n_samples_x_channels_shape, device=input.device).bool()

        if num_perturbed_bands is None:
            channel_perturbation_probabilities = (
                torch.ones(n_samples_x_channels_shape, device=input.device) * perturbation_prob
            )
            channels_to_be_perturbed = torch.bernoulli(channel_perturbation_probabilities).bool()

        else:
            if num_perturbed_bands < 0 or num_perturbed_bands > input.shape[0]:
                raise ValueError(
                    f"Cannot perturb {num_perturbed_bands} bands in the input with {input.shape[0]} channels. The number of perturbed bands must be in the range [0, {input.shape[0]}]"
                )

            channels_to_be_perturbed = torch_random_choice(
                input.shape[0], num_perturbed_bands, n_samples, device=input.device
            )

        # now having chosen the perturbed channels, we can replace them with the baseline

        reshaped_baseline = baseline.unsqueeze(0).repeat_interleave(n_samples, dim=0)
        perturbed_input[channels_to_be_perturbed] = reshaped_baseline[channels_to_be_perturbed]

        perturbed_input.requires_grad_(True)

        return perturbed_input

    def attribute(
        self,
        inputs: list[HSI] | HSI,
        baselines: torch.Tensor | None = None,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        n_samples: int = 5,
        steps_per_batch: int = 1,
        perturbation_prob: float = 0.5,
        num_perturbed_bands: int | None = None,
        perturbation_axis: None | tuple[int | slice] = None,
        stdevs: tuple[float, ...] = (1.0,),
        method: str = "smoothgrad",
    ) -> torch.Tensor:
        if method not in NoiseTunnelType.__members__:
            msg = f"Method must be one of {NoiseTunnelType.__members__.keys()}"
            raise ValueError(msg)

        if self.perturbation_method == "mask":
            logger.debug("Using the mask perturbation method for the Noise Tunnel")
            if baselines is None:
                raise ValueError("Baselines must be provided for the mask perturbation method")

        elif self.perturbation_method == "gaussian":
            logger.debug("Using the gaussian perturbation method for the Noise Tunnel")

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not isinstance(target, list):
            target = [target] * len(inputs)  # type: ignore

        attributions = torch.empty((n_samples, len(inputs)) + inputs[0].image.shape, device=inputs[0].device)

        for batch in range(0, len(inputs)):
            input = inputs[batch]
            targeted = target[batch]
            if self.perturbation_method == "gaussian":
                stdev = stdevs[batch]
                perturbed_input = BaseNoiseTunnel.gaussian_perturb_input(
                    input.image, n_samples, perturbation_axis, stdev
                )
            elif self.perturbation_method == "mask":
                baseline = baselines[batch]  # type: ignore
                perturbed_input = BaseNoiseTunnel.mask_perturb_input(
                    input.image, baseline, n_samples, perturbation_prob, num_perturbed_bands
                )
            for i in range(0, n_samples, steps_per_batch):
                perturbed_batch = []
                for i in range(n_samples):
                    temp_input = deepcopy(input)
                    temp_input.image = perturbed_input[i]
                    perturbed_batch.append(temp_input)

                temp_attr = self.attribute_main(
                    perturbed_batch, target=targeted, additional_forward_args=additional_forward_args
                )
                attributions[i : i + steps_per_batch, batch] = torch.stack(
                    [attr.attributes for attr in temp_attr], dim=0
                )
            else:
                steps_left = n_samples % steps_per_batch
                if steps_left:
                    perturbed_batch = []
                    for i in range(n_samples - steps_left, n_samples):
                        temp_input = deepcopy(input)
                        temp_input.image = perturbed_input[i]
                        perturbed_batch.append(temp_input)

                    temp_attr = self.attribute_main(
                        perturbed_batch, target=targeted, additional_forward_args=additional_forward_args
                    )
                    attributions[-steps_left:, batch] = torch.stack([attr.attributes for attr in temp_attr], dim=0)

        if NoiseTunnelType[method] == NoiseTunnelType.smoothgrad:
            return attributions.mean(dim=0)
        elif NoiseTunnelType[method] == NoiseTunnelType.smoothgrad_sq:
            return (attributions**2).mean(dim=0)
        elif NoiseTunnelType[method] == NoiseTunnelType.vargrad:
            return (attributions**2 - attributions.mean(dim=0) ** 2).mean(dim=0)
        else:
            raise ValueError(f"Aggregation method for NoiseTunnel {method} is not implemented")


class NoiseTunnel(Explainer):
    """Noise Tunnel is a method that is used to explain the model's predictions by adding noise to the input tensor.
    The noise is added to the input tensor, and the model's output is computed. The process is repeated multiple times
    to obtain a distribution of the model's output. The final attribution is computed as the mean of the outputs.
    For more information about the method, see captum documentation: https://captum.ai/api/noise_tunnel.html.

    Attributes:
        _attribution_method (BaseNoiseTunnel): The Noise Tunnel Base method with the gaussian perturbation
    """

    def __init__(self, attribution_method):
        super().__init__(attribution_method)
        validate_attribution_method_initialization(attribution_method)
        self._attribution_method: Attribution = BaseNoiseTunnel(attribution_method, perturbation_method="gaussian")

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        n_samples: int = 5,
        steps_per_batch: int = 1,
        perturbation_axis: None | tuple[int | slice] = None,
        stdevs: float | tuple[float, ...] = 1.0,
        method: Literal["smoothgrad", "smoothgrad_sq", "vargrad"] = "smoothgrad",
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Noise Tunnel method.

        hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            baseline (int | float | torch.Tensor, optional): Baselines define reference value which replaces each
                feature when occluded is computed and can be provided as:
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
            n_samples (int, optional):The number of randomly generated examples per sample in the input batch.
                Random examples are generated by adding gaussian random noise to each sample.
                Default: 5 if nt_samples is not provided.
            perturbation_axis (None | tuple[int | slice], optional): The indices of the input image to be perturbed.
                If set to None, all bands are perturbed. Defaults to None.
            stdevs (float | tuple[float, ...], optional): The standard deviation of gaussian noise with zero mean that
                is added to each input in the batch. If stdevs is a single float value then that same value is used
                for all inputs. If stdevs is a tuple, then the length of the tuple must match the number of inputs as
                each value in the tuple is used for the corresponding input. Default: 1.0
            method (Literal["smoothgrad", "smoothgrad_sq", "vargrad"], optional): Smoothing type of the attributions.
                smoothgrad, smoothgrad_sq or vargrad Default: smoothgrad if type is not provided.
            postprocessing_segmentation_output (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional):
                A segmentation postprocessing function for segmentation problem type. This is required for segmentation
                problem type as attribution methods needs to have 1d output. Defaults to None, which means that the
                attribution method is not used.

        Returns:
            HSIAttributes | list[HSIAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Examples:
            >>> noise_tunnel = NoiseTunnel(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = noise_tunnel.attribute(hsi)
            >>> attributions = noise_tunnel.attribute([hsi, hsi])
            >>> len(attributions)
            2
        """
        if isinstance(stdevs, list):
            stdevs = tuple(stdevs)

        if isinstance(hsi, list):
            if isinstance(stdevs, tuple):
                if len(stdevs) != len(hsi):
                    raise ValueError("The number of stdevs must match the number of input images")
            else:
                stdevs = tuple([stdevs] * len(hsi))
        else:
            if isinstance(stdevs, tuple):
                if len(stdevs) != 1:
                    warnings.warn("The number of stdevs must match the number of input images, using the first stdev")
                    stdevs = tuple([stdevs[0]])
            else:
                stdevs = tuple([stdevs])

        if postprocessing_segmentation_output is not None:

            def adjusted_forward_func(x: torch.Tensor) -> torch.Tensor:
                return postprocessing_segmentation_output(self._attribution_method.forward_func(x), torch.tensor(1.0))

            segmentation_attribution_method = BaseNoiseTunnel(adjusted_forward_func)

            nt_attributes = segmentation_attribution_method.attribute(
                hsi,
                target=target,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                steps_per_batch=steps_per_batch,
                method=method,
                perturbation_axis=perturbation_axis,
                stdevs=stdevs,
            )
        else:
            nt_attributes = self._attribution_method.attribute(
                hsi,
                target=target,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                steps_per_batch=steps_per_batch,
                method=method,
                perturbation_axis=perturbation_axis,
                stdevs=stdevs,
            )

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, nt_attributes)
            ]
        else:
            attributes = HSIAttributes(hsi=hsi, attributes=nt_attributes.squeeze(0), attribution_method=self.get_name())

        return attributes


class HyperNoiseTunnel(Explainer):
    """Hyper Noise Tunnel is our novel method, designed specifically to explain hyperspectral satellite images. It is
    inspired by the behaviour of the classical Noise Tunnel (Smooth Grad) method, but instead of sampling noise into the
    original image, it randomly removes some of the bands. In the process, the created _noised_ samples are close to the
    distribution of the original image yet differ enough to smoothen the produced attribution map.

    Attributes:
        _attribution_method (BaseNoiseTunnel): The Base Noise Tunnel class with mask perturbation method.
    """

    def __init__(self, attribution_method):
        super().__init__(attribution_method)
        validate_attribution_method_initialization(attribution_method)
        self._attribution_method: Attribution = BaseNoiseTunnel(attribution_method, perturbation_method="mask")

    def attribute(
        self,
        hsi: list[HSI] | HSI,
        baselines: int | float | torch.Tensor | None = None,
        target: list[int] | int | None = None,
        additional_forward_args: Any = None,
        n_samples: int = 5,
        steps_per_batch: int = 1,
        perturbation_prob: float = 0.5,
        num_perturbed_bands: int | None = None,
        method: Literal["smoothgrad", "smoothgrad_sq", "vargrad"] = "smoothgrad",
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> HSIAttributes | list[HSIAttributes]:
        """
        Method for generating attributions using the Hyper Noise Tunnel method.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSIAttributes objects.
            baseline (int | float | torch.Tensor, optional): Baselines define reference value which replaces each
                feature when occluded is computed and can be provided as:
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
            n_samples (int, optional):The number of randomly generated examples per sample in the input batch.
                Random examples are generated by adding gaussian random noise to each sample.
                Default: 5 if nt_samples is not provided.
            steps_per_batch (int, optional): The number of the n_samples that will be processed together.
                With the help of this parameter we can avoid out of memory situation and reduce the number of randomly
                generated examples per sample in each batch. Default: None if steps_per_batch is not provided.
                In this case all nt_samples will be processed together.
            perturbation_prob (float, optional): The probability that each band will be perturbed independently.
                Defaults to 0.5.
            num_perturbed_bands (int | None, optional): The maximum number of perturbed bands in the whole image.
                If set to None, the bands are perturbed with probability `perturbation_prob` each. Defaults to None.
            method (Literal["smoothgrad", "smoothgrad_sq", "vargrad"], optional): Smoothing type of the attributions.
                smoothgrad, smoothgrad_sq or vargrad Default: smoothgrad if type is not provided.
            postprocessing_segmentation_output (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional):
                A segmentation postprocessing function for segmentation problem type. This is required for segmentation
                problem type as attribution methods needs to have 1d output. Defaults to None, which means that the
                attribution method is not used.

        Returns:
            HSIAttributes | list[HSIAttributes]: The computed attributions for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Examples:
            >>> hyper_noise_tunnel = HyperNoiseTunnel(explainable_model)
            >>> hsi = HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> attributions = hyper_noise_tunnel.attribute(hsi)
            >>> attributions = hyper_noise_tunnel.attribute([hsi, hsi])
            >>> len(attributions)
            2
        """

        change_orientation = [False]
        original_orientation = ("C", "H", "W")

        if isinstance(hsi, list):
            change_orientation = []
            for i in range(len(hsi)):
                if hsi[i].orientation != ("C", "H", "W"):
                    change_orientation.append(True)
                    original_orientation = hsi[i].orientation
                    hsi[i] = hsi[i].change_orientation("CHW")
                else:
                    change_orientation.append(False)
        else:
            if hsi.orientation != ("C", "H", "W"):
                change_orientation[0] = True
                original_orientation = hsi.orientation
                hsi = hsi.change_orientation("CHW")

        if isinstance(hsi, list):
            if isinstance(baselines, torch.Tensor) and baselines.ndim == 4:
                if len(baselines) != len(hsi):
                    raise ValueError("The number of baselines must match the number of input images")

                baselines = torch.stack(
                    [validate_and_transform_baseline(baselines[i], hsi[i]) for i in range(len(hsi))], dim=0
                )
            else:
                baselines = torch.stack(
                    [validate_and_transform_baseline(baselines, hsi[i]) for i in range(len(hsi))], dim=0
                )
        else:
            baselines = validate_and_transform_baseline(baselines, hsi).unsqueeze(0)

        if postprocessing_segmentation_output is not None:
            new_attributed_method = partial(
                self._attribution_method.attribute_main,
                postprocessing_segmentation_output=postprocessing_segmentation_output,
            )
            new_object = GradientAttribution(lambda x: x)
            new_object.attribute = new_attributed_method
            segmentation_attribution_method = BaseNoiseTunnel(new_object, perturbation_method="mask")

            hnt_attributes = segmentation_attribution_method.attribute(
                hsi,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                steps_per_batch=steps_per_batch,
                method=method,
                perturbation_prob=perturbation_prob,
                num_perturbed_bands=num_perturbed_bands,
            )
        else:
            hnt_attributes = self._attribution_method.attribute(
                hsi,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                steps_per_batch=steps_per_batch,
                method=method,
                perturbation_prob=perturbation_prob,
                num_perturbed_bands=num_perturbed_bands,
            )

        attributes: HSIAttributes | list[HSIAttributes]
        if isinstance(hsi, list):
            attributes = [
                HSIAttributes(hsi=hsi_image, attributes=attribution, attribution_method=self.get_name())
                for hsi_image, attribution in zip(hsi, hnt_attributes)
            ]

            # change back the attributes orientation
            for i in range(len(attributes)):
                if change_orientation[i]:
                    attributes[i] = attributes[i].change_orientation(original_orientation)
        else:
            attributes = HSIAttributes(
                hsi=hsi, attributes=hnt_attributes.squeeze(0), attribution_method=self.get_name()
            )

            # change back the attributes orientation
            if change_orientation[0]:
                attributes = attributes.change_orientation(original_orientation)

        return attributes
