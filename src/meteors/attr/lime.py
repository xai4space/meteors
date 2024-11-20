from __future__ import annotations

from typing_extensions import Literal, Callable, Any, TypeVar, Type
import warnings

from loguru import logger
from itertools import chain

import torch
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import spyndex


from meteors import HSI
from meteors.attr.lime_base import Lime as LimeBase
from meteors.models import ExplainableModel, InterpretableModel, SkLearnLasso
from meteors.utils import torch_dtype_to_python_dtype, change_dtype_of_list, expand_spectral_mask
from meteors.attr import Explainer
from meteors.attr import HSISpatialAttributes, HSISpectralAttributes
from meteors.attr.attributes import ensure_torch_tensor
from meteors.exceptions import (
    ShapeMismatchError,
    BandSelectionError,
    MaskCreationError,
    HSIAttributesError,
)

try:
    from fast_slic import Slic as slic
except ImportError:
    from skimage.segmentation import slic


# Types
IntOrFloat = TypeVar("IntOrFloat", int, float)
ListOfWavelengthsIndices = TypeVar("ListOfWavelengthsIndices", list[tuple[int, int]], tuple[int, int], list[int], int)
ListOfWavelengths = TypeVar(
    "ListOfWavelengths",
    list[tuple[float, float]],
    list[tuple[int, int]],
    tuple[float, float],
    tuple[int, int],
    list[float],
    list[int],
    float,
    int,
)
BandType = TypeVar(
    "BandType",
    list[tuple[float, float]],
    list[tuple[int, int]],
    tuple[float, float],
    tuple[int, int],
    list[float],
    list[int],
    float,
    int,
)


#####################################################################
############################ VALIDATIONS ############################
#####################################################################


def validate_band_names(band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int]) -> None:
    """Validates the band names provided.

    Args:
        band_names (list[str | list[str]] | dict[tuple[str, ...] | str, int]): The band names to validate.
            It can be either a list of strings or lists of strings, or a dictionary with keys as tuples or strings
            and values as integers.

    Raises:
        TypeError: If the band_names is not a list or a dictionary, or if the items in the list or the keys in the
            dictionary are not of the expected types.

    Returns:
        None
    """
    if isinstance(band_names, dict):
        for key, item in band_names.items():
            if not (
                isinstance(key, str) or (isinstance(key, tuple) and all(isinstance(subitem, str) for subitem in key))
            ):
                raise TypeError("All keys in band_names dictionary should be str or tuple of str.")
            if not isinstance(item, int):
                raise TypeError("All values in band_names dictionary should be int.")
    elif isinstance(band_names, list):
        for item in band_names:  # type: ignore
            if not (
                isinstance(item, str) or (isinstance(item, list) and all(isinstance(subitem, str) for subitem in item))
            ):
                raise TypeError("All items in band_names list should be str or list of str.")
    else:
        raise TypeError("band_names should be either a list or a dictionary.")


def validate_band_format(bands: dict[str | tuple[str, ...], BandType], variable_name: str) -> None:
    """Validate the band format for a given variable.

    Args:
        bands (dict[str | tuple[str, ...], BandType]): A dictionary containing band ranges or list of wavelengths.
            The keys can be either a string or a tuple of strings. The values can be a single value, a tuple of two values, or a list of values.
        variable_name (str): The name of the variable being validated.

    Raises:
        TypeError: If the keys are not a string or a tuple of strings, or if the values do not match the expected types.

    Returns:
        None
    """
    for keys, band_ranges in bands.items():
        if not (isinstance(keys, str) or (isinstance(keys, tuple) and all(isinstance(key, str) for key in keys))):
            raise TypeError(f"{variable_name} keys should be string or tuple of strings")
        if isinstance(band_ranges, (int, float)):
            continue
        elif (
            isinstance(band_ranges, tuple)
            and len(band_ranges) == 2
            and all(isinstance(item, (int, float)) for item in band_ranges)
            and band_ranges[0] < band_ranges[1]
        ):
            continue
        elif isinstance(band_ranges, list) and (
            all(
                isinstance(item, tuple)
                and len(item) == 2
                and all(isinstance(subitem, (int, float)) for subitem in item)
                and item[0] < item[1]
                for item in band_ranges
            )
            or all(isinstance(item, (int, float)) for item in band_ranges)
        ):
            continue
        raise TypeError(
            (
                f"{variable_name} should be either a value, list of values, "
                "tuple of two values or list of tuples of two values."
            )
        )


def validate_segment_format(
    segment: tuple[IntOrFloat, IntOrFloat] | list[tuple[IntOrFloat, IntOrFloat]], dtype: Type = int
) -> list[tuple[IntOrFloat, IntOrFloat]]:
    """Validates the format of the segment.

    Args:
        segment (tuple[int | float, int | float] | list[tuple[int | float, int | float]]):
            The segment to validate.
        dtype (Type, optional): The data type of the segment range. Defaults to int.

    Returns:
        list[tuple[int | float, int | float]]: The validated segment range.

    Raises:
        ValueError: If the segment range is not in the correct format.
    """
    if (
        isinstance(segment, tuple)
        and len(segment) == 2
        and all(isinstance(x, dtype) for x in segment)
        and segment[0] < segment[1]
    ):
        logger.debug("Converting tuple segment to list of tuples")
        segment = [segment]  # Standardize single tuple to list of tuples
    elif not (
        isinstance(segment, list)
        and all(
            isinstance(sub_segment, tuple)
            and len(sub_segment) == 2
            and all(isinstance(part, dtype) for part in sub_segment)
            and sub_segment[0] < sub_segment[1]
            for sub_segment in segment
        )
    ):
        raise ValueError(
            (
                f"Each segment range should be a tuple or list of two numbers of data type {dtype} (start, end). "
                f"Where start < end. But got: {segment}"
            )
        )
    return segment


def adjust_and_validate_segment_ranges(
    wavelengths: torch.Tensor, segment_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Adjusts and validates segment ranges against the wavelength dimension.

    This function ensures that each segment range is within the bounds of the wavelength
    dimension. It attempts to adjust out-of-bounds ranges when possible and raises an
    error for ranges that cannot be adjusted.

    Args:
        wavelengths (torch.Tensor): The wavelengths tensor. Its length determines the
            valid range for segments.
        segment_ranges (list[tuple[int, int]]): A list of segment ranges to be validated
            and potentially adjusted. Each tuple represents (start, end) indices.

    Returns:
        list[tuple[int, int]]: The list of validated and potentially adjusted segment ranges.

    Raises:
        ValueError: If a segment range is entirely out of bounds and cannot be adjusted.

    Warns:
        UserWarning: If a segment range is partially out of bounds and is adjusted.

    Notes:
        - Segment ranges are inclusive of both start and end indices.
        - Ranges extending below 0 are adjusted to start at 0 if possible.
        - Ranges extending beyond the wavelength dimension are truncated if possible.
        - Adjustments are only made if at least part of the range is within bounds.
    """
    max_index = len(wavelengths)
    adjusted_ranges = []

    for start, end in segment_ranges:
        if start < 0:
            if end > 0:
                logger.debug(f"Adjusting segment start from {start} to 0")
                start = 0
            else:
                raise ValueError(f"Segment range {(start, end)} is out of bounds")

        if end > max_index:
            if start < max_index:
                logger.debug(f"Adjusting segment end from {(start, end)} to {(start, max_index)}")
                end = max_index
            else:
                raise ValueError(f"Segment range {(start, end)} is out of bounds")
        adjusted_ranges.append((start, end))
    return adjusted_ranges  # type: ignore


def validate_mask_shape(mask_type: Literal["segmentation", "band"], hsi: HSI, mask: torch.Tensor) -> torch.Tensor:
    """Validate mask (segmentation or band mask) shape against the hyperspectral image.

    Args:
        mask_type (Literal[&quot;spatial&quot;, &quot;spectral&quot;]): a problem type specifying type of the attribution for the mask is segmentation or band mask
        hsi (HSI): An original hyperspectral image for which the mask is created
        mask (torch.Tensor): A segmentation or band mask to be validated.

    Raises:
        ValueError: In case the shapes cannot be broadcasted, or the image and band mask orientation is invalid
    Returns:
        torch.Tensor: The validated mask
    """

    if mask_type not in ["segmentation", "band"]:
        raise ValueError(f"Unsupported mask type passed to validation: {mask_type}")

    image_shape = hsi.image.shape
    mask_shape = mask.shape

    if len(mask_shape) == 2 and mask_type == "segmentation":
        mask = mask.unsqueeze(hsi.spectral_axis)
        mask_shape = mask.shape
    elif len(mask_shape) != 3:
        raise ValueError(f"Mask should be a 3D tensor, but got shape: {mask_shape}")

    try:
        broadcasted_shape = torch.broadcast_shapes(image_shape, mask_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Cannot broadcast image and mask of shapes {image_shape} and {mask_shape} respectively: {e}"
        ) from e

    if broadcasted_shape != image_shape:
        raise ShapeMismatchError(f"HSI and {mask_type} mask have mismatched shapes: {image_shape}, {mask_shape}")

    # check on which dims the shapes match - the segmentation mask can differ only in the band dimension, band mask can differ in the height and width dimensions
    shape_matches = [broadcasted_shape[i] == mask_shape[i] for i in range(3)]
    orientation_mismatches = {hsi.orientation[i] for i in range(3) if not shape_matches[i]}

    if mask_type == "segmentation" and ("H" in orientation_mismatches or "W" in orientation_mismatches):
        raise ValueError(
            f"Image and mask orientation mismatch: {hsi.orientation} and {mask_shape}."
            + "Segmentation mask should differ only in the band dimension"
        )

    if mask_type == "band" and "C" in orientation_mismatches:
        raise ValueError(
            f"Image and mask orientation mismatch: {hsi.orientation} and {mask_shape}."
            + "Band mask should differ only in the height and width dimensions"
        )

    mask = mask.expand_as(hsi.image)

    return mask


###################################################################
############################ EXPLAINER ############################
###################################################################


class Lime(Explainer):
    """Lime class is a subclass of Explainer and represents the Lime explainer. Lime is an interpretable model-agnostic
    explanation method that explains the predictions of a black-box model by approximating it with a simpler
    interpretable model. The Lime method is based on the [`captum` implementation](https://captum.ai/api/lime.html)
    and is an implementation of an idea coming from the [original paper on Lime](https://arxiv.org/abs/1602.04938),
    where more details about this method can be found.

    Args:
        explainable_model (ExplainableModel): The explainable model to be explained.
        interpretable_model (InterpretableModel): The interpretable model used to approximate the black-box model.
            Defaults to `SkLearnLasso` with alpha parameter set to 0.08.
        similarity_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The similarity function used by Lime.
            Defaults to None.
        perturb_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The perturbation function used by Lime.
            Defaults to None.
    """

    def __init__(
        self,
        explainable_model: ExplainableModel,
        interpretable_model: InterpretableModel = SkLearnLasso(alpha=0.08),
        similarity_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
        perturb_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(explainable_model)
        self.interpretable_model = interpretable_model
        self._attribution_method: LimeBase = self._construct_lime(
            self.explainable_model.forward_func, interpretable_model, similarity_func, perturb_func
        )

    @staticmethod
    def _construct_lime(
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        interpretable_model: InterpretableModel,
        similarity_func: Callable | None,
        perturb_func: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> LimeBase:
        """Constructs the LimeBase object.

        Args:
            forward_func (Callable[[torch.Tensor], torch.Tensor]): The forward function of the explainable model.
            interpretable_model (InterpretableModel): The interpretable model used to approximate the black-box model.
            similarity_func (Callable | None): The similarity function used by Lime.
            perturb_func (Callable[[torch.Tensor], torch.Tensor] | None): The perturbation function used by Lime.

        Returns:
            LimeBase: The constructed LimeBase object.
        """
        return LimeBase(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )

    @staticmethod
    def get_segmentation_mask(
        hsi: HSI,
        segmentation_method: Literal["patch", "slic"] = "slic",
        **segmentation_method_params: Any,
    ) -> torch.Tensor:
        """Generates a segmentation mask for the given hsi using the specified segmentation method.

        Args:
            hsi (HSI): The input hyperspectral image for which the segmentation mask needs to be generated.
            segmentation_method (Literal["patch", "slic"], optional): The segmentation method to be used.
                Defaults to "slic".
            **segmentation_method_params (Any): Additional parameters specific to the chosen segmentation method.

        Returns:
            torch.Tensor: The segmentation mask as a tensor.

        Raises:
            TypeError: If the input hsi is not an instance of the HSI class.
            ValueError: If an unsupported segmentation method is specified.

        Examples:
            >>> hsi = meteors.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
            >>> segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="slic")
            >>> segmentation_mask.shape
            torch.Size([1, 240, 240])
            >>> segmentation_mask = meteors.attr.Lime.get_segmentation_mask(hsi, segmentation_method="patch", patch_size=2)
            >>> segmentation_mask.shape
            torch.Size([1, 240, 240])
            >>> segmentation_mask[0, :2, :2]
            torch.tensor([[1, 1],
                          [1, 1]])
            >>> segmentation_mask[0, 2:4, :2]
            torch.tensor([[2, 2],
                          [2, 2]])
        """
        if not isinstance(hsi, HSI):
            raise TypeError("hsi should be an instance of HSI class")

        try:
            if segmentation_method == "slic":
                return Lime._get_slic_segmentation_mask(hsi, **segmentation_method_params)
            elif segmentation_method == "patch":
                return Lime._get_patch_segmentation_mask(hsi, **segmentation_method_params)
            else:
                raise ValueError(f"Unsupported segmentation method: {segmentation_method}")
        except Exception as e:
            raise MaskCreationError(f"Error creating segmentation mask using method {segmentation_method}: {e}")

    @staticmethod
    def get_band_mask(
        hsi: HSI,
        band_names: None | list[str | list[str]] | dict[tuple[str, ...] | str, int] = None,
        band_indices: None | dict[str | tuple[str, ...], ListOfWavelengthsIndices] = None,
        band_wavelengths: None | dict[str | tuple[str, ...], ListOfWavelengths] = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
    ) -> tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """Generates a band mask based on the provided hsi and band information.

        Remember you need to provide either band_names, band_indices, or band_wavelengths to create the band mask.
        If you provide more than one, the band mask will be created using only one using the following priority:
        band_names > band_wavelengths > band_indices.

        Args:
            hsi (HSI): The input hyperspectral image.
            band_names (None | list[str | list[str]] | dict[tuple[str, ...] | str, int], optional):
                The names of the spectral bands to include in the mask. Defaults to None.
            band_indices (None | dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int] | list[int]], optional):
                The indices or ranges of indices of the spectral bands to include in the mask. Defaults to None.
            band_wavelengths (None | dict[str | tuple[str, ...], list[tuple[float, float]] | tuple[float, float], list[float], float], optional):
                The wavelengths or ranges of wavelengths of the spectral bands to include in the mask. Defaults to None.
            device (str | torch.device | None, optional):
                The device to use for computation. Defaults to None.
            repeat_dimensions (bool, optional):
                Whether to repeat the dimensions of the mask to match the input hsi shape. Defaults to False.

        Returns:
            tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]: A tuple containing the band mask tensor and a dictionary
            mapping band names to segment IDs.

        Raises:
            TypeError: If the input hsi is not an instance of the HSI class.
            ValueError: If no band names, indices, or wavelengths are provided.

        Examples:
            >>> hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
            >>> band_names = ["R", "G"]
            >>> band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
            >>> dict_labels_to_segment_ids
            {"R": 1, "G": 2}
            >>> band_indices = {"RGB": [0, 1, 2]}
            >>> band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)
            >>> dict_labels_to_segment_ids
            {"RGB": 1}
            >>> band_wavelengths = {"RGB": [(462.08, 465.27), (465.27, 468.47), (468.47, 471.68)]}
            >>> band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)
            >>> dict_labels_to_segment_ids
            {"RGB": 1}
        """
        if not isinstance(hsi, HSI):
            raise TypeError("hsi should be an instance of HSI class")

        try:
            if not (band_names is not None or band_indices is not None or band_wavelengths is not None):
                raise ValueError("No band names, indices, or wavelengths are provided.")

            # validate types
            dict_labels_to_segment_ids = None
            if band_names is not None:
                logger.debug("Getting band mask from band names of spectral bands")
                if band_wavelengths is not None or band_indices is not None:
                    ignored_params = [
                        param
                        for param in ["band_wavelengths", "band_indices"]
                        if param in locals() and locals()[param] is not None
                    ]
                    ignored_params_str = " and ".join(ignored_params)
                    logger.info(
                        f"Only the band names will be used to create the band mask. The additional parameters {ignored_params_str} will be ignored."
                    )
                try:
                    validate_band_names(band_names)
                    band_groups, dict_labels_to_segment_ids = Lime._get_band_wavelengths_indices_from_band_names(
                        hsi.wavelengths, band_names
                    )
                except Exception as e:
                    raise BandSelectionError(f"Incorrect band names provided: {e}") from e
            elif band_wavelengths is not None:
                logger.debug("Getting band mask from band groups given by ranges of wavelengths")
                if band_indices is not None:
                    logger.info(
                        "Only the band wavelengths will be used to create the band mask. The band_indices will be ignored."
                    )
                validate_band_format(band_wavelengths, variable_name="band_wavelengths")
                try:
                    band_groups = Lime._get_band_indices_from_band_wavelengths(
                        hsi.wavelengths,
                        band_wavelengths,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Incorrect band ranges wavelengths provided, please check if provided wavelengths are correct: {e}"
                    ) from e
            elif band_indices is not None:
                logger.debug("Getting band mask from band groups given by ranges of indices")
                validate_band_format(band_indices, variable_name="band_indices")
                try:
                    band_groups = Lime._get_band_indices_from_input_band_indices(hsi.wavelengths, band_indices)
                except Exception as e:
                    raise ValueError(
                        f"Incorrect band ranges indices provided, please check if provided indices are correct: {e}"
                    ) from e

            return Lime._create_tensor_band_mask(
                hsi,
                band_groups,
                dict_labels_to_segment_ids=dict_labels_to_segment_ids,
                device=device,
                repeat_dimensions=repeat_dimensions,
                return_dict_labels_to_segment_ids=True,
            )
        except Exception as e:
            raise MaskCreationError(f"Error creating band mask: {e}") from e

    @staticmethod
    def _make_band_names_indexable(segment_name: list[str] | tuple[str, ...] | str) -> tuple[str, ...] | str:
        """Converts a list of strings into a tuple of strings if necessary to make it indexable.

        Args:
            segment_name (list[str] | tuple[str, ...] | str): The segment name to be converted.

        Returns:
            tuple[str, ...] | str: The converted segment name.

        Raises:
            TypeError: If the segment_name is not of type list or string.
        """
        if (
            isinstance(segment_name, tuple) and all(isinstance(subitem, str) for subitem in segment_name)
        ) or isinstance(segment_name, str):
            return segment_name
        elif isinstance(segment_name, list) and all(isinstance(subitem, str) for subitem in segment_name):
            return tuple(segment_name)
        raise TypeError(f"Incorrect segment {segment_name} type. Should be either a list or string")

    @staticmethod
    # @lru_cache(maxsize=32) Can't use with lists as they are not hashable
    def _extract_bands_from_spyndex(segment_name: list[str] | tuple[str, ...] | str) -> tuple[str, ...] | str:
        """Extracts bands from the given segment name.

        Args:
            segment_name (list[str] | tuple[str, ...] | str): The name of the segment.
                Users may pass either band names or indices names, as in the spyndex library.

        Returns:
            tuple[str, ...] | str: A tuple of band names if multiple bands are extracted,
                or a single band name if only one band is extracted.

        Raises:
            BandSelectionError: If the provided band name is invalid.
                The band name must be either in `spyndex.indices` or `spyndex.bands`.
        """
        if isinstance(segment_name, str):
            segment_name = (segment_name,)
        elif isinstance(segment_name, list):
            segment_name = tuple(segment_name)

        band_names_segment: list[str] = []
        for band_name in segment_name:
            if band_name in spyndex.indices:
                band_names_segment += list(spyndex.indices[band_name].bands)
            elif band_name in spyndex.bands:
                band_names_segment.append(band_name)
            else:
                raise BandSelectionError(
                    f"Invalid band name {band_name}, band name must be either in `spyndex.indices` or `spyndex.bands`"
                )

        return tuple(set(band_names_segment)) if len(band_names_segment) > 1 else band_names_segment[0]

    @staticmethod
    def _get_indices_from_wavelength_indices_range(
        wavelengths: torch.Tensor, ranges: list[tuple[int, int]] | tuple[int, int]
    ) -> list[int]:
        """Converts wavelength indices ranges to list indices.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            ranges (list[tuple[int, int]] | tuple[int, int]): The wavelength indices ranges.

        Returns:
            list[int]: The indices of bands corresponding to the wavelength indices ranges.
        """
        validated_ranges_list = validate_segment_format(ranges)
        validated_ranges_list = adjust_and_validate_segment_ranges(wavelengths, validated_ranges_list)

        return list(
            set(
                chain.from_iterable(
                    [list(range(int(validated_range[0]), int(validated_range[1]))) for validated_range in ranges]  # type: ignore
                )
            )
        )

    @staticmethod
    def _get_band_wavelengths_indices_from_band_names(
        wavelengths: torch.Tensor,
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int],
    ) -> tuple[dict[tuple[str, ...] | str, list[int]], dict[tuple[str, ...] | str, int]]:
        """Extracts band wavelengths indices from the given band names.

        This function takes a list or dictionary of band names or segments and extracts the list of wavelengths indices
        associated with each segment. It returns a tuple containing a dictionary with mapping segment labels into
        wavelength indices and a dictionary mapping segment labels into segment ids.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            band_names (list[str | list[str]] | dict[tuple[str, ...] | str, int]):
                A list or dictionary with band names or segments.

        Returns:
            tuple[dict[tuple[str, ...] | str, list[int]], dict[tuple[str, ...] | str, int]]:
                A tuple containing the dictionary with mapping segment labels into wavelength indices and the mapping
                from segment labels into segment ids.

        Raises:
            TypeError: If the band names are not in the correct format.
        """
        if isinstance(band_names, str):
            band_names = [band_names]
        if isinstance(band_names, list):
            logger.debug("band_names is a list of segments, creating a dictionary of segments")
            band_names_hashed = [Lime._make_band_names_indexable(segment) for segment in band_names]
            dict_labels_to_segment_ids = {segment: idx + 1 for idx, segment in enumerate(band_names_hashed)}
            segments_list = band_names_hashed
        elif isinstance(band_names, dict):
            dict_labels_to_segment_ids = band_names.copy()
            segments_list = tuple(band_names.keys())  # type: ignore
        else:
            raise TypeError("Incorrect band_names type. It should be a dict or a list")
        segments_list_after_mapping = [Lime._extract_bands_from_spyndex(segment) for segment in segments_list]
        band_indices: dict[tuple[str, ...] | str, list[int]] = {}
        for original_segment, segment in zip(segments_list, segments_list_after_mapping):
            segment_indices_ranges: list[tuple[int, int]] = []
            if isinstance(segment, str):
                segment = (segment,)
            for band_name in segment:
                min_wavelength = spyndex.bands[band_name].min_wavelength
                max_wavelength = spyndex.bands[band_name].max_wavelength

                if min_wavelength > wavelengths.max() or max_wavelength < wavelengths.min():
                    logger.debug(
                        f"Band {band_name} is not present in the given wavelengths. "
                        f"Band ranges from {min_wavelength} nm to {max_wavelength} nm and the HSI wavelengths "
                        f"range from {wavelengths.min():.2f} nm to {wavelengths.max():.2f} nm. The given band will be skipped"
                    )
                else:
                    segment_indices_ranges += Lime._convert_wavelengths_to_indices(
                        wavelengths,
                        (spyndex.bands[band_name].min_wavelength, spyndex.bands[band_name].max_wavelength),
                    )

            segment_list = Lime._get_indices_from_wavelength_indices_range(wavelengths, segment_indices_ranges)
            band_indices[original_segment] = segment_list
        return band_indices, dict_labels_to_segment_ids

    @staticmethod
    def _convert_wavelengths_to_indices(
        wavelengths: torch.Tensor, ranges: list[tuple[float, float]] | tuple[float, float]
    ) -> list[tuple[int, int]]:
        """Converts wavelength ranges to index ranges.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            ranges (list[tuple[float, float]] | tuple[float, float]): The wavelength ranges.

        Returns:
            list[tuple[int, int]]: The index ranges corresponding to the wavelength ranges.
        """
        indices = []
        if isinstance(ranges, tuple):
            ranges = [ranges]

        for start, end in ranges:
            start_idx = torch.searchsorted(wavelengths, start, side="left")
            end_idx = torch.searchsorted(wavelengths, end, side="right")
            indices.append((start_idx.item(), end_idx.item()))
        return indices

    @staticmethod
    def _get_band_indices_from_band_wavelengths(
        wavelengths: torch.Tensor,
        band_wavelengths: dict[str | tuple[str, ...], ListOfWavelengths],
    ) -> dict[str | tuple[str, ...], list[int]]:
        """Converts the ranges or list of wavelengths into indices.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            band_wavelengths (dict): A dictionary mapping segment labels to wavelength list or ranges.

        Returns:
            dict: A dictionary mapping segment labels to index ranges.

        Raises:
            TypeError: If band_wavelengths is not a dictionary.
        """
        if not isinstance(band_wavelengths, dict):
            raise TypeError("band_wavelengths should be a dictionary")

        band_indices: dict[str | tuple[str, ...], list[int]] = {}
        for segment_label, segment in band_wavelengths.items():
            try:
                dtype = torch_dtype_to_python_dtype(wavelengths.dtype)
                if isinstance(segment, (float, int)):
                    segment = [dtype(segment)]  # type: ignore
                if isinstance(segment, list) and all(isinstance(x, (float, int)) for x in segment):
                    segment_dtype = change_dtype_of_list(segment, dtype)
                    indices = Lime._convert_wavelengths_list_to_indices(wavelengths, segment_dtype)  # type: ignore
                else:
                    if isinstance(segment, list):
                        segment_dtype = [
                            tuple(change_dtype_of_list(list(ranges), dtype))  # type: ignore
                            for ranges in segment
                        ]
                    else:
                        segment_dtype = tuple(change_dtype_of_list(segment, dtype))

                    valid_segment_range = validate_segment_format(segment_dtype, dtype)
                    range_indices = Lime._convert_wavelengths_to_indices(wavelengths, valid_segment_range)  # type: ignore
                    valid_indices_format = validate_segment_format(range_indices)
                    valid_range_indices = adjust_and_validate_segment_ranges(wavelengths, valid_indices_format)
                    indices = Lime._get_indices_from_wavelength_indices_range(wavelengths, valid_range_indices)
            except Exception as e:
                raise ValueError(f"Problem with segment {segment_label}: {e}") from e

            band_indices[segment_label] = indices

        return band_indices

    @staticmethod
    def _convert_wavelengths_list_to_indices(wavelengths: torch.Tensor, ranges: list[float]) -> list[int]:
        """Converts a list of wavelengths into indices.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            ranges (list[float]): The list of wavelengths.

        Returns:
            list[int]: The indices corresponding to the wavelengths.
        """
        indices = []
        for wavelength in ranges:
            index = (wavelengths == wavelength).nonzero(as_tuple=False)
            number_of_elements = torch.numel(index)
            if number_of_elements == 1:
                indices.append(index.item())
            elif number_of_elements == 0:
                raise ValueError(f"Couldn't find wavelength of value {wavelength} in list of wavelength")
            else:
                raise ValueError(f"Wavelength of value {wavelength} was present more than once in list of wavelength")
        return indices

    @staticmethod
    def _get_band_indices_from_input_band_indices(
        wavelengths: torch.Tensor,
        input_band_indices: dict[str | tuple[str, ...], ListOfWavelengthsIndices],
    ) -> dict[str | tuple[str, ...], list[int]]:
        """Get band indices from band list or ranges indices.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            band_indices (dict[str | tuple[str, ...], ListOfWavelengthsIndices]):
                A dictionary mapping segment labels to a list of wavelength indices.

        Returns:
            dict[str | tuple[str, ...], list[int]]: A dictionary mapping segment labels to a list of band indices.

        Raises:
            TypeError: If `band_indices` is not a dictionary.
        """
        if not isinstance(input_band_indices, dict):
            raise TypeError("band_indices should be a dictionary")

        band_indices: dict[str | tuple[str, ...], list[int]] = {}
        for segment_label, indices in input_band_indices.items():
            try:
                if isinstance(indices, int):
                    indices = [indices]  # type: ignore
                if isinstance(indices, list) and all(isinstance(x, int) for x in indices):
                    indices: list[int] = indices  # type: ignore
                else:
                    valid_indices_format = validate_segment_format(indices)  # type: ignore
                    valid_range_indices = adjust_and_validate_segment_ranges(wavelengths, valid_indices_format)
                    indices = Lime._get_indices_from_wavelength_indices_range(wavelengths, valid_range_indices)  # type: ignore

                band_indices[segment_label] = indices  # type: ignore
            except Exception as e:
                raise ValueError(f"Problem with segment {segment_label}") from e

        return band_indices

    @staticmethod
    def _check_overlapping_segments(dict_labels_to_indices: dict[str | tuple[str, ...], list[int]]) -> None:
        """Check for overlapping segments.

        Args:
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
                A dictionary mapping segment labels to indices.

        Returns:
            None
        """
        overlapping_segments: list[tuple[str | tuple[str, ...], str | tuple[str, ...]]] = []
        labels = list(dict_labels_to_indices.keys())

        for i, segment_label in enumerate(labels):
            for second_label in labels[i + 1 :]:
                indices = dict_labels_to_indices[segment_label]
                second_indices = dict_labels_to_indices[second_label]

                if set(indices) & set(second_indices):
                    overlapping_segments.append((segment_label, second_label))

        for label_first, label_second in overlapping_segments:
            label_first_str = label_first if isinstance(label_first, str) else "/".join(label_first)
            label_second_str = label_second if isinstance(label_second, str) else "/".join(label_second)

            logger.warning(
                f"Segments {label_first_str} and {label_second_str} are overlapping,"
                " overlapping wavelengths will be assigned to only one"
            )

    @staticmethod
    def _validate_and_create_dict_labels_to_segment_ids(
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None,
        segment_labels: list[str | tuple[str, ...]],
    ) -> dict[str | tuple[str, ...], int]:
        """Validates and creates a dictionary mapping segment labels to segment IDs.

        Args:
            dict_labels_to_segment_ids (dict[str | tuple[str, ...], int] | None):
                The existing mapping from segment labels to segment IDs, or None if it doesn't exist.
            segment_labels (list[str | tuple[str, ...]]): The list of segment labels.

        Returns:
            dict[str | tuple[str, ...], int]: A tuple containing the validated dictionary mapping segment
            labels to segment IDs and a boolean flag indicating whether the segment labels are hashed.

        Raises:
            ValueError: If the length of `dict_labels_to_segment_ids` doesn't match the length of `segment_labels`.
            ValueError: If a segment label is not present in `dict_labels_to_segment_ids`.
            ValueError: If there are non-unique segment IDs in `dict_labels_to_segment_ids`.
        """
        if dict_labels_to_segment_ids is None:
            logger.debug("Creating mapping from segment labels into ids")
            return {segment: idx + 1 for idx, segment in enumerate(segment_labels)}

        logger.debug("Using existing mapping from segment labels into segment ids")

        if len(dict_labels_to_segment_ids) != len(segment_labels):
            raise ValueError(
                (
                    f"Incorrect dict_labels_to_segment_ids - length mismatch. Expected: "
                    f"{len(segment_labels)}, Actual: {len(dict_labels_to_segment_ids)}"
                )
            )

        unique_segment_ids = set(dict_labels_to_segment_ids.values())
        if len(unique_segment_ids) != len(segment_labels):
            raise ValueError("Non unique segment ids in the dict_labels_to_segment_ids")

        logger.debug("Passed mapping is correct")
        return dict_labels_to_segment_ids

    @staticmethod
    def _create_single_dim_band_mask(
        hsi: HSI,
        dict_labels_to_indices: dict[str | tuple[str, ...], list[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int],
        device: torch.device,
    ) -> torch.Tensor:
        """Create a one-dimensional band mask based on the given image, labels, and segment IDs.

        Args:
            hsi (HSI): The input hsi.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
                A dictionary mapping labels or label tuples to lists of indices.
            dict_labels_to_segment_ids (dict[str | tuple[str, ...], int]):
                A dictionary mapping labels or label tuples to segment IDs.
            device (torch.device): The device to use for the tensor.

        Returns:
            torch.Tensor: The one-dimensional band mask tensor.

        Raises:
            ValueError: If the indices for a segment are out of bounds for the one-dimensional band mask.
        """
        band_mask_single_dim = torch.zeros(len(hsi.wavelengths), dtype=torch.int64, device=device)

        segment_labels = list(dict_labels_to_segment_ids.keys())

        for segment_label in segment_labels[::-1]:
            segment_indices = dict_labels_to_indices[segment_label]
            segment_id = dict_labels_to_segment_ids[segment_label]
            are_indices_valid = all(0 <= idx < band_mask_single_dim.shape[0] for idx in segment_indices)
            if not are_indices_valid:
                raise ValueError(
                    (
                        f"Indices for segment {segment_label} are out of bounds for the one-dimensional band mask"
                        f"of shape {band_mask_single_dim.shape}"
                    )
                )
            band_mask_single_dim[segment_indices] = segment_id

        return band_mask_single_dim

    @staticmethod
    def _create_tensor_band_mask(
        hsi: HSI,
        dict_labels_to_indices: dict[str | tuple[str, ...], list[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
        return_dict_labels_to_segment_ids: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """Create a tensor band mask from dictionaries. The band mask is created based on the given hsi, labels, and
        segment IDs. The band mask is a tensor with the same shape as the input hsi and contains segment IDs, where each
        segment is represented by a unique ID. The band mask will be used to attribute the hsi using the LIME method.

        Args:
            hsi (HSI): The input hsi.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]): A dictionary mapping labels to indices.
            dict_labels_to_segment_ids (dict[str | tuple[str, ...], int] | None, optional):
                A dictionary mapping labels to segment IDs. Defaults to None.
            device (str | torch.device | None, optional): The device to use. Defaults to None.
            repeat_dimensions (bool, optional): Whether to repeat dimensions. Defaults to False.
            return_dict_labels_to_segment_ids (bool, optional):
                Whether to return the dictionary mapping labels to segment IDs. Defaults to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
                The tensor band mask or a tuple containing the tensor band mask
                and the dictionary mapping labels to segment IDs.
        """
        if device is None:
            device = hsi.device
        segment_labels = list(dict_labels_to_indices.keys())

        logger.debug(f"Creating a band mask on the device {device} using {len(segment_labels)} segments")

        # Check for overlapping segments
        Lime._check_overlapping_segments(dict_labels_to_indices)

        # Create or validate dict_labels_to_segment_ids
        dict_labels_to_segment_ids = Lime._validate_and_create_dict_labels_to_segment_ids(
            dict_labels_to_segment_ids, segment_labels
        )

        # Create single-dimensional band mask
        band_mask_single_dim = Lime._create_single_dim_band_mask(
            hsi, dict_labels_to_indices, dict_labels_to_segment_ids, device
        )

        # Expand band mask to match image dimensions
        band_mask = expand_spectral_mask(hsi, band_mask_single_dim, repeat_dimensions)

        if return_dict_labels_to_segment_ids:
            return band_mask, dict_labels_to_segment_ids
        return band_mask

    def attribute(  # type: ignore
        self,
        hsi: list[HSI] | HSI,
        target: list[int] | int | None = None,
        attribution_type: Literal["spatial", "spectral"] | None = None,
        additional_forward_args: Any = None,
        **kwargs: Any,
    ) -> HSISpatialAttributes | HSISpectralAttributes | list[HSISpatialAttributes] | list[HSISpectralAttributes]:
        """A wrapper function to attribute the image using the LIME method. It executes either the
        `get_spatial_attributes` or `get_spectral_attributes` method based on the provided `attribution_type`. For more
        detailed description of the methods, please refer to the respective method documentation.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSISpatialAttributes or HSISpectralAttributes objects.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            attribution_type (Literal["spatial", "spectral"] | None, optional): The type of attribution to be computed.
                User can compute spatial or spectral attributions with the LIME method. If None, the method will
                throw an error. Defaults to None.
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            kwargs (Any): Additional keyword arguments for the LIME method.

        Returns:
            HSISpectralAttributes | HSISpatialAttributes | list[HSISpectralAttributes | HSISpatialAttributes]:
                The computed attributions Spectral or Spatial for the input hyperspectral image(s).
                if a list of HSI objects is provided, the attributions are computed for each HSI object in the list.

        Raises:
            RuntimeError: If the Lime object is not initialized or is not an instance of LimeBase.
            ValueError: If number of HSI images is not equal to the number of masks provided.

        Examples:
            >>> simple_model = lambda x: torch.rand((x.shape[0], 2))
            >>> hsi = mt.HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> segmentation_mask = torch.randint(1, 4, (1, 240, 240))
            >>> lime = meteors.attr.Lime(
                    explainable_model=ExplainableModel(simple_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.1)
                )
            >>> spatial_attribution = lime.attribute(hsi, segmentation_mask=segmentation_mask, target=0, attribution_type="spatial")
            >>> spatial_attribution.hsi
            HSI(shape=(4, 240, 240), dtype=torch.float32)
            >>> band_mask = torch.randint(1, 4, (4, 1, 1)).repeat(1, 240, 240)
            >>> band_names = ["R", "G", "B"]
            >>> spectral_attribution = lime.attribute(
            ...     hsi, band_mask=band_mask, band_names=band_names, target=0, attribution_type="spectral"
            ... )
            >>> spectral_attribution.hsi
            HSI(shape=(4, 240, 240), dtype=torch.float32)
        """
        if attribution_type == "spatial":
            return self.get_spatial_attributes(
                hsi, target=target, additional_forward_args=additional_forward_args, **kwargs
            )
        elif attribution_type == "spectral":
            return self.get_spectral_attributes(
                hsi, target=target, additional_forward_args=additional_forward_args, **kwargs
            )
        raise ValueError(f"Unsupported attribution type: {attribution_type}. Use 'spatial' or 'spectral'")

    def get_spatial_attributes(
        self,
        hsi: list[HSI] | HSI,
        segmentation_mask: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor] | None = None,
        target: list[int] | int | None = None,
        n_samples: int = 10,
        perturbations_per_eval: int = 4,
        verbose: bool = False,
        segmentation_method: Literal["slic", "patch"] = "slic",
        additional_forward_args: Any = None,
        **segmentation_method_params: Any,
    ) -> list[HSISpatialAttributes] | HSISpatialAttributes:
        """
        Get spatial attributes of an hsi image using the LIME method. Based on the provided hsi and segmentation mask
        LIME method attributes the `superpixels` provided by the segmentation mask. Please refer to the original paper
        `https://arxiv.org/abs/1602.04938` for more details or to Christoph Molnar's book
        `https://christophm.github.io/interpretable-ml-book/lime.html`.

        This function attributes the hyperspectral image using the LIME (Local Interpretable Model-Agnostic Explanations)
        method for spatial data. It returns an `HSISpatialAttributes` object that contains the hyperspectral image,,
        the attributions, the segmentation mask, and the score of the interpretable model used for the explanation.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSISpatialAttributes objects.
            segmentation_mask (np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor] | None, optional):
                A segmentation mask according to which the attribution should be performed.
                The segmentation mask should have a 2D or 3D shape, which can be broadcastable to the shape of the
                input image. The only dimension on which the image and the mask shapes can differ is the spectral
                dimension, marked with letter `C` in the `image.orientation` parameter. If None, a new segmentation mask
                is created using the `segmentation_method`. Additional parameters for the segmentation method may be
                passed as kwargs. If multiple HSI images are provided, a list of segmentation masks can be provided,
                one for each image. If list is not provided method will assume that the same segmentation mask is used
                    for all images. Defaults to None.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            n_samples (int, optional): The number of samples to generate/analyze in LIME. The more the better but slower.
                Defaults to 10.
            perturbations_per_eval (int, optional): The number of perturbations to evaluate at once
                (Simply the inner batch size). Defaults to 4.
            verbose (bool, optional): Whether to show the progress bar. Defaults to False.
            segmentation_method (Literal["slic", "patch"], optional):
                Segmentation method used only if `segmentation_mask` is None. Defaults to "slic".
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            **segmentation_method_params (Any): Additional parameters for the segmentation method.

        Returns:
            HSISpatialAttributes | list[HSISpatialAttributes]: An object containing the image, the attributions,
                the segmentation mask, and the score of the interpretable model used for the explanation.

        Raises:
            RuntimeError: If the Lime object is not initialized or is not an instance of LimeBase.
            MaskCreationError: If there is an error creating the segmentation mask.
            ValueError: If the number of segmentation masks is not equal to the number of HSI images provided.
            HSIAttributesError: If there is an error during creating spatial attribution.

        Examples:
            >>> simple_model = lambda x: torch.rand((x.shape[0], 2))
            >>> hsi = mt.HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> segmentation_mask = torch.randint(1, 4, (1, 240, 240))
            >>> lime = meteors.attr.Lime(
                    explainable_model=ExplainableModel(simple_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.1)
                )
            >>> spatial_attribution = lime.get_spatial_attributes(hsi, segmentation_mask=segmentation_mask, target=0)
            >>> spatial_attribution.hsi
            HSI(shape=(4, 240, 240), dtype=torch.float32)
            >>> spatial_attribution.attributes.shape
            torch.Size([4, 240, 240])
            >>> spatial_attribution.segmentation_mask.shape
            torch.Size([1, 240, 240])
            >>> spatial_attribution.score
            1.0
        """
        if self._attribution_method is None or not isinstance(self._attribution_method, LimeBase):
            raise RuntimeError("Lime object not initialized")  # pragma: no cover

        if isinstance(hsi, HSI):
            hsi = [hsi]

        if not all(isinstance(hsi_image, HSI) for hsi_image in hsi):
            raise TypeError("All of the input hyperspectral images must be of type HSI")

        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(hsi[0], segmentation_method, **segmentation_method_params)

            logger.warning(
                "Segmentation mask is created based on the first HSI image provided, this approach may not be optimal as "
                "the same segmentation mask may not be the best suitable for all images",
            )

        if isinstance(segmentation_mask, tuple):
            segmentation_mask = tuple(segmentation_mask)
        elif not isinstance(segmentation_mask, list):
            segmentation_mask = [segmentation_mask] * len(hsi)

        if len(hsi) != len(segmentation_mask):
            raise ValueError(
                f"Number of segmentation masks should be equal to the number of HSI images provided, provided {len(segmentation_mask)}"
            )

        segmentation_mask = [
            ensure_torch_tensor(mask, f"Segmentation mask number {idx+1} should be None, numpy array, or torch tensor")
            for idx, mask in enumerate(segmentation_mask)
        ]
        segmentation_mask = [
            mask.unsqueeze(0).moveaxis(0, hsi_img.spectral_axis) if mask.ndim != hsi_img.image.ndim else mask
            for hsi_img, mask in zip(hsi, segmentation_mask)
        ]
        segmentation_mask = [
            validate_mask_shape("segmentation", hsi_img, mask) for hsi_img, mask in zip(hsi, segmentation_mask)
        ]

        hsi_input = torch.stack([hsi_img.get_image() for hsi_img in hsi], dim=0)
        segmentation_mask = torch.stack(segmentation_mask, dim=0)

        assert segmentation_mask.shape == hsi_input.shape

        segmentation_mask = segmentation_mask.to(self.device)
        hsi_input = hsi_input.to(self.device)

        lime_attributes, score = self._attribution_method.attribute(
            inputs=hsi_input,
            target=target,
            feature_mask=segmentation_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            additional_forward_args=additional_forward_args,
            show_progress=verbose,
            return_input_shape=True,
        )

        try:
            spatial_attribution = [
                HSISpatialAttributes(
                    hsi=hsi_img,
                    attributes=lime_attr,
                    mask=segmentation_mask[idx].expand_as(hsi_img.image),
                    score=score.item(),
                    attribution_method="Lime",
                )
                for idx, (hsi_img, lime_attr) in enumerate(zip(hsi, lime_attributes))
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error during creating spatial attribution {e}") from e

        return spatial_attribution[0] if len(spatial_attribution) == 1 else spatial_attribution

    def get_spectral_attributes(
        self,
        hsi: list[HSI] | HSI,
        band_mask: np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor] | None = None,
        target: list[int] | int | None = None,
        n_samples: int = 10,
        perturbations_per_eval: int = 4,
        verbose: bool = False,
        additional_forward_args: Any = None,
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int] | None = None,
    ) -> HSISpectralAttributes | list[HSISpectralAttributes]:
        """
        Attributes the hsi image using LIME method for spectral data. Based on the provided hsi and band mask, the LIME
        method attributes the hsi based on `superbands` (clustered bands) provided by the band mask.
        Please refer to the original paper `https://arxiv.org/abs/1602.04938` for more details or to
        Christoph Molnar's book `https://christophm.github.io/interpretable-ml-book/lime.html`.

        The function returns a HSISpectralAttributes object that contains the image, the attributions, the band mask,
        the band names, and the score of the interpretable model used for the explanation.

        Args:
            hsi (list[HSI] | HSI): Input hyperspectral image(s) for which the attributions are to be computed.
                If a list of HSI objects is provided, the attributions are computed for each HSI object in the list.
                The output will be a list of HSISpatialAttributes objects.
            band_mask (np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor] | None, optional): Band mask that
                is used for the spectral attribution. The band mask should have a 1D or 3D shape, which can be
                broadcastable to the shape of the input image. The only dimensions on which the image and the mask shapes
                can differ is the height and width dimensions, marked with letters `H` and `W` in the `image.orientation`
                parameter. If equals to None, the band mask is created within the function. If multiple HSI images are
                provided, a list of band masks can be provided, one for each image. If list is not provided method will
                assume that the same band mask is used for all images. Defaults to None.
            target (list[int] | int | None, optional): target class index for computing the attributions. If None,
                methods assume that the output has only one class. If the output has multiple classes, the target index
                must be provided. For multiple input images, a list of target indices can be provided, one for each
                image or single target value will be used for all images. Defaults to None.
            n_samples (int, optional): The number of samples to generate/analyze in LIME. The more the better but slower.
                Defaults to 10.
            perturbations_per_eval (int, optional): The number of perturbations to evaluate at once
                (Simply the inner batch size). Defaults to 4.
            verbose (bool, optional): Whether to show the progress bar. Defaults to False.
            segmentation_method (Literal["slic", "patch"], optional):
                Segmentation method used only if `segmentation_mask` is None. Defaults to "slic".
            additional_forward_args (Any, optional): If the forward function requires additional arguments other than
                the inputs for which attributions should not be computed, this argument can be provided.
                It must be either a single additional argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors or any arbitrary python types.
                These arguments are provided to forward_func in order following the arguments in inputs.
                Note that attributions are not computed with respect to these arguments. Default: None
            band_names (list[str] | dict[str | tuple[str, ...], int] | None, optional): Band names. Defaults to None.

        Returns:
            HSISpectralAttributes | list[HSISpectralAttributes]: An object containing the image, the attributions,
                the band mask, the band names, and the score of the interpretable model used for the explanation.

        Raises:
            RuntimeError: If the Lime object is not initialized or is not an instance of LimeBase.
            MaskCreationError: If there is an error creating the band mask.
            ValueError: If the number of band masks is not equal to the number of HSI images provided.
            HSIAttributesError: If there is an error during creating spectral attribution.

        Examples:
            >>> simple_model = lambda x: torch.rand((x.shape[0], 2))
            >>> hsi = mt.HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> band_mask = torch.randint(1, 4, (4, 1, 1)).repeat(1, 240, 240)
            >>> band_names = ["R", "G", "B"]
            >>> lime = meteors.attr.Lime(
                    explainable_model=ExplainableModel(simple_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.1)
                )
            >>> spectral_attribution = lime.get_spectral_attributes(hsi, band_mask=band_mask, band_names=band_names, target=0)
            >>> spectral_attribution.hsi
            HSI(shape=(4, 240, 240), dtype=torch.float32)
            >>> spectral_attribution.attributes.shape
            torch.Size([4, 240, 240])
            >>> spectral_attribution.band_mask.shape
            torch.Size([4, 240, 240])
            >>> spectral_attribution.band_names
            ["R", "G", "B"]
            >>> spectral_attribution.score
            1.0
        """

        if self._attribution_method is None or not isinstance(self._attribution_method, LimeBase):
            raise RuntimeError("Lime object not initialized")  # pragma: no cover

        if isinstance(hsi, HSI):
            hsi = [hsi]

        if not all(isinstance(hsi_image, HSI) for hsi_image in hsi):
            raise TypeError("All of the input hyperspectral images must be of type HSI")

        if band_mask is None:
            created_bands = [self.get_band_mask(hsi_img, band_names) for hsi_img in hsi]
            band_mask, band_name_list = zip(*created_bands)
            band_names = band_name_list[0]

        if isinstance(band_mask, tuple):
            band_mask = list(band_mask)
        elif not isinstance(band_mask, list):
            band_mask = [band_mask]

        if len(hsi) != len(band_mask):
            if len(band_mask) == 1:
                band_mask = band_mask * len(hsi)
                logger.debug("Reusing the same band mask for all images")
            else:
                raise ValueError(
                    f"Number of band masks should be equal to the number of HSI images provided, provided {len(band_mask)}"
                )

        band_mask = [
            ensure_torch_tensor(mask, f"Band mask number {idx+1} should be None, numpy array, or torch tensor")
            for idx, mask in enumerate(band_mask)
        ]
        band_mask = [
            mask.unsqueeze(-1).unsqueeze(-1).moveaxis(0, hsi_img.spectral_axis)
            if mask.ndim != hsi_img.image.ndim
            else mask
            for hsi_img, mask in zip(hsi, band_mask)
        ]
        band_mask = [validate_mask_shape("band", hsi_img, mask) for hsi_img, mask in zip(hsi, band_mask)]

        hsi_input = torch.stack([hsi_img.get_image() for hsi_img in hsi], dim=0)
        band_mask = torch.stack(band_mask, dim=0)

        if band_names is None:
            band_names = {str(segment): idx for idx, segment in enumerate(torch.unique(band_mask))}
        else:
            logger.debug(
                "Band names are provided and will be used. In the future, there should be an option to validate them."
            )

        assert hsi_input.shape == band_mask.shape

        hsi_input = hsi_input.to(self.device)
        band_mask = band_mask.to(self.device)

        lime_attributes, score = self._attribution_method.attribute(
            inputs=hsi_input,
            target=target,
            feature_mask=band_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            additional_forward_args=additional_forward_args,
            show_progress=verbose,
            return_input_shape=True,
        )

        try:
            spectral_attribution = [
                HSISpectralAttributes(
                    hsi=hsi_img,
                    attributes=lime_attr,
                    mask=band_mask[idx].expand_as(hsi_img.image),
                    band_names=band_names,
                    score=score.item(),
                    attribution_method="Lime",
                )
                for idx, (hsi_img, lime_attr) in enumerate(zip(hsi, lime_attributes))
            ]
        except Exception as e:
            raise HSIAttributesError(f"Error during creating spectral attribution {e}") from e

        return spectral_attribution[0] if len(spectral_attribution) == 1 else spectral_attribution

    @staticmethod
    def _get_slic_segmentation_mask(
        hsi: HSI, num_interpret_features: int = 10, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Creates a segmentation mask using the SLIC method.

        Args:
            hsi (HSI): An HSI object for which the segmentation mask is created.
            num_interpret_features (int, optional): Number of segments. Defaults to 10.
            *args: Additional positional arguments to be passed to the SLIC method.
            **kwargs: Additional keyword arguments to be passed to the SLIC method.

        Returns:
            torch.Tensor: An output segmentation mask.
        """
        segmentation_mask = slic(
            hsi.get_image().cpu().detach().numpy(),
            n_segments=num_interpret_features,
            mask=hsi.spatial_binary_mask.cpu().detach().numpy(),
            channel_axis=hsi.spectral_axis,
            *args,
            **kwargs,
        )

        if segmentation_mask.min() == 1:
            segmentation_mask -= 1

        segmentation_mask = torch.from_numpy(segmentation_mask)
        segmentation_mask = segmentation_mask.unsqueeze(dim=hsi.spectral_axis)

        return segmentation_mask

    @staticmethod
    def _get_patch_segmentation_mask(hsi: HSI, patch_size: int | float = 10, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Creates a segmentation mask using the patch method - creates small squares of the same size
            and assigns a unique value to each square.

        Args:
            hsi (HSI): An HSI object for which the segmentation mask is created.
            patch_size (int, optional): Size of the patch, the hsi size should be divisible by this value.
                Defaults to 10.

        Returns:
            torch.Tensor: An output segmentation mask.
        """
        if patch_size < 1 or not isinstance(patch_size, (int, float)):
            raise ValueError("Invalid patch_size. patch_size must be a positive integer")

        if hsi.image.shape[1] % patch_size != 0 or hsi.image.shape[2] % patch_size != 0:
            raise ValueError("Invalid patch_size. patch_size must be a factor of both width and height of the hsi")

        height, width = hsi.image.shape[1], hsi.image.shape[2]

        idx_mask = torch.arange(height // patch_size * width // patch_size, device=hsi.device).reshape(
            height // patch_size, width // patch_size
        )
        idx_mask += 1
        segmentation_mask = torch.repeat_interleave(idx_mask, patch_size, dim=0)
        segmentation_mask = torch.repeat_interleave(segmentation_mask, patch_size, dim=1)
        segmentation_mask = segmentation_mask * hsi.spatial_binary_mask
        # segmentation_mask = torch.repeat_interleave(
        # torch.unsqueeze(segmentation_mask, dim=hsi.spectral_axis),
        # repeats=hsi.image.shape[hsi.spectral_axis], dim=hsi.spectral_axis)
        segmentation_mask = segmentation_mask.unsqueeze(dim=hsi.spectral_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx

        return segmentation_mask
