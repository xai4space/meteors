from __future__ import annotations

from typing_extensions import Annotated, Self, Literal, Callable, Any, TypeVar, Type
import warnings

from abc import ABC
from loguru import logger
from functools import cached_property
from itertools import chain

import torch
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import spyndex

from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationInfo
from pydantic.functional_validators import BeforeValidator

from meteors import HSI
from meteors.lime_base import Lime as LimeBase
from meteors.utils.models import ExplainableModel, InterpretableModel, SkLearnLasso
from meteors.utils.utils import torch_dtype_to_python_dtype, change_dtype_of_list, expand_spectral_mask

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


def ensure_torch_tensor(value: np.ndarray | torch.Tensor, context: str) -> torch.Tensor:
    """Ensures the input is a PyTorch tensor, converting it if necessary.

    This function validates that the input is either a NumPy array or a PyTorch tensor,
    and converts NumPy arrays to PyTorch tensors. It's useful for standardizing inputs
    in functions that require PyTorch tensors.

    Args:
        value (np.ndarray | torch.Tensor): The input value to be validated and potentially converted.
        context (str): A string describing the context of the conversion, used in error and debug messages.

    Returns:
        torch.Tensor: The input value as a PyTorch tensor.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        logger.debug(f"Converting {context} from NumPy array to PyTorch tensor")
        return torch.from_numpy(value)

    raise TypeError(f"{context} must be a NumPy array or PyTorch tensor")


def validate_and_convert_attributes(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validates and converts the attributes to a PyTorch tensor.

    This function ensures that the input attributes are in the correct format
    (either a NumPy array or a PyTorch tensor) and converts them to a PyTorch
    tensor if necessary.

    Args:
        value (np.ndarray | torch.Tensor): The attributes to be validated and potentially converted.
            This can be either a NumPy array or a PyTorch tensor.

    Returns:
        torch.Tensor: The attributes as a PyTorch tensor.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    return ensure_torch_tensor(value, "Attributes")


def validate_and_convert_mask(value: np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
    """Ensures the `superpixel` or `superband` mask is a PyTorch tensor if provided, converting it if necessary.

    This function validates that the input mask is either a NumPy array
    or a PyTorch tensor, and converts it to a PyTorch tensor if it's a NumPy array.
    if the input is None, it returns None.

    Args:
        value (np.ndarray | torch.Tensor | None): The mask to be validated and potentially converted.

    Returns:
        torch.Tensor | None: The mask as a PyTorch tensor or None if the input is None.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    return ensure_torch_tensor(value, "`superpixel` or `superband` mask") if value is not None else None


def validate_shapes(attributes: torch.Tensor, hsi: HSI) -> None:
    """Validates that the shape of the attributes tensor matches the shape of the image.

    Args:
        attributes (torch.Tensor): The attributes tensor to validate.
        hsi (HSI): The hsi object to compare the shape with.

    Raises:
        ValueError: If the shape of the attributes tensor does not match the shape of the hsi.
    """
    if attributes.shape != hsi.image.shape:
        raise ValueError("Attributes must have the same shape as the hsi")


def align_band_names_with_mask(band_names: dict[str, int], band_mask: torch.Tensor) -> dict[str, int]:
    """Aligns the band names dictionary with the unique values in the band mask.

    This function ensures that the band_names dictionary correctly represents all unique
    values in the band_mask. It adds a 'not_included' category if necessary and validates
    that all mask values are accounted for in the band names.

    Args:
        band_names (dict[str, int]): A dictionary mapping band names to their corresponding
                                     integer values in the mask.
        band_mask (torch.Tensor): A tensor representing the band mask, where each unique
                                  integer corresponds to a different band or category.

    Returns:
        dict[str, int]: The updated band_names dictionary, potentially including a
                        'not_included' category if 0 is present in the mask but not in
                        the original band_names.

    Raises:
        ValueError: If the set of values in band_names doesn't match the unique values
                    in the band_mask after accounting for the 'not_included' category.

    Warns:
        UserWarning: If a 'not_included' category (0) is added to the band_names.

    Notes:
        - The function assumes that 0 in the mask represents 'not_included' areas if
          not explicitly defined in the input band_names.
        - All unique values in the mask must be present in the band_names dictionary
          after the alignment process.
    """
    unique_mask_values = set(band_mask.unique().int().tolist())
    band_name_values = set(band_names.values())

    # Check if 0 is in the mask but not in band_names
    if 0 in unique_mask_values and 0 not in band_name_values:
        warnings.warn(
            "Band mask contains `0` values which are not covered by the provided band names. "
            "Adding 'not_included' to band names."
        )
        if "not_included" in band_names:
            raise ValueError(
                "Band names should not contain 'not_included' if 0 is present in the mask and not in band names"
            )
        band_names["not_included"] = 0
        band_name_values.add(0)

    # Validate that all mask values are in band_names
    if unique_mask_values != band_name_values:
        raise ValueError("Band names should have all unique values in mask")

    return band_names


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
                warnings.warn(f"Adjusting segment start from {start} to 0")
                start = 0
            else:
                raise ValueError(f"Segment range {(start, end)} is out of bounds")

        if end > max_index:
            if start < max_index:
                warnings.warn(f"Adjusting segment end from {(start, end)} to {(start, max_index)}")
                end = max_index
            else:
                raise ValueError(f"Segment range {(start, end)} is out of bounds")
        adjusted_ranges.append((start, end))
    return adjusted_ranges  # type: ignore


def validate_tensor(value: Any, error_message: str) -> torch.Tensor:
    """Validates the input value and converts it to a torch.Tensor if necessary.

    Args:
        value (Any): The input value to be validated.
        error_message (str): The error message to be raised if the value is not valid.

    Returns:
        torch.Tensor: The validated and converted tensor.

    Raises:
        TypeError: If the value is not an instance of np.ndarray or torch.Tensor.
    """
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(error_message)
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    return value


def validate_segment_range(wavelengths: torch.Tensor, segment_range: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Validates the segment range and adjusts it if possible.

    Args:
        wavelengths (torch.Tensor): The wavelengths tensor.
        segment_range (list[tuple[int, int]]): The segment range to be validated.

    Returns:
        list[tuple[int, int]]: The validated segment range.

    Raises:
        ValueError: If the segment range is out of bounds.
    """
    wavelengths_max_index = wavelengths.shape[0]
    out_segment_range = []
    for segment in segment_range:
        new_segment: list[int] = list(segment)
        if new_segment[0] < 0:
            if new_segment[1] >= 1:
                new_segment[0] = 0
            else:
                raise ValueError(f"Segment range {segment} is out of bounds")

        if new_segment[1] > wavelengths_max_index:
            if new_segment[0] <= wavelengths_max_index - 1:
                new_segment[1] = wavelengths_max_index
            else:
                raise ValueError(f"Segment range {segment} is out of bounds")
        out_segment_range.append(tuple(new_segment))
    return out_segment_range  # type: ignore


def resolve_inference_device(device: str | torch.device | None, info: ValidationInfo) -> torch.device:
    """Resolves and returns the device to be used for inference.

    This function determines the appropriate PyTorch device for inference based on the input
    parameters and available information. It handles three scenarios:
    1. If a specific device is provided, it validates and returns it.
    2. If no device is specified (None), it uses the device of the input hsi.
    3. If a string is provided, it attempts to convert it to a torch.device.

    Args:
        device (str | torch.device | None): The desired device for inference.
            If None, the device of the input hsi will be used.
        info (ValidationInfo): An object containing additional validation information,
            including the input hsi data.

    Returns:
        torch.device: The resolved PyTorch device for inference.

    Raises:
        ValueError: If no device is specified and the hsi is not present in the info data,
            or if the provided device string is invalid.
        TypeError: If the provided device is neither None, a string, nor a torch.device.
    """
    if device is None:
        if "hsi" not in info.data:
            raise ValueError("HSI is not present in the data, INTERNAL ERROR")
        hsi: torch.Tensor = info.data["hsi"]
        device = hsi.device
    elif isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValueError(f"Device {device} is not valid") from e
    if not isinstance(device, torch.device):
        raise TypeError("Device should be a string or torch device")

    logger.debug(f"Device for inference: {device.type}")
    return device


######################################################################
############################ EXPLANATIONS ############################
######################################################################


class HSIAttributes(BaseModel):
    """Represents an object that contains hsi attributes and explanations.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        mask (torch.Tensor | None): `superpixel` or `superband` mask used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
    """

    hsi: Annotated[
        HSI,
        Field(
            description="Hyperspectral image object for which the explanations were created.",
        ),
    ]
    attributes: Annotated[
        torch.Tensor,
        BeforeValidator(validate_and_convert_attributes),
        Field(
            description="Attributions (explanations) for the hsi.",
        ),
    ]
    score: Annotated[
        float,
        Field(
            le=1.0,
            description="R^2 score of interpretable model used for the explanation.",
        ),
    ]
    mask: Annotated[
        torch.Tensor | None,
        BeforeValidator(validate_and_convert_mask),
        Field(
            description="`superpixel` or `superband` mask used for the explanation.",
        ),
    ] = None
    device: Annotated[
        torch.device,
        BeforeValidator(resolve_inference_device),
        Field(
            validate_default=True,
            exclude=True,
            description=(
                "Device to be used for inference. If None, the device of the input hsi will be used. "
                "Defaults to None."
            ),
        ),
    ] = None

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        This method should be implemented in the subclass.

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _validate_hsi_attributions_and_mask(self) -> None:
        """Validates the hsi attributions with mask and performs necessary operations to ensure compatibility with the
        device.

        Raises:
            ValueError: If the shapes of the attributes and hsi tensors do not match.
            ValueError: If the mask is provided and its shape does not match the hsi.
        """
        validate_shapes(self.attributes, self.hsi)

        self.attributes = self.attributes.to(self.device)
        if self.device != self.hsi.device:
            self.hsi.to(self.device)

        if self.mask is not None:
            validate_shapes(self.mask, self.hsi)
            self.mask = self.mask.to(self.device)

    @model_validator(mode="after")
    def validate_hsi_attributions(self) -> Self:
        """Validates the hsi attributions.

        This method performs validation on the hsi attributions to ensure they are correct.

        Returns:
            Self: The current instance of the class.
        """
        self._validate_hsi_attributions_and_mask()
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the hsi and attributes tensors to the specified device.

        Args:
            device (str or torch.device): The device to move the tensors to.

        Returns:
            Self: The modified object with tensors moved to the specified device.

        Examples:
            >>> attrs = HSIAttributes(hsi, attributes, score=0.5)
            >>> attrs.to("cpu")
            >>> attrs.hsi.device
            device(type='cpu')
            >>> attrs.attributes.device
            device(type='cpu')
            >>> attrs.to("cuda")
            >>> attrs.hsi.device
            device(type='cuda')
            >>> attrs.attributes.device
            device(type='cuda')
        """
        self.hsi = self.hsi.to(device)
        self.attributes = self.attributes.to(device)
        self.device = self.hsi.device
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self

    def change_orientation(self, target_orientation: tuple[str, str, str] | list[str] | str, inplace=False) -> Self:
        """Changes the orientation of the image data along with the attributions to the target orientation.

        Args:
            target_orientation (tuple[str, str, str] | list[str] | str): The target orientation for the attribution data.
                This should be a tuple of three one-letter strings in any order: "C", "H", "W".
            inplace (bool, optional): Whether to modify the data in place or return a new object.

        Returns:
            Self: The updated Image object with the new orientation.

        Raises:
            ValueError: If the target orientation is not a valid tuple of three one-letter strings.
        """
        current_orientation = self.hsi.orientation
        hsi = self.hsi.change_orientation(target_orientation, inplace=inplace)
        if inplace:
            attrs = self
        else:
            attrs = self.model_copy()
            attrs.hsi = hsi

        if current_orientation != attrs.hsi.orientation:
            new_orientation = attrs.hsi.orientation
            attrs.attributes = attrs.attributes.permute(
                current_orientation.index(new_orientation[0]),
                current_orientation.index(new_orientation[1]),
                current_orientation.index(new_orientation[2]),
            )

            if self.mask is not None:
                attrs.mask = self.mask.permute(
                    current_orientation.index(new_orientation[0]),
                    current_orientation.index(new_orientation[1]),
                    current_orientation.index(new_orientation[2]),
                )

        return attrs


class HSISpatialAttributes(HSIAttributes):
    """Represents spatial attributes of an hsi used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        mask (torch.Tensor | None): Original Spatial (Segmentation) mask used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
        segmentation_mask (torch.Tensor): Spatial (Segmentation) mask used for the explanation.
    """

    @property
    def segmentation_mask(self) -> torch.Tensor:
        """Returns the spatial segmentation mask.

        This method selects the segmentation mask in 2D format if it has 2 dimensions,
        if its dimension matches the hsi, it selects the first index along the spectral axis.
        else, it raises a ValueError.

        Returns:
            torch.Tensor: The spatial segmentation mask.

        Raises:
            ValueError: If the segmentation mask is not provided.
            ValueError: If the segmentation mask has an unsupported number of dimensions.

        Examples:
            >>> segmentation_mask = torch.zeros((3, 2, 2))
            >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
            >>> attrs.spatial_segmentation_mask
            tensor([[0., 0.],
                    [0., 0.]])
        """
        if self.mask is None:
            raise ValueError("Segmentation mask is not provided")
        return self.mask.select(dim=self.hsi.spectral_axis, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        In the case of spatial attributes, the flattened attributes are 2D spatial attributes.

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        >>> segmentation_mask = torch.zeros((3, 2, 2))
        >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
        >>> attrs.flattened_attributes
            tensor([[0., 0.],
                    [0., 0.]])
        """
        return self.attributes.select(dim=self.hsi.spectral_axis, index=0)


class HSISpectralAttributes(HSIAttributes):
    """Represents an hsi with spectral attributes used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        mask (torch.Tensor | None): Original Band mask used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
        band_names (dict[str, int]): Dictionary that translates the band names into the band segment ids.
    """

    band_names: Annotated[
        dict[str, int],
        Field(
            description="Dictionary that translates the band names into the band segment ids.",
        ),
    ]

    @property
    def band_mask(self) -> torch.Tensor:
        """Returns a band mask.

        The method selects the appropriate dimensions from the `band_mask` tensor
        based on the `axis_to_select` and returns a flattened version of the selected
        tensor.

        Returns:
            torch.Tensor: The flattened band mask tensor.

        Raises:
            ValueError: If the band mask is not provided.

        Examples:
            >>> band_names = {"R": 0, "G": 1, "B": 2}
            >>> attrs = HSISpectralAttributes(hsi, attributes, score=0.5, band_mask=band_mask)
            >>> attrs.spectral_band_mask
            torch.tensor([0, 1, 2])
        """
        if self.mask is None:
            raise ValueError("Band mask is not provided")
        axis = list(range(self.mask.ndim))
        axis.remove(self.hsi.spectral_axis)
        return self.mask.select(dim=axis[0], index=0).select(dim=axis[1] - 1, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        In the case of spectral attributes, the flattened attributes are 1D spectral attributes.

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        axis = list(range(self.attributes.ndim))
        axis.remove(self.hsi.spectral_axis)
        return self.attributes.select(dim=axis[0], index=0).select(dim=axis[1] - 1, index=0)

    def _validate_hsi_attributions_and_mask(self) -> None:
        super()._validate_hsi_attributions_and_mask()
        if self.mask is not None:
            align_band_names_with_mask(self.band_names, self.mask)


###################################################################
############################ EXPLAINER ############################
###################################################################


class Explainer(ABC):
    """Explainer class for explaining models.

    Args:
        explainable_model (ExplainableModel): The explainable model to be explained.
        interpretable_model (InterpretableModel): The interpretable model used to approximate the black-box model

    Notes:
        - Current supported interpretable models are `LinearModel`, `SkLearnLinearModel`,
            `SkLearnLasso` and `SkLearnRidge`.
    """

    def __init__(self, explainable_model: ExplainableModel, interpretable_model: InterpretableModel):
        self.explainable_model = explainable_model
        self.interpretable_model = interpretable_model

    @cached_property
    def device(self) -> torch.device:
        """Get the device on which the explainable model is located.

        Returns:
            torch.device: The device on which the explainable model is located.
        """
        try:
            device = next(self.explainable_model.forward_func.parameters()).device  # type: ignore
        except Exception:
            logger.debug("Could not extract device from the explainable model, setting device to cpu")
            logger.warning("Not a torch model, setting device to cpu")
            device = torch.device("cpu")
        return device

    def to(self, device: str | torch.device) -> Self:
        """Move the explainable model to the specified device.

        Args:
            device (str or torch.device): The device to move the explainable model to.

        Returns:
            Self: The updated Explainer instance.
        """
        self.explainable_model = self.explainable_model.to(device)
        return self


class Lime(Explainer):
    """Lime class is a subclass of Explainer and represents the Lime explainer. Lime is an interpretable model-agnostic
    explanation method that explains the predictions of a black-box model by approximating it with a simpler
    interpretable model.

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
        super().__init__(explainable_model, interpretable_model)
        self._lime = self._construct_lime(
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
            hsi (HSI): The input hsi for which the segmentation mask needs to be generated.
            segmentation_method (Literal["patch", "slic"], optional): The segmentation method to be used.
                Defaults to "slic".
            **segmentation_method_params (Any): Additional parameters specific to the chosen segmentation method.

        Returns:
            torch.Tensor: The segmentation mask as a tensor.

        Raises:
            ValueError: If the input hsi is not an instance of the HSI class.
            ValueError: If an unsupported segmentation method is specified.

        Examples:
            >>> hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
            >>> segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="slic")
            >>> segmentation_mask.shape
            torch.Size([1, 240, 240])
            >>> segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="patch", patch_size=2)
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
            raise ValueError("Image should be an instance of Image class")

        if segmentation_method == "slic":
            return Lime._get_slick_segmentation_mask(hsi, **segmentation_method_params)
        elif segmentation_method == "patch":
            return Lime._get_patch_segmentation_mask(hsi, **segmentation_method_params)
        else:
            raise ValueError(f"Unsupported segmentation method: {segmentation_method}")

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
            hsi (HSI): The input hsi.
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
            ValueError: If the input hsi is not an instance of the HSI class.
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
            raise ValueError("hsi should be an instance of HSI class")

        assert (
            band_names is not None or band_indices is not None or band_wavelengths is not None
        ), "No band names, indices, or wavelengths are provided."

        # validate types
        dict_labels_to_segment_ids = None
        if band_names is not None:
            logger.debug("Getting band mask from band names of spectral bands")
            if band_wavelengths is not None or band_indices is not None:
                logger.info("Only the band names will be used to create the band mask")
            try:
                validate_band_names(band_names)
                band_groups, dict_labels_to_segment_ids = Lime._get_band_wavelengths_indices_from_band_names(
                    hsi.wavelengths, band_names
                )
            except Exception as e:
                raise ValueError(f"Incorrect band names provided: {e}") from e
        elif band_wavelengths is not None:
            logger.debug("Getting band mask from band groups given by ranges of wavelengths")
            if band_indices is not None:
                logger.info("Only the band wavelengths will be used to create the band mask")
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

    @staticmethod
    def _make_band_names_indexable(segment_name: list[str] | tuple[str, ...] | str) -> tuple[str, ...] | str:
        """Converts a list of strings into a tuple of strings if necessary to make it indexable.

        Args:
            segment_name (list[str] | tuple[str, ...] | str): The segment name to be converted.

        Returns:
            tuple[str, ...] | str: The converted segment name.

        Raises:
            ValueError: If the segment_name is not of type list or string.
        """
        if (
            isinstance(segment_name, tuple) and all(isinstance(subitem, str) for subitem in segment_name)
        ) or isinstance(segment_name, str):
            return segment_name
        elif isinstance(segment_name, list) and all(isinstance(subitem, str) for subitem in segment_name):
            return tuple(segment_name)
        raise ValueError(f"Incorrect segment {segment_name} type. Should be either a list or string")

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
            ValueError: If the provided band name is invalid.
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
                raise ValueError(
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
            raise ValueError("Incorrect band_names type. It should be a dict or a list")
        segments_list_after_mapping = [Lime._extract_bands_from_spyndex(segment) for segment in segments_list]
        band_indices: dict[tuple[str, ...] | str, list[int]] = {}
        for original_segment, segment in zip(segments_list, segments_list_after_mapping):
            try:
                segment_indices_ranges: list[tuple[int, int]] = []
                for band_name in segment:
                    segment_indices_ranges += Lime._convert_wavelengths_to_indices(
                        wavelengths, (spyndex.bands[band_name].min_wavelength, spyndex.bands[band_name].max_wavelength)
                    )

                segment_list = Lime._get_indices_from_wavelength_indices_range(wavelengths, segment_indices_ranges)
                band_indices[original_segment] = segment_list
            except Exception as e:
                raise ValueError(f"Problem with segment {original_segment} and bands {segment}") from e
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
            ValueError: If band_wavelengths is not a dictionary.
        """
        if not isinstance(band_wavelengths, dict):
            raise ValueError("band_wavelengths should be a dictionary")

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
            ValueError: If `band_indices` is not a dictionary.
        """
        if not isinstance(input_band_indices, dict):
            raise ValueError("band_indices should be a dictionary")

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
    def _check_overlapping_segments(hsi: HSI, dict_labels_to_indices: dict[str | tuple[str, ...], list[int]]) -> None:
        """Check for overlapping segments in the given hsi.

        Args:
            hsi (HSI): The hsi object containing the wavelengths.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
                A dictionary mapping segment labels to indices.

        Returns:
            None
        """
        overlapping_segments: dict[int, str | tuple[str, ...]] = {}
        for segment_label, indices in dict_labels_to_indices.items():
            for idx in indices:
                if hsi.wavelengths[idx].item() in overlapping_segments.keys():
                    logger.warning(
                        (
                            f"Segments {overlapping_segments[hsi.wavelengths[idx].item()]} "
                            f"and {segment_label} are overlapping on wavelength {hsi.wavelengths[idx].item()}"
                        )
                    )
                overlapping_segments[hsi.wavelengths[idx].item()] = segment_label

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
        Lime._check_overlapping_segments(hsi, dict_labels_to_indices)

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

    def get_spatial_attributes(
        self,
        hsi: HSI,
        segmentation_mask: np.ndarray | torch.Tensor | None = None,
        target: int | None = None,
        n_samples: int = 10,
        perturbations_per_eval: int = 4,
        verbose: bool = False,
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        segmentation_method: Literal["slic", "patch"] = "slic",
        **segmentation_method_params: Any,
    ) -> HSISpatialAttributes:
        """
        Get spatial attributes of an hsi image using the LIME method. Based on the provided hsi and segmentation mask
        LIME method attributes the `superpixels` provided by the segmentation mask. Please refer to the original paper
        `https://arxiv.org/abs/1602.04938` for more details or to Christoph Molnar's book
        `https://christophm.github.io/interpretable-ml-book/lime.html`.

        This function attributes the hsi using the LIME (Local Interpretable Model-Agnostic Explanations)
        method for spatial data. It returns an `HSISpatialAttributes` object that contains the hsi,
        the attributions, the segmentation mask, and the score of the interpretable model used for the explanation.

        Args:
            hsi (HSI): An HSI object for which the attribution is performed.
            segmentation_mask (np.ndarray | torch.Tensor | None, optional):
                A segmentation mask according to which the attribution should be performed.
                If None, a new segmentation mask is created using the `segmentation_method`.
                    Additional parameters for the segmentation method may be passed as kwargs. Defaults to None.
            target (int, optional): If the model creates more than one output, it analyzes the given target.
                Defaults to None.
            n_samples (int, optional): The number of samples to generate/analyze in LIME. The more the better but slower. Defaults to 10.
            perturbations_per_eval (int, optional): The number of perturbations to evaluate at once (Simply the inner batch size).
                Defaults to 4.
            verbose (bool, optional): Whether to show the progress bar. Defaults to False.
                postprocessing_segmentation_output (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None): A
                segmentation postprocessing function for segmentation problem type. This is required for segmentation problem type as
                lime surrogate model needs to be optimized on the 1d output, and the model should be able to modify the model output with
                inner lime active region mask as input and return the 1d output (for example number of pixel for each class) and not class mask.
                   Defaults to None.
            segmentation_method (Literal["slic", "patch"], optional):
                Segmentation method used only if `segmentation_mask` is None. Defaults to "slic".
            **segmentation_method_params (Any): Additional parameters for the segmentation method.

        Returns:
            HSISpatialAttributes: An `HSISpatialAttributes` object that contains the hsi, the attributions,
                the segmentation mask, and the score of the interpretable model used for the explanation.

        Raises:
            ValueError: If the Lime object is not initialized or is not an instance of LimeBase.
            AssertionError: If explainable model type is `segmentation` and `postprocessing_segmentation_output` is not provided.
            AssertionError: If the hsi is not an instance of the HSI class.

        Examples:
            >>> simple_model = lambda x: torch.rand((x.shape[0], 2))
            >>> hsi = mt.HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> segmentation_mask = torch.randint(1, 4, (1, 240, 240))
            >>> lime = mt_lime.Lime(
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
        if self._lime is None or not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")  # pragma: no cover

        assert isinstance(hsi, HSI), "hsi should be an instance of HSI class"

        if self.explainable_model.problem_type == "segmentation":
            assert postprocessing_segmentation_output, (
                "postprocessing_segmentation_output is required for segmentation problem type, please provide "
                "the `postprocessing_segmentation_output`. For a reference "
                "we provided an example function to use `agg_segmentation_postprocessing` from `meteors.utils.utils` module"
            )
        elif postprocessing_segmentation_output is not None:
            logger.warning(
                "postprocessing_segmentation_output is provided but the problem is not segmentation, will be ignored"
            )
            postprocessing_segmentation_output = None

        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(hsi, segmentation_method, **segmentation_method_params)
        segmentation_mask = ensure_torch_tensor(
            segmentation_mask, "Segmentation mask should be None, numpy array, or torch tensor"
        )
        if segmentation_mask.ndim != hsi.image.ndim:
            segmentation_mask = segmentation_mask.unsqueeze(dim=hsi.spectral_axis)

        hsi = hsi.to(self.device)
        segmentation_mask = segmentation_mask.to(self.device)

        lime_attributes, score = self._lime.attribute(
            inputs=hsi.get_image().unsqueeze(0),
            target=target,
            feature_mask=segmentation_mask.unsqueeze(0),
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            model_postprocessing=postprocessing_segmentation_output,
            show_progress=verbose,
            return_input_shape=True,
        )

        spatial_attribution = HSISpatialAttributes(
            hsi=hsi,
            attributes=lime_attributes.squeeze(0),
            mask=segmentation_mask.expand_as(hsi.image),
            score=score,
        )

        return spatial_attribution

    def get_spectral_attributes(
        self,
        hsi: HSI,
        band_mask: np.ndarray | torch.Tensor | None = None,
        target=None,
        n_samples: int = 10,
        perturbations_per_eval: int = 4,
        verbose: bool = False,
        postprocessing_segmentation_output: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int] | None = None,
    ) -> HSISpectralAttributes:
        """
        Attributes the hsi image using LIME method for spectral data. Based on the provided hsi and band mask, the LIME
        method attributes the hsi based on `superbands` (clustered bands) provided by the band mask.
        Please refer to the original paper `https://arxiv.org/abs/1602.04938` for more details or to
        Christoph Molnar's book `https://christophm.github.io/interpretable-ml-book/lime.html`.

        The function returns an HSISpectralAttributes object that contains the hsi, the attributions, the band mask,
        the band names, and the score of the interpretable model used for the explanation.

        Args:
            hsi (HSI): An HSI object for which the attribution is performed.
            band_mask (np.ndarray | torch.Tensor | None, optional): Band mask that is used for the spectral attribution.
                If equals to None, the band mask is created within the function. Defaults to None.
            target (int, optional): If the model creates more than one output, it analyzes the given target.
                Defaults to None.
            n_samples (int, optional): The number of samples to generate/analyze in LIME. The more the better but slower. Defaults to 10.
            perturbations_per_eval (int, optional): The number of perturbations to evaluate at once (Simply the inner batch size).
                Defaults to 4.
            verbose (bool, optional): Specifies whether to show progress during the attribution process. Defaults to False.
            postprocessing_segmentation_output: (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None):
                A segmentation postprocessing function for segmentation problem type. This is required for segmentation problem type as
                lime surrogate model needs to be optimized on the 1d output, and the model should be able to modify the model output with
                inner lime active region mask as input and return the 1d output (for example number of pixel for each class) and not class mask.
                Defaults to None.
            band_names (list[str] | dict[str | tuple[str, ...], int] | None, optional): Band names. Defaults to None.

        Returns:
            HSISpectralAttributes: An HSISpectralAttributes object containing the hsi, the attributions,
                the band mask, the band names, and the score of the interpretable model used for the explanation.

        Raises:
            ValueError: If the Lime object is not initialized or is not an instance of LimeBase.
            AssertionError: If explainable model type is `segmentation` and `postprocessing_segmentation_output` is not provided.
            AssertionError: If the hsi is not an instance of the HSI class.

        Examples:
            >>> simple_model = lambda x: torch.rand((x.shape[0], 2))
            >>> hsi = mt.HSI(image=torch.ones((4, 240, 240)), wavelengths=[462.08, 465.27, 468.47, 471.68])
            >>> band_mask = torch.randint(1, 4, (4, 1, 1)).repeat(1, 240, 240)
            >>> band_names = ["R", "G", "B"]
            >>> lime = mt_lime.Lime(
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

        if self._lime is None or not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")  # pragma: no cover

        if self.explainable_model.problem_type == "segmentation":
            assert postprocessing_segmentation_output, (
                "postprocessing_segmentation_output is required for segmentation problem type, please provide "
                "the `postprocessing_segmentation_output`. For a reference "
                "we provided an example function to use `agg_segmentation_postprocessing` from `meteors.utils.utils` module"
            )
        elif postprocessing_segmentation_output is not None:
            logger.warning(
                "postprocessing_segmentation_output is provided but the problem is not segmentation, will be ignored"
            )
            postprocessing_segmentation_output = None

        assert isinstance(hsi, HSI), "hsi should be an instance of HSI class"

        if band_mask is None:
            band_mask, band_names = self.get_band_mask(hsi, band_names)
        band_mask = ensure_torch_tensor(band_mask, "Band mask should be None, numpy array, or torch tensor")
        if band_mask.shape != hsi.image.shape:
            band_mask = expand_spectral_mask(hsi, band_mask, repeat_dimensions=True)
        band_mask = band_mask.int()

        if band_names is None:
            unique_segments = torch.unique(band_mask)
            band_names = {str(segment): idx for idx, segment in enumerate(unique_segments)}
        else:
            # checking consistency of names
            # unique_segments = torch.unique(band_mask)
            # if isinstance(band_names, dict):
            #     assert set(unique_segments).issubset(set(band_names.values())), "Incorrect band names"
            logger.debug(
                "Band names are provided and will be used. In the future, there should be an option to validate them."
            )

        hsi = hsi.to(self.device)
        band_mask = band_mask.to(self.device)

        lime_attributes, score = self._lime.attribute(
            inputs=hsi.get_image().unsqueeze(0),
            target=target,
            feature_mask=band_mask.unsqueeze(0),
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            model_postprocessing=postprocessing_segmentation_output,
            show_progress=verbose,
            return_input_shape=True,
        )

        spectral_attribution = HSISpectralAttributes(
            hsi=hsi,
            attributes=lime_attributes.squeeze(0),
            mask=band_mask.expand_as(hsi.image),
            band_names=band_names,
            score=score,
        )

        return spectral_attribution

    @staticmethod
    def _get_slick_segmentation_mask(
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
            mask=hsi.spatial_mask.cpu().detach().numpy(),
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
        segmentation_mask = segmentation_mask * hsi.spatial_mask
        # segmentation_mask = torch.repeat_interleave(
        # torch.unsqueeze(segmentation_mask, dim=hsi.spectral_axis),
        # repeats=hsi.image.shape[hsi.spectral_axis], dim=hsi.spectral_axis)
        segmentation_mask = segmentation_mask.unsqueeze(dim=hsi.spectral_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx

        return segmentation_mask
