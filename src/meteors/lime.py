from __future__ import annotations

from typing_extensions import Annotated, Self, Literal, Callable, Any, TypeVar
from abc import ABC
from loguru import logger
from functools import cached_property, lru_cache
from itertools import chain

import torch
import numpy as np
import spyndex
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.functional_validators import BeforeValidator

from meteors import Image
from meteors.image import validate_device
from meteors.lime_base import Lime as LimeBase
from meteors.utils.models import ExplainableModel, InterpretableModel

try:
    from fast_slic import Slic as slic
except ImportError:
    from skimage.segmentation import slic

# Constants
HSI_AXIS_ORDER = [2, 1, 0]  # (bands, rows, columns)

# Types
IntOrFloat = TypeVar("IntOrFloat", int, float)

#####################################################################
############################ VALIDATIONS ############################
#####################################################################


def validate_torch_tensor_type(value: np.ndarray | torch.Tensor, error_message: str) -> torch.Tensor:
    """Validates the type of the input value and converts it to a torch.Tensor if necessary.

    Args:
        value (np.ndarray | torch.Tensor): The input value to be validated.
        error_message (str): The error message to be raised if the value is not of the expected type.

    Returns:
        torch.Tensor: The input value converted to a torch.Tensor if necessary.

    Raises:
        TypeError: If the value is not of type np.ndarray or torch.Tensor.
    """
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(error_message)

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    return value


def validate_attributes(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validates the attributes value.

    Args:
        value (np.ndarray | torch.Tensor): The attributes value to be validated.

    Returns:
        torch.Tensor: The validated attributes value.

    Raises:
        TypeError: If the attributes value is not a numpy array or torch tensor.
    """
    return validate_torch_tensor_type(value, "Attributes must be either numpy array or torch tensor")


def validate_segmentation_mask(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validates the segmentation mask.

    Args:
        value (np.ndarray | torch.Tensor): The segmentation mask to be validated.

    Returns:
        torch.Tensor: The validated segmentation mask.

    Raises:
        TypeError: If the segmentation mask is not a numpy array or torch tensor.
    """
    return validate_torch_tensor_type(value, "Segmentation mask must be either numpy array or torch tensor")


def validate_band_mask(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validates the band mask.

    Args:
        value (np.ndarray | torch.Tensor): The band mask to be validated.

    Returns:
        torch.Tensor: The validated band mask.

    Raises:
        TypeError: If the band mask is neither a numpy array nor a torch tensor.
    """
    return validate_torch_tensor_type(value, "Band mask must be either numpy array or torch tensor")


def validate_shapes(attributes: torch.Tensor, image: Image) -> None:
    """Validates that the shape of the attributes tensor matches the shape of the image.

    Args:
        attributes (torch.Tensor): The attributes tensor to validate.
        image (Image): The image object to compare the shape with.

    Raises:
        ValueError: If the shape of the attributes tensor does not match the shape of the image.
    """
    if attributes.shape != image.image.shape:
        raise ValueError("Attributes must have the same shape as the image")


def validate_band_names_with_mask(band_names: dict[str, int], band_mask: torch.Tensor) -> dict[str, int]:
    """Validates the band names with a band mask and adds a 'not_included' key to the band_names dictionary if
    necessary.

    Args:
        band_names (dict[str, int]): A dictionary containing band names as keys and their corresponding values.
        band_mask (torch.Tensor): A tensor representing the band mask.

    Returns:
        dict[str, int]: The updated band_names dictionary.
    """
    if 0 not in band_names.values() and 0 in torch.unique(band_mask):
        band_names["not_included"] = 0
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
        if not all(isinstance(k, (tuple, str)) and isinstance(v, int) for k, v in band_names.items()):
            raise TypeError("All keys in band_names must be tuple or str, and all values must be int.")
    elif isinstance(band_names, list):
        if not all(isinstance(item, (str, list)) for item in band_names):
            raise TypeError("All items in band_names list should be str or list.")
    else:
        raise TypeError("band_names should be either a list or a dictionary.")


def validate_band_ranges(band_ranges: Any, variable_name: str) -> None:
    """Validates the band ranges dictionary.

    Args:
        band_ranges (Any): The band ranges dictionary to validate.
        variable_name (str): The name of the variable being validated.

    Raises:
        TypeError: If the band_ranges dictionary is not of the expected format.

    Returns:
        None
    """
    if not isinstance(band_ranges, dict) or not all(
        isinstance(k, (tuple, str)) and isinstance(v, (list, tuple)) for k, v in band_ranges.items()
    ):
        raise TypeError(
            f"{variable_name} should be a dictionary with keys of type tuple or str and values of type list or tuple."
        )


def validate_segment_format_range(
    segment_range: tuple[int, int] | list[tuple[int, int]] | list[int],
) -> list[tuple[int, int]]:
    """Validates the format of the segment range.

    Args:
        segment_range (tuple[int, int] | list[tuple[int, int]] | list[int]): The segment range to validate.

    Returns:
        list[tuple[int, int]]: The validated segment range.

    Raises:
        ValueError: If the segment range is not in the correct format.
    """
    if isinstance(segment_range, list) and all(isinstance(x, int) for x in segment_range):
        segment_range = [(x, x + 1) for x in segment_range]  # type: ignore
    elif (
        isinstance(segment_range, tuple) and len(segment_range) == 2 and all(isinstance(x, int) for x in segment_range)
    ):
        segment_range = [segment_range]  # Standardize single tuple to list of tuples
    elif not (
        isinstance(segment_range, list)
        and all(
            isinstance(part, tuple) and len(part) == 2 and all(isinstance(x, int) for x in part) and part[0] < part[1]
            for part in segment_range
        )
    ):
        raise ValueError(
            f"Each segment range should be a tuple or list of two numbers (start, end). Where start < end. But got: {segment_range}"
        )
    return segment_range  # type: ignore


def validate_segment_range(
    wavelengths: torch.Tensor, segment_range: list[tuple[IntOrFloat, IntOrFloat]]
) -> list[tuple[IntOrFloat, IntOrFloat]]:
    """Validates the segment range and adjusts it if possible.

    Args:
        wavelengths (torch.Tensor): The wavelengths tensor.
        segment_range (list[tuple[IntOrFloat, IntOrFloat]]): The segment range to be validated.

    Returns:
        list[tuple[IntOrFloat, IntOrFloat]]: The validated segment range.

    Raises:
        ValueError: If the segment range is out of bounds.
    """
    wavelengths_max_index = wavelengths.shape[0]
    out_segment_range = []
    for segment in segment_range:
        new_segment: list[IntOrFloat] = list(segment)
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


######################################################################
############################ EXPLANATIONS ############################
######################################################################


class ImageAttributes(BaseModel):
    """Represents an object that contains image attributes and explanations.

    Attributes:
        image (Image): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the image.
        score (float): R^2 score of interpretable model used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input image will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
    """

    image: Annotated[
        Image,
        Field(
            description="Hyperspectral image object for which the explanations were created.",
        ),
    ]
    attributes: Annotated[
        torch.Tensor,
        BeforeValidator(validate_attributes),
        Field(
            description="Attributions (explanations) for the image.",
        ),
    ]
    score: Annotated[
        float,
        Field(
            le=1.0,
            description="R^2 score of interpretable model used for the explanation.",
        ),
    ]
    device: Annotated[
        torch.device,
        BeforeValidator(validate_device),
        Field(
            validate_default=True,
            exclude=True,
            description=(
                "Device to be used for inference. If None, the device of the input image will be used. "
                "Defaults to None."
            ),
        ),
    ] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        This method should be implemented in the subclass.

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _validate_image_attributions(self) -> None:
        """Validates the image attributions and performs necessary operations to ensure compatibility with the device.

        Raises:
            ValueError: If the shapes of the attributes and image tensors do not match.
        """
        validate_shapes(self.attributes, self.image)

        self.attributes = self.attributes.to(self.device)
        if self.device != self.image.device:
            self.image.to(self.device)

    @model_validator(mode="after")
    def validate_image_attributions(self) -> Self:
        """Validates the image attributions.

        This method performs validation on the image attributions to ensure they are correct.

        Returns:
            Self: The current instance of the class.
        """
        self._validate_image_attributions()
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the image and attributes tensors to the specified device.

        Args:
            device (str or torch.device): The device to move the tensors to.

        Returns:
            Self: The modified object with tensors moved to the specified device.
        """
        self.image = self.image.to(device)
        self.attributes = self.attributes.to(device)
        self.device = self.image.device
        return self


class ImageSpatialAttributes(ImageAttributes):
    """Represents spatial attributes of an image used for explanation.

    Attributes:
        segmentation_mask (torch.Tensor): Spatial (Segmentation) mask used for the explanation.
    """

    segmentation_mask: Annotated[
        torch.Tensor,
        BeforeValidator(validate_segmentation_mask),
        Field(
            description="Spatial (Segmentation) mask used for the explanation.",
        ),
    ]

    @property
    def flattened_segmentation_mask(self) -> torch.Tensor:
        return self.segmentation_mask.select(dim=self.image.band_axis, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        return self.flattened_segmentation_mask

    @model_validator(mode="after")
    def validate_image_attributions(self) -> Self:
        """Validates the image attributions.

        This method is responsible for validating the image attributions
        and performing any necessary operations on the segmentation mask.

        Returns:
            Self: The current instance of the class.
        """
        super()._validate_image_attributions()
        self.segmentation_mask = self.segmentation_mask.to(self.device)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the Lime object and its segmentation mask to the specified device.

        Args:
            device (str or torch.device): The device to move the object and mask to.

        Returns:
            Self: The Lime object itself.
        """
        super().to(device)
        self.segmentation_mask = self.segmentation_mask.to(device)
        return self


class ImageSpectralAttributes(ImageAttributes):
    """Represents an image with spectral attributes used for explanation.

    Attributes:
        band_mask (torch.Tensor): Band mask used for the explanation.
        band_names (dict[str, int]): Dictionary that translates the band names into the band segment ids.
    """

    band_mask: Annotated[
        torch.Tensor,
        BeforeValidator(validate_band_mask),
        Field(
            description="Band mask used for the explanation.",
        ),
    ]
    band_names: Annotated[
        dict[str, int],
        Field(
            description="Dictionary that translates the band names into the band segment ids.",
        ),
    ]

    @property
    def flattened_band_mask(self) -> torch.Tensor:
        """Returns a flattened band mask tensor.

        The method selects the appropriate dimensions from the `band_mask` tensor
        based on the `axis_to_select` and returns a flattened version of the selected
        tensor.

        Returns:
            torch.Tensor: The flattened band mask tensor.
        """
        axis_to_select = HSI_AXIS_ORDER.copy()
        axis_to_select.remove(self.image.band_axis)
        return self.band_mask.select(dim=axis_to_select[0], index=0).select(dim=axis_to_select[1], index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns the flattened band mask as a torch.Tensor.

        Returns:
            torch.Tensor: The flattened band mask.
        """
        return self.flattened_band_mask

    @model_validator(mode="after")
    def validate_image_attributions(self) -> Self:
        """Validates the image attributions.

        This method performs validation on the image attributions by calling the
        base class's `_validate_image_attributions` method. It also converts the
        `band_mask` attribute to the device specified by `self.device`.

        Returns:
            Self: The current instance of the class.
        """
        super()._validate_image_attributions()
        self.band_mask = self.band_mask.to(self.device)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the Lime object to the specified device.

        Args:
            device (str or torch.device): The device to move the object to.

        Returns:
            Self: The Lime object itself.
        """
        super().to(device)
        self.band_mask = self.band_mask.to(device)
        self.band_names = validate_band_names_with_mask(self.band_names, self.band_mask)
        return self


###################################################################
############################ EXPLAINER ############################
###################################################################


class Explainer(ABC):
    """Explainer class for explaining models.

    Args:
        explainable_model (ExplainableModel): The explainable model to be explained.
        interpretable_model (InterpretableModel): The interpretable model used to approximate the black-box model
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
        similarity_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The similarity function used by Lime.
            Defaults to None.
        perturb_func (Callable[[torch.Tensor], torch.Tensor] | None, optional): The perturbation function used by Lime.
            Defaults to None.
    """

    def __init__(
        self,
        explainable_model: ExplainableModel,
        interpretable_model: InterpretableModel,
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
        image: Image,
        segmentation_method: Literal["patch", "slic"] = "slic",
        **segmentation_method_params: Any,
    ) -> torch.Tensor:
        """Generates a segmentation mask for the given image using the specified segmentation method.

        Args:
            image (Image): The input image for which the segmentation mask needs to be generated.
            segmentation_method (Literal["patch", "slic"], optional): The segmentation method to be used.
                Defaults to "slic".
            **segmentation_method_params (Any): Additional parameters specific to the chosen segmentation method.

        Returns:
            torch.Tensor: The segmentation mask as a tensor.

        Raises:
            ValueError: If the input image is not an instance of the Image class.
            ValueError: If an unsupported segmentation method is specified.
        """
        if not isinstance(image, Image):
            raise ValueError("Image should be an instance of Image class")

        if segmentation_method == "slic":
            return Lime._get_slick_segmentation_mask(image, **segmentation_method_params)
        elif segmentation_method == "patch":
            return Lime._get_patch_segmentation_mask(image, **segmentation_method_params)
        else:
            raise ValueError(f"Unsupported segmentation method: {segmentation_method}")

    @staticmethod
    def get_band_mask(
        image: Image,
        band_names: None | list[str | list[str]] | dict[tuple[str, ...] | str, int] = None,
        band_groups: dict[str | tuple[str, ...], list[int]] | None = None,
        band_ranges_indices: None | dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int]] = None,
        band_ranges_wavelengths: None
        | dict[str | tuple[str, ...], list[tuple[float, float]] | tuple[float, float]] = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
    ) -> tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """Generates a band mask based on the provided image and band information.

        Args:
            image (Image): The input image.
            band_names (None | list[str | list[str]] | dict[tuple[str, ...] | str, int], optional):
                The names of the spectral bands to include in the mask. Defaults to None.
            band_groups (dict[str | tuple[str, ...], list[int]] | None, optional):
                The groups of bands to include in the mask. Defaults to None.
            band_ranges_indices (None | dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int]], optional):
                The ranges of band indices to include in the mask. Defaults to None.
            band_ranges_wavelengths (None | dict[str | tuple[str, ...], list[tuple[float, float]] | tuple[float, float]], optional):
                The ranges of band wavelengths to include in the mask. Defaults to None.
            device (str | torch.device | None, optional):
                The device to use for computation. Defaults to None.
            repeat_dimensions (bool, optional):
                Whether to repeat the dimensions of the mask to match the input image shape. Defaults to False.

        Returns:
            tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]: A tuple containing the band mask tensor and a dictionary
            mapping band names to segment IDs.

        Raises:
            ValueError: If the input image is not an instance of the Image class.
            ValueError: If no band names, groups, or ranges are provided.
        """
        if not isinstance(image, Image):
            raise ValueError("Image should be an instance of Image class")

        # validate types
        dict_labels_to_segment_ids = None
        if band_names is not None:
            logger.debug("Getting band mask from band names of spectral bands")
            validate_band_names(band_names)
            band_ranges_wavelengths, dict_labels_to_segment_ids = Lime._get_band_ranges_wavelengths_from_band_names(  # type: ignore
                band_names
            )
        if band_ranges_wavelengths is not None:
            logger.debug("Getting band mask from band groups given by ranges of wavelengths")
            band_ranges_indices = Lime._get_band_range_indices_from_band_ranges_wavelengths(  # type: ignore
                image,
                band_ranges_wavelengths,
            )
        if band_ranges_indices is not None:
            logger.debug("Getting band mask from band groups given by ranges of indices")
            validate_band_ranges(band_ranges_indices, variable_name="band_ranges_indices")
            band_groups = Lime._get_band_groups_from_band_ranges_indices(
                image.wavelengths,
                band_ranges_indices,
            )
        if band_groups is None:
            raise ValueError("No band names, groups, or ranges provided")
        validate_band_ranges(band_groups, variable_name="band_groups")

        return Lime._create_tensor_band_mask(
            image,
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
        if isinstance(segment_name, (tuple, str)):
            return segment_name
        elif isinstance(segment_name, list):
            return tuple(segment_name)
        raise ValueError(f"Incorrect segment {segment_name} type. Should be either a list or string")

    @staticmethod
    @lru_cache(maxsize=32)
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
            segment_name = tuple(segment_name)
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

        return tuple(band_names_segment) if len(band_names_segment) > 1 else band_names_segment[0]

    @staticmethod
    def _get_band_ranges_wavelengths_from_band_names(
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int],
    ) -> tuple[dict[tuple[str, ...] | str, list[tuple[float, float]]], dict[tuple[str, ...] | str, int]]:
        """Extracts ranges of wavelengths from the band names.

        This function takes a list or dictionary of band names or segments and extracts the ranges of wavelengths
        associated with each segment. It returns a tuple containing a dictionary with mapping segment labels into
        wavelength ranges and a dictionary mapping segment labels into segment ids.

        Args:
            band_names (list[str | list[str]] | dict[tuple[str, ...] | str, int]):
                A list or dictionary with band names or segments.

        Returns:
            tuple[dict[tuple[str, ...] | str, list[tuple[float, float]]], dict[tuple[str, ...] | str, int]]:
                A tuple containing the dictionary with mapping segment labels into wavelength ranges and the mapping
                from segment labels into segment ids.
        """

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
        band_ranges_wavelengths: dict[tuple[str, ...] | str, list[tuple[float, float]]] = {}
        for segment in segments_list_after_mapping:
            band_ranges_wavelengths[segment] = [
                (spyndex.bands[band_name].min_wavelength, spyndex.bands[band_name].max_wavelength)
                for band_name in segment
            ]

        return band_ranges_wavelengths, dict_labels_to_segment_ids

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
        if len(ranges) == 2 and all(isinstance(x, (int, float)) for x in ranges):
            ranges = [tuple(ranges)]  # type: ignore

        for start, end in ranges:  # type: ignore
            start_idx = torch.searchsorted(wavelengths, start, side="right")
            end_idx = torch.searchsorted(wavelengths, end, side="left")
            indices.append((start_idx.item(), end_idx.item()))
        return indices

    @staticmethod
    def _get_band_range_indices_from_band_ranges_wavelengths(
        image: Image,
        band_ranges_wavelengths: dict[str | tuple[str, ...], list[tuple[float, float]] | tuple[float, float]],
    ) -> dict[str | tuple[str, ...], list[tuple[int, int]]]:
        """Converts the ranges of wavelengths into ranges of indices.

        Args:
            image (Image): The image object containing the wavelengths.
            band_ranges_wavelengths (dict): A dictionary mapping segment labels to wavelength ranges.

        Returns:
            dict: A dictionary mapping segment labels to index ranges.

        Raises:
            ValueError: If band_ranges_wavelengths is not a dictionary.
        """
        if not isinstance(band_ranges_wavelengths, dict):
            raise ValueError("band_ranges_wavelengths should be a dictionary")

        band_range_indices: dict[str | tuple[str, ...], list[tuple[int, int]]] = {}
        for segment_label, segment_range in band_ranges_wavelengths.items():
            indices = Lime._convert_wavelengths_to_indices(image.wavelengths, segment_range)
            valid_indices_format = validate_segment_format_range(indices)
            valid_range_indices = validate_segment_range(image.wavelengths, valid_indices_format)
            band_range_indices[segment_label] = valid_range_indices

        return band_range_indices

    @staticmethod
    def _get_band_groups_from_band_ranges_indices(
        wavelengths: torch.Tensor,
        band_ranges_indices: dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int]],
    ) -> dict[str | tuple[str, ...], list[int]]:
        """Converts the ranges of indices into actual indices of bands.

        Args:
            wavelengths (torch.Tensor): The tensor containing the wavelengths.
            band_ranges_indices (dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int]]):
                A dictionary mapping segment labels to a list of index ranges or a single index range.

        Returns:
            dict[str | tuple[str, ...], list[int]]: A dictionary mapping segment labels to a list of band indices.
        """
        band_groups: dict[tuple[str, ...] | str, list[int]] = {}
        for segment_label, segment_range in band_ranges_indices.items():
            validated_ranges_list = validate_segment_format_range(segment_range)
            validated_ranges_list = validate_segment_range(wavelengths, validated_ranges_list)

            band_groups[segment_label] = list(
                chain.from_iterable(
                    [
                        list(range(int(validated_range[0]), int(validated_range[1])))
                        for validated_range in validated_ranges_list
                    ]
                )
            )

        return band_groups

    @staticmethod
    def _check_overlapping_segments(
        image: Image, dict_labels_to_indices: dict[str | tuple[str, ...], list[int]]
    ) -> None:
        """Check for overlapping segments in the given image.

        Args:
            image (Image): The image object containing the wavelengths.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
                A dictionary mapping segment labels to indices.

        Returns:
            None
        """
        overlapping_segments: dict[int, str | tuple[str, ...]] = {}
        for segment_label, indices in dict_labels_to_indices.items():
            for idx in indices:
                if image.wavelengths[idx] in overlapping_segments:
                    logger.warning(
                        f"Bands {overlapping_segments[image.wavelengths[idx]]} and {segment_label} are overlapping"
                    )
                overlapping_segments[image.wavelengths[idx]] = segment_label

    @staticmethod
    def _validate_and_create_dict_labels_to_segment_ids(
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None,
        segment_labels: list[str | tuple[str, ...]],
    ) -> tuple[dict[str | tuple[str, ...], int], bool]:
        """Validates and creates a dictionary mapping segment labels to segment IDs.

        Args:
            dict_labels_to_segment_ids (dict[str | tuple[str, ...], int] | None):
                The existing mapping from segment labels to segment IDs, or None if it doesn't exist.
            segment_labels (list[str | tuple[str, ...]]): The list of segment labels.

        Returns:
            tuple[dict[str | tuple[str, ...], int], bool]: A tuple containing the validated dictionary mapping segment
            labels to segment IDs and a boolean flag indicating whether the segment labels are hashed.

        Raises:
            ValueError: If the length of `dict_labels_to_segment_ids` doesn't match the length of `segment_labels`.
            ValueError: If a segment label is not present in `dict_labels_to_segment_ids`.
            ValueError: If there are non-unique segment IDs in `dict_labels_to_segment_ids`.
        """
        hashed = False

        if dict_labels_to_segment_ids is None:
            logger.debug("Creating mapping from segment labels into ids")
            return {segment: idx + 1 for idx, segment in enumerate(segment_labels)}, hashed

        logger.debug("Using existing mapping from segment labels into segment ids")

        if len(dict_labels_to_segment_ids) != len(segment_labels):
            raise ValueError(
                (
                    f"Incorrect dict_labels_to_segment_ids - length mismatch. Expected: "
                    f"{len(segment_labels)}, Actual: {len(dict_labels_to_segment_ids)}"
                )
            )

        segment_labels_hashed = [Lime._extract_bands_from_spyndex(segment) for segment in segment_labels]

        for segment_label in segment_labels:
            if segment_label not in dict_labels_to_segment_ids and segment_label not in segment_labels_hashed:
                raise ValueError(f"{segment_label} not present in the dict_labels_to_segment_ids")
            if segment_label in segment_labels_hashed:
                logger.debug(
                    "Detected that the segment labels are converted using hashing function. Hashed flag set to True"
                )
                hashed = True

        unique_segment_ids = set(dict_labels_to_segment_ids.values())
        if len(unique_segment_ids) != len(segment_labels):
            raise ValueError("Non unique segment ids in the dict_labels_to_segment_ids")

        logger.debug("Passed mapping is correct")
        return dict_labels_to_segment_ids, hashed

    @staticmethod
    def _create_single_dim_band_mask(
        image: Image,
        dict_labels_to_indices: dict[str | tuple[str, ...], list[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int],
        device: torch.device,
        hashed: bool,
    ) -> torch.Tensor:
        """Create a one-dimensional band mask based on the given image, labels, and segment IDs.

        Args:
            image (Image): The input image.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
                A dictionary mapping labels or label tuples to lists of indices.
            dict_labels_to_segment_ids (dict[str | tuple[str, ...], int]):
                A dictionary mapping labels or label tuples to segment IDs.
            device (torch.device): The device to use for the tensor.
            hashed (bool): A flag indicating whether the labels should be hashed.

        Returns:
            torch.Tensor: The one-dimensional band mask tensor.

        Raises:
            ValueError: If the indices for a segment are out of bounds for the one-dimensional band mask.
        """
        band_mask_single_dim = torch.zeros(len(image.wavelengths), dtype=torch.int64, device=device)

        segment_labels = list(dict_labels_to_segment_ids.keys())
        if hashed:
            segment_labels = [Lime._extract_bands_from_spyndex(segment) for segment in segment_labels]

        for segment_label in segment_labels[::-1]:
            segment_indices = dict_labels_to_indices[segment_label]
            segment_id = dict_labels_to_segment_ids[segment_label]

            are_indices_valid = all(0 <= idx < dim for idx, dim in zip(segment_indices, band_mask_single_dim.shape))
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
    def _expand_band_mask(image: Image, band_mask_single_dim: torch.Tensor, repeat_dimensions: bool) -> torch.Tensor:
        """Expands the band mask to match the dimensions of the input image.

        Args:
            image (Image): The input image.
            band_mask_single_dim (torch.Tensor): The band mask tensor with a single dimension.
            repeat_dimensions (bool): Whether to repeat the dimensions of the band mask to match the image.

        Returns:
            torch.Tensor: The expanded band mask tensor.
        """
        band_axis = HSI_AXIS_ORDER.copy()
        band_axis.remove(image.band_axis)
        band_mask = band_mask_single_dim.unsqueeze(band_axis[0]).unsqueeze(band_axis[1])

        if repeat_dimensions:
            size_image = image.image.size()
            size_mask = band_mask.size()

            repeat_dims = [s2 // s1 for s1, s2 in zip(size_mask, size_image)]
            band_mask = band_mask.repeat(repeat_dims)

        return band_mask

    @staticmethod
    def _create_tensor_band_mask(
        image: Image,
        dict_labels_to_indices: dict[str | tuple[str, ...], list[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
        return_dict_labels_to_segment_ids: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """Create a tensor band mask from dictionaries. The band mask is created based on the given image, labels, and
        segment IDs. The band mask is a tensor with the same shape as the input image and contains segment IDs, where
        each segment is represented by a unique ID. The band mask will be used to attribute the image using the LIME
        method.

        Args:
            image (Image): The input image.
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
            device = image.image.device

        segment_labels = list(dict_labels_to_indices.keys())
        hashed = False

        logger.debug(f"Creating a band mask on the device {device} using {len(segment_labels)} segments")

        # Check for overlapping segments
        Lime._check_overlapping_segments(image, dict_labels_to_indices)

        # Create or validate dict_labels_to_segment_ids
        dict_labels_to_segment_ids, hashed = Lime._validate_and_create_dict_labels_to_segment_ids(
            dict_labels_to_segment_ids, segment_labels
        )

        # Create single-dimensional band mask
        band_mask_single_dim = Lime._create_single_dim_band_mask(
            image, dict_labels_to_indices, dict_labels_to_segment_ids, device, hashed
        )

        # Expand band mask to match image dimensions
        band_mask = Lime._expand_band_mask(image, band_mask_single_dim, repeat_dimensions)

        if return_dict_labels_to_segment_ids:
            return band_mask, dict_labels_to_segment_ids
        return band_mask

    def get_spatial_attributes(
        self,
        image: Image,
        segmentation_mask: np.ndarray | torch.Tensor | None = None,
        target: int | None = None,
        segmentation_method: Literal["slic", "patch"] = "slic",
        **segmentation_method_params: Any,
    ) -> ImageSpatialAttributes:
        """
        Get spatial attributes of an image using the LIME method. Based on the provided image and segmentation mask
        LIME method attributes the `superpixels` provided by the segmentation mask. Please refer to the original paper
        `https://arxiv.org/abs/1602.04938` for more details or to Christoph Molnar's book
        `https://christophm.github.io/interpretable-ml-book/lime.html`.

        This function attributes the image using the LIME (Local Interpretable Model-Agnostic Explanations)
        method for spatial data. It returns an `ImageSpatialAttributes` object that contains the image,
        the attributions, the segmentation mask, and the score of the interpretable model used for the explanation.

        Args:
            image (Image): An `Image` object for which the attribution is performed.
            segmentation_mask (np.ndarray | torch.Tensor | None, optional):
                A segmentation mask according to which the attribution should be performed.
                If None, a new segmentation mask is created using the `segmentation_method`.
                    Additional parameters for the segmentation method may be passed as kwargs. Defaults to None.
            target (int, optional): If the model creates more than one output, it analyzes the given target.
                Defaults to None.
            segmentation_method (Literal["slic", "patch"], optional):
                Segmentation method used only if `segmentation_mask` is None. Defaults to "slic".
            **segmentation_method_params (Any): Additional parameters for the segmentation method.

        Returns:
            ImageSpatialAttributes: An `ImageSpatialAttributes` object that contains the image, the attributions,
                the segmentation mask, and the score of the interpretable model used for the explanation.

        Raises:
            ValueError: If the Lime object is not initialized or is not an instance of LimeBase.
            ValueError: If the problem type of the explainable model is not "regression".
            AssertionError: If the image is not an instance of the Image class.
        """
        if self._lime is None or not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")

        if self.explainable_model.problem_type != "regression":
            raise ValueError("For now only the regression problem is supported")

        assert isinstance(image, Image), "Image should be an instance of Image class"

        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(image, segmentation_method, **segmentation_method_params)
        segmentation_mask = validate_torch_tensor_type(
            segmentation_mask, "Segmentation mask should be None, numpy array, or torch tensor"
        )

        image = image.to(self.device)
        segmentation_mask = segmentation_mask.to(self.device)

        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=segmentation_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=True,
            return_input_shape=True,
        )

        spatial_attribution = ImageSpatialAttributes(
            image=image,
            attributes=lime_attributes[0],
            segmentation_mask=segmentation_mask,
            score=score,
        )

        return spatial_attribution

    def get_spectral_attributes(
        self,
        image: Image,
        band_mask: np.ndarray | torch.Tensor | None = None,
        target=None,
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int] | None = None,
        verbose=False,
    ) -> ImageSpectralAttributes:
        """
        Attributes the image using LIME method for spectral data. Based on the provided image and band mask, the LIME
        method attributes the image based on `superbands` (clustered bands) provided by the band mask.
        Please refer to the original paper `https://arxiv.org/abs/1602.04938` for more details or to
        Christoph Molnar's book `https://christophm.github.io/interpretable-ml-book/lime.html`.

        The function returns an ImageSpectralAttributes object that contains the image, the attributions, the band mask,
        the band names, and the score of the interpretable model used for the explanation.

        Args:
            image (Image): An Image for which the attribution is performed.
            band_mask (np.ndarray | torch.Tensor | None, optional): Band mask that is used for the spectral attribution.
                If equals to None, the band mask is created within the function. Defaults to None.
            target (int, optional): If the model creates more than one output, it analyzes the given target.
                Defaults to None.
            band_names (list[str] | dict[str | tuple[str, ...], int] | None, optional): Band names. Defaults to None.
            verbose (bool, optional): Specifies whether to show progress during the attribution process. Defaults to False.

        Returns:
            ImageSpectralAttributes: An ImageSpectralAttributes object containing the image, the attributions,
                the band mask, the band names, and the score of the interpretable model used for the explanation.
        """

        if self._lime is None or not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")

        if self.explainable_model.problem_type != "regression":
            raise ValueError("For now only the regression problem is supported")

        assert isinstance(image, Image), "Image should be an instance of Image class"

        if band_mask is None:
            band_mask, band_names = self.get_band_mask(image, band_names)
        band_mask = validate_torch_tensor_type(band_mask, "Band mask should be None, numpy array, or torch tensor")

        if band_names is None:
            unique_segments = torch.unique(band_mask)
            band_names = {segment: idx for idx, segment in enumerate(unique_segments)}
        else:
            # checking consistency of names
            # unique_segments = torch.unique(band_mask)
            # if isinstance(band_names, dict):
            #     assert set(unique_segments).issubset(set(band_names.values())), "Incorrect band names"
            logger.debug("Band names are provided, using them. In future it there should be an option to validate them")

        image = image.to(self.device)
        band_mask = band_mask.to(self.device)

        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=band_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=verbose,
            return_input_shape=True,
        )

        spectral_attribution = ImageSpectralAttributes(
            image=image,
            attributes=lime_attributes[0],
            band_mask=band_mask,
            band_names=band_names,
            score=score,
        )

        return spectral_attribution

    @staticmethod
    def _get_slick_segmentation_mask(
        image: Image, num_interpret_features: int = 10, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Creates a segmentation mask using the SLIC method.

        Args:
            image (Image): An Image for which the segmentation mask is created.
            num_interpret_features (int, optional): Number of segments. Defaults to 10.
            *args: Additional positional arguments to be passed to the SLIC method.
            **kwargs: Additional keyword arguments to be passed to the SLIC method.

        Returns:
            torch.Tensor: An output segmentation mask.
        """
        segmentation_mask = slic(
            image.image.cpu().detach().numpy(),
            n_segments=num_interpret_features,
            mask=np.array(image.get_squeezed_binary_mask.to("cpu")),
            channel_axis=image.band_axis,
            *args,
            **kwargs,
        )

        if segmentation_mask.min() == 1:
            segmentation_mask -= 1

        segmentation_mask = torch.from_numpy(segmentation_mask)
        segmentation_mask = segmentation_mask.unsqueeze(dim=image.band_axis)

        return segmentation_mask

    @staticmethod
    def _get_patch_segmentation_mask(image: Image, patch_size=10, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Creates a segmentation mask using the patch method - creates small squares of the same size
            and assigns a unique value to each square.

        Args:
            image (Image): An Image for which the segmentation mask is created.
            patch_size (int, optional): Size of the patch, the image size should be divisible by this value.
                Defaults to 10.

        Returns:
            torch.Tensor: An output segmentation mask.
        """
        logger.warning("Patch segmentation only works for band_index = 0 now")

        if image.image.shape[1] % patch_size != 0 or image.image.shape[2] % patch_size != 0:
            raise ValueError("Invalid patch_size. patch_size must be a factor of both width and height of the image")

        height, width = image.image.shape[1], image.image.shape[2]

        mask_zero = image.image.bool()[0]
        idx_mask = torch.arange(height // patch_size * width // patch_size, device=image.image.device).reshape(
            height // patch_size, width // patch_size
        )
        idx_mask += 1
        segmentation_mask = torch.repeat_interleave(idx_mask, patch_size, dim=0)
        segmentation_mask = torch.repeat_interleave(segmentation_mask, patch_size, dim=1)
        segmentation_mask = segmentation_mask * mask_zero
        # segmentation_mask = torch.repeat_interleave(
        # torch.unsqueeze(segmentation_mask, dim=image.band_axis),
        # repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        segmentation_mask = segmentation_mask.unsqueeze(dim=image.band_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx

        return segmentation_mask
