from __future__ import annotations

from typing import Literal
from typing_extensions import Annotated, Self
import warnings
from loguru import logger


import torch
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationInfo
from pydantic.functional_validators import BeforeValidator


from meteors import HSI


# Constants
HSI_AXIS_ORDER = [2, 1, 0]  # (bands, rows, columns)
AVAILABLE_ATTRIBUTION_METHODS = [
    "Lime",
    "Integrated Gradients",
    "Saliency",
    "Input X Gradient",
    "Occlusion",
    "Hyper Noise Tunnel",
    "Noise Tunnel",
]


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


def ensure_hsi_object(value: HSI) -> HSI:
    """Ensures the input is a HSI object.

    This function validates that the input is a HSI object, raising an error if it's not.

    Args:
        value (HSI): The input value to be validated.

    Returns:
        HSI: The input value as a HSI object.

    Raises:
        TypeError: If the input is not a HSI object.
    """
    if not isinstance(value, HSI):
        raise TypeError("HSI object must be provided")
    return value


def validate_and_convert_segmentation_mask(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Ensures the segmentation mask is a PyTorch tensor, converting it if necessary.

    This function validates that the input segmentation mask is either a NumPy array
    or a PyTorch tensor, and converts it to a PyTorch tensor if it's a NumPy array.

    Args:
        value (np.ndarray | torch.Tensor): The segmentation mask to be validated and potentially converted.

    Returns:
        torch.Tensor: The segmentation mask as a PyTorch tensor.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    return ensure_torch_tensor(value, "Segmentation mask")


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


def validate_shapes(attributes: torch.Tensor, hsi: HSI) -> None:
    """Validates that the shape of the attributes tensor matches the shape of the hsi.

    Args:
        attributes (torch.Tensor): The attributes tensor to validate.
        hsi (HSI): The HSI object to compare the shape with.

    Raises:
        ValueError: If the shape of the attributes tensor does not match the shape of the hsi.
    """
    if attributes.shape != hsi.image.shape:
        raise ValueError("Attributes must have the same shape as the hsi")


def validate_and_convert_band_mask(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Ensures the band mask is a PyTorch tensor, converting it if necessary.

    This function validates that the input band mask is either a NumPy array
    or a PyTorch tensor, and converts it to a PyTorch tensor if it's a NumPy array.

    Args:
        value (np.ndarray | torch.Tensor): The band mask to be validated and potentially converted.

    Returns:
        torch.Tensor: The band mask as a PyTorch tensor.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    return ensure_torch_tensor(value, "Band mask")


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
    unique_mask_values = set(band_mask.unique().tolist())
    band_name_values = set(band_names.values())

    # Check if 0 is in the mask but not in band_names
    if 0 in unique_mask_values and 0 not in band_name_values:
        warnings.warn(
            "Band mask contains `0` values which are not covered by the provided band names. "
            "Adding 'not_included' to band names."
        )
        band_names["not_included"] = 0
        band_name_values.add(0)

    # Validate that all mask values are in band_names
    if unique_mask_values != band_name_values:
        raise ValueError("Band names should have all unique values in mask")

    return band_names


def validate_attribution_method(value: str) -> str:
    value = value.title()
    if value not in AVAILABLE_ATTRIBUTION_METHODS:
        raise ValueError(f"Attribution method must be one of {AVAILABLE_ATTRIBUTION_METHODS}, got {value} instead")
    return value


def resolve_inference_device_attributes(device: str | torch.device | None, info: ValidationInfo) -> torch.device:
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
            raise ValueError("The HSI image is not present in the attributes data, INTERNAL ERROR")

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
    """Represents an object that contains Hyperspectral image attributes and explanations.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation. Used only for LIME attributes
        approximation_error (float): Approximation error of the explanation. Used only for IG attributes
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
        attribution_method (str): The method used to generate the explanation.
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
    attribution_method: Annotated[
        str,
        BeforeValidator(validate_attribution_method),
        Field(
            description="The method used to generate the explanation.",
        ),
    ]
    score: Annotated[
        float | None,
        Field(
            validate_default=True,
            le=1.0,
            ge=-1.0,
            description="R^2 score of interpretable model used for the explanation. Used only for LIME attributes",
        ),
    ] = None
    approximation_error: Annotated[
        float | None,
        Field(
            ge=1.0,
            description="Approximation error of the explanation. Also known as convergence delta. Used only for IG attributes",
        ),
    ] = None
    device: Annotated[
        torch.device,
        BeforeValidator(resolve_inference_device_attributes),
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

    def _validate_hsi_attributions(self) -> None:
        """Validates the hsi attributions and performs necessary operations to ensure compatibility with the device.

        Raises:
            ValueError: If the shapes of the attributes and hsi tensors do not match.
        """
        validate_shapes(self.attributes, self.hsi)

        self.attributes = self.attributes.to(self.device)
        if self.device != self.hsi.device:
            self.hsi.to(self.device)

    @model_validator(mode="after")
    def validate_hsi_attributions(self) -> Self:
        """Validates the hsi attributions.

        This method performs validation on the hsi attributions to ensure they are correct.

        Returns:
            Self: The current instance of the class.
        """
        self._validate_hsi_attributions()
        return self

    @model_validator(mode="after")
    def validate_score_and_error(self) -> Self:
        """Validates the score and error attributes.

        This method validates the score and error attributes based on the attribution method.

        Returns:
            Self: The current instance of the class.
        """
        if self.attribution_method.title() != "Lime" and self.score is not None:
            raise ValueError("Score should not be provided for non-LIME attributes")
        if self.attribution_method.title() == "Lime" and self.score is None:
            raise ValueError("Score must be provided for LIME attributes")
        if self.attribution_method.title() != "Integrated Gradients" and self.approximation_error is not None:
            raise ValueError("Approximation error should only be provided for IG attributes")
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
        return self


class HSISpatialAttributes(HSIAttributes):
    """Represents spatial attributes of an hsi used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
        segmentation_mask (torch.Tensor): Spatial (Segmentation) mask used for the explanation.
    """

    segmentation_mask: Annotated[
        torch.Tensor,
        BeforeValidator(validate_and_convert_segmentation_mask),
        Field(
            description="Spatial (Segmentation) mask used for the explanation.",
        ),
    ]
    attribution_method: Annotated[
        str,
        Literal["Lime"],
        Field(
            description="The method used to generate the explanation.",
        ),
    ] = "Lime"

    @property
    def spatial_segmentation_mask(self) -> torch.Tensor:
        """Returns the spatial segmentation mask.

        This method selects the segmentation mask along the specified dimension
        and returns the first index.

        Returns:
            torch.Tensor: The spatial segmentation mask.

        Examples:
            >>> segmentation_mask = torch.zeros((3, 2, 2))
            >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
            >>> attrs.spatial_segmentation_mask
            tensor([[0., 0.],
                    [0., 0.]])
        """
        return self.segmentation_mask.select(dim=self.hsi.spectral_axis, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        In the case of spatial attributes, the flattened attributes are the same as the `spatial_segmentation_mask`

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        >>> segmentation_mask = torch.zeros((3, 2, 2))
        >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
        >>> attrs.flattened_attributes
            tensor([[0., 0.],
                    [0., 0.]])
        """
        return self.spatial_segmentation_mask

    @model_validator(mode="after")
    def validate_hsi_attributions(self) -> Self:
        """Validates the hsi attributions.

        This method is responsible for validating the hsi attributions
        and performing any necessary operations on the segmentation mask.

        Returns:
            Self: The current instance of the class.
        """
        super()._validate_hsi_attributions()
        self.segmentation_mask = self.segmentation_mask.to(self.device)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the Lime object and its segmentation mask to the specified device.

        Args:
            device (str or torch.device): The device to move the object and mask to.

        Returns:
            Self: The Lime object itself.

        Examples:
            >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
            >>> attrs.to("cpu")
            >>> attrs.segmentation_mask.device
            device(type='cpu')
            >>> attrs.hsi.device
            device(type='cpu')
            >>> attrs.to("cuda")
            >>> attrs.segmentation_mask.device
            device(type='cuda')
        """
        super().to(device)
        self.segmentation_mask = self.segmentation_mask.to(device)
        return self


class HSISpectralAttributes(HSIAttributes):
    """Represents an hsi with spectral attributes used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral hsi object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        model_config (ConfigDict): Configuration dictionary for the model.
        band_mask (torch.Tensor): Band mask used for the explanation.
        band_names (dict[str, int]): Dictionary that translates the band names into the band segment ids.
    """

    band_mask: Annotated[
        torch.Tensor,
        BeforeValidator(validate_and_convert_band_mask),
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
    attribution_method: Annotated[
        str,
        Literal["Lime"],
        Field(
            description="The method used to generate the explanation.",
        ),
    ] = "Lime"

    @property
    def spectral_band_mask(self) -> torch.Tensor:
        """Returns a spectral band mask.

        The method selects the appropriate dimensions from the `band_mask` tensor
        based on the `axis_to_select` and returns a flattened version of the selected
        tensor.

        Returns:
            torch.Tensor: The flattened band mask tensor.

        Examples:
            >>> band_names = {"R": 0, "G": 1, "B": 2}
            >>> attrs = HSISpectralAttributes(hsi, attributes, score=0.5, band_mask=band_mask)
            >>> attrs.spectral_band_mask
            torch.tensor([0, 1, 2])
        """
        axis_to_select = HSI_AXIS_ORDER.copy()
        axis_to_select.remove(self.hsi.spectral_axis)
        return self.band_mask.select(dim=axis_to_select[0], index=0).select(dim=axis_to_select[1], index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes.

        In the case of spectral attributes, the flattened attributes are the same as the `spectral_band_mask`

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        return self.spectral_band_mask

    @model_validator(mode="after")
    def validate_hsi_attributions(self) -> Self:
        """Validates the hsi attributions.

        This method performs validation on the hsi attributions by calling the
        base class's `_validate_hsi_attributions` method. It also converts the
        `band_mask` attribute to the device specified by `self.device`.

        Returns:
            Self: The current instance of the class.
        """
        super()._validate_hsi_attributions()
        self.band_mask = self.band_mask.to(self.device)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the Lime object to the specified device.

        Args:
            device (str or torch.device): The device to move the object to.

        Returns:
            Self: The Lime object itself.

        Examples:
            >>> attrs = HSISpectralAttributes(hsi, attributes, score=0.5, band_mask=band_mask)
            >>> attrs.to("cpu")
            >>> attrs.band_mask.device
            device(type='cpu')
            >>> attrs.hsi.device
            device(type='cpu')
            >>> attrs.to("cuda")
            >>> attrs.band_mask.device
            device(type='cuda')
        """
        super().to(device)
        self.band_mask = self.band_mask.to(device)
        self.band_names = align_band_names_with_mask(self.band_names, self.band_mask)
        return self
