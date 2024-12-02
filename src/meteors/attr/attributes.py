from __future__ import annotations

from typing_extensions import Annotated, Self
from loguru import logger

import torch
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationInfo, AfterValidator, BeforeValidator

from meteors import HSI
from meteors.exceptions import ShapeMismatchError, MaskCreationError, HSIAttributesError


# Constants
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
        ShapeMismatchError: If the shape of the attributes tensor does not match the shape of the hsi.
    """
    if list(attributes.shape) != list(hsi.image.shape):
        raise ShapeMismatchError(
            f"Attributes and HSI have different, unmatching shapes: {attributes.shape}, {hsi.image.shape}"
        )


def align_band_names_with_mask(
    band_names: dict[str | tuple[str, ...], int], band_mask: torch.Tensor
) -> dict[str | tuple[str, ...], int]:
    """Aligns the band names dictionary with the unique values in the band mask.

    This function ensures that the band_names dictionary correctly represents all unique
    values in the band_mask. It adds a 'not_included' category if necessary and validates
    that all mask values are accounted for in the band names.

    Args:
        band_names (dict[str | tuple[str, ...], int]): A dictionary mapping band names to their corresponding
                                     integer values in the mask.
        band_mask (torch.Tensor): A tensor representing the band mask, where each unique
                                  integer corresponds to a different band or category.

    Returns:
        dict[str | tuple[str, ...], int]: The updated band_names dictionary, potentially including a
                        'not_included' category if 0 is present in the mask but not in
                        the original band_names.

    Raises:
        MaskCreationError: If the band_names dictionary contains a 'not_included' category and some
                    unique values in the mask are not present in the band_names.
        MaskCreationError: If the band_names dictionary does not contain all unique values in the mask.
    Warns:
        UserWarning: If a 'not_included' category (0) is added to the band_names.

    Notes:
        - All unique values in the mask must be present in the band_names dictionary
          after the alignment process.
    """
    unique_mask_values = set(band_mask.unique().int().tolist())
    band_name_values = set(band_names.values())

    # Check if 0 is in the mask but not in band_names
    for value in unique_mask_values:
        if value not in band_name_values:
            if "not_included" in band_names:
                raise MaskCreationError(
                    "Band names should not contain 'not_included' if some unique ids are present in the mask and not in band names"
                )
            else:
                logger.info(
                    f"Adding 'not_included' to band names because {value} ids is present in the mask and not in band names"
                )
                band_names["not_included"] = value
                band_name_values.add(value)
                # exit the loop as we have already added the 'not_included' category
                break

    # Validate that all mask values are in band_names
    if not set(unique_mask_values).issubset(set(band_name_values)):
        raise MaskCreationError("Band names should have all unique values in mask")
    if len(unique_mask_values) != len(band_name_values):
        logger.warning(
            "There exists bands defined in the band_names field that are not present in the mask. Removing them."
        )

        band_names = {k: v for k, v in band_names.items() if v in unique_mask_values}

    return band_names


def validate_attribution_method(value: str | None) -> str | None:
    """Validates the attribution method.

    Args:
        value (str | None): The attribution method to be validated.

    Returns:
        str | None: The validated attribution method.
    """
    if value is None:
        return value
    value = value.title()
    if value not in AVAILABLE_ATTRIBUTION_METHODS:
        logger.warning(
            f"Unknown attribution method: {value}. The core implemented methods are {AVAILABLE_ATTRIBUTION_METHODS}"
        )
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
        HSIAttributesError: If the HSI image is not present in the info data.
        ValueError: If no device is specified or if the provided device string is invalid.
        TypeError: If the provided device is neither None, a string, nor a torch.device.
    """
    if device is None:
        if "hsi" not in info.data:
            raise HSIAttributesError("The HSI image is not present in the attributes data, INTERNAL ERROR")

        hsi: torch.Tensor = info.data["hsi"]
        device = hsi.device
    elif isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValueError(f"Device {device} is not valid") from e  # Invalid inference device ERROR
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
        score (float): The score provided by the interpretable model. Can be None if method don't provide one.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        attribution_method (str | None): The method used to generate the explanation. Defaults to None.
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
        str | None,
        AfterValidator(validate_attribution_method),
        Field(
            description="The method used to generate the explanation.",
        ),
    ] = None
    score: Annotated[
        float | None,
        Field(
            validate_default=True,
            description="The score provided by the interpretable model. Can be None if method don't provide one.",
        ),
    ] = None
    mask: Annotated[
        torch.Tensor | None,
        BeforeValidator(validate_and_convert_mask),
        Field(
            description="`superpixel` or `superband` mask used for the explanation.",
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
        raise NotImplementedError("The `flattened_attributes` property must be implemented in the subclass")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Returns the orientation of the hsi.

        Returns:
            tuple[str, str, str]: The orientation of the hsi corresponding to the attributes.
        """
        return self.hsi.orientation

    def _validate_hsi_attributions_and_mask(self) -> None:
        """Validates the hsi attributions and performs necessary operations to ensure compatibility with the device.

        Raises:
            ValueError: If the shapes of the attributes and hsi tensors do not match.
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
            OrientationError: If the target orientation is not a valid tuple of three one-letter strings.
        """
        current_orientation = self.orientation
        hsi = self.hsi.change_orientation(target_orientation, inplace=inplace)
        if inplace:
            attrs = self
        else:
            attrs = self.model_copy()
            attrs.hsi = hsi

        # now change the orientation of the attributes
        if current_orientation == target_orientation:
            return attrs

        permute_dims = [current_orientation.index(dim) for dim in target_orientation]

        attrs.attributes = attrs.attributes.permute(permute_dims)

        if attrs.mask is not None:
            attrs.mask = attrs.mask.permute(permute_dims)
        return attrs


class HSISpatialAttributes(HSIAttributes):
    """Represents spatial attributes of an hsi used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral image object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): The score provided by the interpretable model. Can be None if method don't provide one.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        attribution_method (str | None): The method used to generate the explanation. Defaults to None.
        segmentation_mask (torch.Tensor): Spatial (Segmentation) mask used for the explanation.
        flattened_attributes (torch.Tensor): Spatial 2D attribution map.
    """

    @property
    def segmentation_mask(self) -> torch.Tensor:
        """Returns the 2D spatial segmentation mask that has the same size as the hsi image.

        Returns:
            torch.Tensor: The segmentation mask tensor.

        Raises:
            HSIAttributesError: If the segmentation mask is not provided in the attributes object.
        """
        if self.mask is None:
            raise HSIAttributesError("Segmentation mask is not provided in the attributes object")
        return self.mask.select(dim=self.hsi.spectral_axis, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes, with removed repeated dimensions.

        In the case of spatial attributes, the flattened attributes are 2D spatial attributes of shape (rows, columns) and the spectral dimension is removed.

        Examples:
            >>> segmentation_mask = torch.zeros((3, 2, 2))
            >>> attrs = HSISpatialAttributes(hsi, attributes, score=0.5, segmentation_mask=segmentation_mask)
            >>> attrs.flattened_attributes
                tensor([[0., 0.],
                        [0., 0.]])

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        return self.attributes.select(dim=self.hsi.spectral_axis, index=0)

    def _validate_hsi_attributions_and_mask(self) -> None:
        """Validates the hsi attributions and performs necessary operations to ensure compatibility with the device.

        Raises:
            HSIAttributesError: If the segmentation mask is not provided in the attributes object.
        """
        super()._validate_hsi_attributions_and_mask()
        if self.mask is None:
            raise HSIAttributesError("Segmentation mask is not provided in the attributes object")


class HSISpectralAttributes(HSIAttributes):
    """Represents an hsi with spectral attributes used for explanation.

    Attributes:
        hsi (HSI): Hyperspectral hsi object for which the explanations were created.
        attributes (torch.Tensor): Attributions (explanations) for the hsi.
        score (float): R^2 score of interpretable model used for the explanation.
        device (torch.device): Device to be used for inference. If None, the device of the input hsi will be used.
            Defaults to None.
        attribution_method (str | None): The method used to generate the explanation. Defaults to None.
        band_mask (torch.Tensor): Band mask used for the explanation.
        band_names (dict[str | tuple[str, ...], int]): Dictionary that translates the band names into the band segment ids.
        flattened_attributes (torch.Tensor): Spectral 1D attribution map.
    """

    band_names: Annotated[
        dict[str | tuple[str, ...], int],
        Field(
            description="Dictionary that translates the band names into the band segment ids.",
        ),
    ]

    @property
    def band_mask(self) -> torch.Tensor:
        """Returns a 1D band mask - a band mask with removed repeated dimensions (num_bands, ),
        where num_bands is the number of bands in the hsi image.

        The method selects the appropriate dimensions from the `band_mask` tensor
        based on the `axis_to_select` and returns a flattened version of the selected
        tensor.

        Returns:
            torch.Tensor: The flattened band mask tensor.

        Examples:
            >>> band_names = {"R": 0, "G": 1, "B": 2}
            >>> attrs = HSISpectralAttributes(hsi, attributes, score=0.5, mask=band_mask)
            >>> attrs.flattened_band_mask
            torch.tensor([0, 1, 2])
        """
        if self.mask is None:
            raise ValueError("Band mask is not provided")
        axis_to_select = [i for i in range(self.hsi.image.ndim) if i != self.hsi.spectral_axis]
        return self.mask.select(dim=axis_to_select[0], index=0).select(dim=axis_to_select[1] - 1, index=0)

    @property
    def flattened_attributes(self) -> torch.Tensor:
        """Returns a flattened tensor of attributes with removed repeated dimensions.

        In the case of spectral attributes, the flattened attributes are 1D tensor of shape (num_bands, ), where num_bands is the number of bands in the hsi image.

        Returns:
            torch.Tensor: A flattened tensor of attributes.
        """
        axis = [i for i in range(self.attributes.ndim) if i != self.hsi.spectral_axis]
        return self.attributes.select(dim=axis[0], index=0).select(dim=axis[1] - 1, index=0)

    def _validate_hsi_attributions_and_mask(self) -> None:
        """Validates the hsi attributions and performs necessary operations to ensure compatibility with the device.

        Raises:
            HSIAttributesError: If the band mask is not provided in the attributes object
        """
        super()._validate_hsi_attributions_and_mask()
        if self.mask is None:
            raise HSIAttributesError("Band mask is not provided in the attributes object")

        self.band_names = align_band_names_with_mask(self.band_names, self.mask)
