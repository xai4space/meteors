from __future__ import annotations
from typing_extensions import Annotated, Self

import torch
import numpy as np
from functools import lru_cache
from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, model_validator
from pydantic.functional_validators import BeforeValidator

import spyndex


def get_band_axis(orientation: tuple[str, str, str]) -> int:
    """Returns the index of the band axis in the orientation list.

    Returns:
        int: The index of the band axis.
    """
    return orientation.index("C")


def validate_orientation(value: tuple[str, str, str]) -> tuple[str, str, str]:
    """Validates the orientation value.

    Args:
        value (tuple[str, str, str]):
            The orientation value to be validated. It should be a tuple of three one-letter strings.

    Returns:
        tuple[str, str, str]: The validated orientation value.

    Raises:
        ValueError: If the value is not a tuple of three one-letter strings
            or if it does not contain 'W', 'H', and 'C' in any order.
    """
    if not isinstance(value, tuple):
        value = tuple(value)

    if len(value) != 3 or any(elem not in ["H", "W", "C"] for elem in value):
        raise ValueError("Orientation must be a tuple of 'H', 'W', and 'C' in any order.")
    return value


def validate_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validates the input image and converts it to a torch tensor if necessary.

    Args:
        image (np.ndarray | torch.Tensor): The input image to be validated.

    Returns:
        torch.Tensor: The validated image as a torch tensor.

    Raises:
        TypeError: If the image is not a numpy array or torch tensor.
    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("Image should be a numpy array or torch tensor")

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    return image


def validate_device(device: str | torch.device | None, info: ValidationInfo) -> torch.device:
    """Validates and returns the device to be used for inference.

    Args:
        device (str | torch.device | None): The device to be used for inference.
            If None, the device of the input image will be used.
        info (ValidationInfo): The validation information.

    Returns:
        torch.device: The validated device.

    Raises:
        TypeError: If the device is not a string or torch device.
    """
    if device is None:
        image: torch.Tensor = info.data["image"]
        device = image.device
    elif isinstance(device, str):
        device = torch.device(device)
    if not isinstance(device, torch.device):
        raise TypeError("Device should be a string or torch device")
    return device


def validate_wavelengths(
    wavelengths: torch.Tensor | np.ndarray | list[int | float] | tuple[int | float],
) -> torch.Tensor:
    """Validates the input wavelengths and converts them to a numpy array if necessary.

    Args:
        wavelengths (torch.Tensor | np.ndarray | list[int | float] | tuple[int | float]):
            The input wavelengths to be validated.

    Returns:
        torch.Tensor: The validated and converted wavelengths as a torch tensor.

    Raises:
        ValueError: If the wavelengths cannot be converted to a torch tensor.
    """
    if not isinstance(wavelengths, (np.ndarray, torch.Tensor, list, tuple)):
        raise ValueError("Wavelengths should be a numpy array, torch tensor or a sequence of integers or floats")

    if not isinstance(wavelengths, torch.Tensor):
        wavelengths = torch.as_tensor(wavelengths)
    return wavelengths


def validate_shapes(wavelengths: np.ndarray, image: torch.Tensor, band_axis: int) -> None:
    """Validates the shape of the wavelengths array against the image tensor.

    Args:
        wavelengths (np.ndarray): Array of wavelengths.
        image (torch.Tensor): Image tensor.
        band_axis (int): Index of the band axis in the image tensor.

    Raises:
        ValueError: If the length of wavelengths does not correspond to the number of channels in the image tensor.
    """
    if wavelengths.shape[0] != image.shape[band_axis]:
        raise ValueError("Length of wavelengths must match the number of channels in the image.")


def validate_binary_mask(mask: np.ndarray | torch.Tensor | None | str, info: ValidationInfo) -> torch.Tensor:
    """Validates and processes a binary mask.

    Args:
        mask (np.ndarray | torch.Tensor | None | str): The binary mask to validate and process.
            It can be a numpy ndarray, a PyTorch tensor, a string 'artificial', or None.
        info (ValidationInfo): Additional information needed for processing the mask.

    Returns:
        torch.Tensor: The processed binary mask.

    Raises:
        ValueError: If the binary mask is not a tensor, numpy ndarray, or the string 'artificial'.
        ValueError: If the binary mask has a different shape than the image.
    """
    if mask is not None and not isinstance(mask, (torch.Tensor, np.ndarray, str)):
        raise ValueError(
            "Binary mask should be a tensor, numpy ndarray or a string 'artificial' which will create an automatic mask"
        )

    image: torch.Tensor = info.data["image"]
    band_axis: int = get_band_axis(info.data["orientation"])
    device: torch.device = info.data["device"]

    if mask is None:
        binary_mask = torch.ones_like(image, dtype=torch.bool, device=device)
    elif isinstance(mask, np.ndarray):
        binary_mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    elif isinstance(mask, str):
        if mask == "artificial":
            binary_mask = torch.index_select(image, 0, torch.tensor([0], device=device)).bool()[
                0
            ]  # get the first channel and encode it to bools
            binary_mask = torch.repeat_interleave(
                binary_mask.unsqueeze(dim=band_axis),
                repeats=image.shape[band_axis],
                dim=band_axis,
            )
        else:
            raise ValueError(
                "Mask should be a tensor, numpy ndarray or a string 'artificial' which will create an automatic mask"
            )
    else:
        binary_mask = mask.bool().to(device)

    if binary_mask.shape != image.shape:
        raise ValueError("Binary mask should have the same shape as the image")

    return binary_mask


class Image(BaseModel):
    image: Annotated[  # Should always be a first field
        torch.Tensor,
        BeforeValidator(validate_image),
        Field(description="Hyperspectral image. Converted to torch tensor."),
    ]
    wavelengths: Annotated[
        torch.Tensor,
        BeforeValidator(validate_wavelengths),
        Field(description="Wavelengths present in the image. Defaults to None."),
    ]
    orientation: Annotated[
        tuple[str, str, str],
        BeforeValidator(validate_orientation),
        Field(
            description=(
                'Orientation of the image - sequence of three one-letter strings in any order: "C", "H", "W" '
                'meaning respectively channels, height and width of the image. Defaults to ("C", "H", "W").'
            ),
        ),
    ] = ("C", "H", "W")
    device: Annotated[
        torch.device,
        BeforeValidator(validate_device),
        Field(
            validate_default=True,
            exclude=True,
            description="Device to be used for inference. If None, the device of the input image will be used. Defaults to None.",
        ),
    ] = None
    binary_mask: Annotated[
        torch.Tensor,
        BeforeValidator(validate_binary_mask),
        Field(
            validate_default=True,
            description=(
                "Binary mask used to cover not important parts of the base image, masked parts have values equals to 0. "
                "Converted to torch tensor. Defaults to None."
            ),
        ),
    ] = None

    @property
    def band_axis(self) -> int:
        """Returns the index of the band axis in the orientation list.

        Returns:
            int: The index of the band axis.
        """
        return get_band_axis(self.orientation)

    @property
    def get_squeezed_binary_mask(self) -> torch.Tensor:
        """Returns the squeezed binary mask tensor. We assume that the binary mask is a 3D tensor with the same mask for
        all bands.

        Returns:
            torch.Tensor | None: The squeezed binary mask tensor.
        """
        transposed_binary_mask = (
            self.binary_mask if self.band_axis == 2 else torch.moveaxis(self.binary_mask, self.band_axis, 2)
        )  # move channel axis to the back
        return transposed_binary_mask[:, :, 0]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_image_data(self) -> Self:
        """Validates the image data by checking the shape of the wavelengths, image, and band_axis.

        Returns:
            Self: The instance of the class.
        """
        validate_shapes(self.wavelengths, self.image, self.band_axis)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Moves the image and binary mask (if available) to the specified device.

        Args:
            device (str or torch.device): The device to move the image and binary mask to.

        Returns:
            Self: The updated Image object.
        """
        self.image = self.image.to(device)
        self.binary_mask = self.binary_mask.to(device)
        self.device = self.image.device
        return self

    @lru_cache
    def get_rgb_image(
        self, mask: bool = True, cutoff_min: bool = False, output_rgb_band_axis: int | None = None
    ) -> torch.Tensor:
        """Returns the RGB image of the meteor data.

        Args:
            mask (bool, optional): Whether to apply a mask to the image. Defaults to True.
            cutoff_min (bool, optional): Whether to apply a minimum cutoff to the image. Defaults to False.
            output_rgb_band_axis (int | None, optional): The axis where the RGB bands should be placed.
                If None, uses the default band axis. Defaults to None.

        Returns:
            torch.Tensor: The RGB image tensor.

        Examples:
            >>> # Create an Image object
            >>> hsi_image = Image(image=torch.rand(10, 10, 10), wavelengths=np.arange(10))
            >>> # Get the RGB image with default settings
            >>> rgb_image = hsi_image.get_rgb_image()
            >>> rgb_image.shape
            torch.Size([10, 10, 3])
            >>> # Get the RGB image with a specific output band axis
            >>> rgb_image = get_rgb_image(output_rgb_band_axis=0)
            >>> # Get the RGB image without applying a mask
            >>> rgb_image = get_rgb_image(mask=False)
        """
        if output_rgb_band_axis is None:
            output_rgb_band_axis = self.band_axis

        rgb_img = torch.stack(
            [
                self.select_single_band_from_name(band, normalize=True, mask=mask, cutoff_min=cutoff_min)
                for band in ["R", "G", "B"]
            ],
            dim=self.band_axis,
        )

        return (
            rgb_img
            if output_rgb_band_axis == self.band_axis
            else torch.moveaxis(rgb_img, self.band_axis, output_rgb_band_axis)
        )

    def _get_central_band(
        self, selected_wavelengths: torch.Tensor, mask: bool = True, cutoff_min: bool = False, normalize: bool = True
    ) -> torch.Tensor:
        """Retrieves the central band from the image based on the selected wavelengths.

        Args:
            selected_wavelengths (torch.Tensor): The selected wavelengths.
            mask (bool, optional): Flag indicating whether to apply a binary mask. Defaults to True.
            cutoff_min (bool, optional): Flag indicating whether to cutoff the minimum value. Defaults to False.
            normalize (bool, optional): Flag indicating whether to normalize the wave values. Defaults to True.

        Returns:
            torch.Tensor: The central band of the image.
        """
        start_index = np.where(self.wavelengths == selected_wavelengths[0])[0][0]

        if mask:
            transposed_binary_mask = (
                self.binary_mask if self.band_axis == 2 else torch.moveaxis(self.binary_mask, self.band_axis, 2)
            )  # move channel axis to the back

        transposed_image = (
            self.image if self.band_axis == 2 else torch.moveaxis(self.image, self.band_axis, 2)
        )  # move channel axis to the back

        center_band = len(selected_wavelengths) // 2
        band_index = start_index + center_band

        wave = transposed_image[:, :, band_index]
        if normalize:
            if cutoff_min:
                wave_min = wave[wave != 0].min()
            else:
                wave_min = wave.min()

            wave = (wave - wave_min) / (wave.max() - wave_min)

            if cutoff_min:
                wave[wave == wave.min] = 0  # the values that equaled to 0 previously

        if mask:
            wave = wave * (transposed_binary_mask[:, :, band_index])

        return wave

    def select_single_band_from_name(
        self, band_name: str, method="center", mask=True, cutoff_min=False, normalize=True
    ) -> torch.Tensor:
        """Selects a single band from the image based on the given band name.

        Args:
            band_name (str): The name of the band to select.
            method (str, optional): The method to use for band selection. Defaults to "center".
            mask (bool, optional): Whether to apply a mask to the selected band. Defaults to True.
            cutoff_min (bool, optional): Whether to cutoff the minimum value of the selected band.
                Defaults to False.
            normalize (bool, optional): Whether to normalize the selected band. Defaults to True.

        Returns:
            numpy.ndarray: The selected band as a numpy array.

        Raises:
            AssertionError: If the specified band name is not found.

        Raises:
            NotImplementedError: If the specified method is not supported.
        """
        band = spyndex.bands.get(band_name)
        assert band is not None, f"Band {band_name} not found"

        min_wave = band.min_wavelength
        max_wave = band.max_wavelength

        selected_wavelengths = self.wavelengths[(self.wavelengths >= min_wave) & (self.wavelengths <= max_wave)]
        if method == "center":
            return self._get_central_band(selected_wavelengths, mask, cutoff_min, normalize)
        else:
            raise NotImplementedError("Only center method is supported for now")
