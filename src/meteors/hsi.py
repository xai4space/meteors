from __future__ import annotations
from typing_extensions import Annotated, Self
import warnings

from meteors.exceptions import ShapeMismatchError, BandSelectionError

import torch
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, model_validator, PlainValidator

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import spyndex

#####################################################################
############################ VALIDATIONS ############################
#####################################################################


def get_channel_axis(orientation: tuple[str, str, str]) -> int:
    """Returns the index of the channel axis in the orientation list.

    Returns:
        int: The index of the band axis.
    """
    return orientation.index("C")


def validate_orientation(value: tuple[str, str, str] | list[str] | str) -> tuple[str, str, str]:
    """Validates the orientation tuple.

    Args:
        value (tuple[str, str, str] | list[str] | str):
            The orientation value to be validated. It should be a tuple of three one-letter strings.

    Returns:
        tuple[str, str, str]: The validated orientation value.

    Raises:
        ValueError: If the value is not a tuple of three one-letter strings
            or if it does not contain 'W', 'H', and 'C' in any order.
    """
    if not isinstance(value, tuple):
        value = tuple(value)  # type: ignore

    if len(value) != 3 or set(value) != {"H", "W", "C"}:
        raise ValueError(value)
    return value  # type: ignore


def ensure_image_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Ensures the input image is a PyTorch tensor, converting it if necessary.

    This function validates that the input is either a numpy array or a PyTorch tensor,
    and converts numpy arrays to PyTorch tensors. If the input is already a PyTorch tensor,
    it is returned unchanged.

    Args:
        image (np.ndarray | torch.Tensor): The input image to be converted/validated.

    Returns:
        torch.Tensor: The image as a PyTorch tensor.

    Raises:
        TypeError: If the image is neither a numpy array nor a PyTorch tensor.
    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("Image must be either a numpy array or a PyTorch tensor")

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    return image


def resolve_inference_device_hsi(device: str | torch.device | None, info: ValidationInfo) -> torch.device:
    """Resolves and returns the device to be used for inference.

    This function determines the appropriate PyTorch device for inference based on the input
    parameters and available information. It handles three scenarios:
    1. If a specific device is provided, it validates and returns it.
    2. If no device is specified (None), it uses the device of the input tensor image.
    3. If a string is provided, it attempts to convert it to a torch.device.

    Args:
        device (str | torch.device | None): The desired device for inference.
            If None, the device of the input hsi will be used.
        info (ValidationInfo): An object containing additional validation information,
            including the input hsi data.

    Returns:
        torch.device: The resolved PyTorch device for inference.

    Raises:
        RuntimeError: raised if the hsi is not present in the info data,
        TypeError: If the provided device is neither None, a string, nor a torch.device.
        ValueError: If the provided device string is invalid.
    """
    if device is None:
        if "image" not in info.data:
            raise RuntimeError(
                "Hyperspectral image tensor is not present in the HSI object, an internal error occurred"
            )
        image: torch.Tensor = info.data["image"]
        device = image.device
    elif isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as e:
            raise ValueError(f"Device {device} is not valid") from e
    if not isinstance(device, torch.device):
        raise TypeError("Device should be a string or torch device")

    logger.debug(f"Device for inference: {device.type}")
    return device


def ensure_wavelengths_tensor(
    wavelengths: torch.Tensor | np.ndarray | list[int | float] | tuple[int | float],
) -> torch.Tensor:
    """Converts the input wavelengths to a PyTorch tensor.

    This function takes wavelength data in various formats and ensures it is converted
    to a PyTorch tensor. It accepts PyTorch tensors, NumPy arrays, lists, or tuples of
    numeric values.

    Args:
        wavelengths (torch.Tensor | np.ndarray | list[int | float] | tuple[int | float]):
            The input wavelengths in any of the supported formats.

    Returns:
        torch.Tensor: The wavelengths as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor, NumPy array, list, or tuple.
        ValueError: If the wavelengths cannot be converted to a PyTorch tensor.
    """
    if not isinstance(wavelengths, (torch.Tensor, np.ndarray, list, tuple)):
        raise TypeError("Wavelengths must be a PyTorch tensor, NumPy array, list, or tuple of numeric values")

    try:
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.as_tensor(wavelengths)
    except Exception as e:
        raise TypeError(f"Failed to convert wavelengths to a PyTorch tensor: {str(e)}") from e

    return wavelengths


def validate_shapes(wavelengths: torch.Tensor, image: torch.Tensor, spectral_axis: int) -> None:
    """Validates that the length of the wavelengths matches the number of channels in the image tensor.

    Args:
        wavelengths (torch.Tensor): Array of wavelengths.
        image (torch.Tensor): Image tensor.
        spectral_axis (int): Index of the band axis in the image tensor.

    Raises:
        ShapeMismatchError: If the length of wavelengths does not correspond to the number of channels in the image tensor.
    """
    if wavelengths.shape[0] != image.shape[spectral_axis]:
        raise ShapeMismatchError(
            f"Length of wavelengths must match the number of channels in the image. Passed {wavelengths.shape[0]} wavelengths for {image.shape[spectral_axis]} channels",
        )


def process_and_validate_binary_mask(
    mask: np.ndarray | torch.Tensor | None | str, info: ValidationInfo
) -> torch.Tensor:
    """Processes and validates a binary mask, ensuring it's in the correct format and shape.

    This function handles various input types for binary masks, including None (creates a mask of ones),
    'artificial' (creates a mask based on the first channel of the image), numpy arrays, and PyTorch tensors.
    It ensures the output is a PyTorch tensor of the correct shape and type.

    Args:
        mask (np.ndarray | torch.Tensor | None | str): The input binary mask or mask specification.
            - If None: Creates a mask of ones with the same shape as the image.
            - If 'artificial': Creates a mask based on the first channel of the image.
            - If numpy array or PyTorch tensor: Converts to a boolean PyTorch tensor.
        info (ValidationInfo): A dataclass containing additional required information:
            - 'image': The reference image (PyTorch tensor) for shape and device.
            - 'orientation': The orientation of the image data.
            - 'device': The PyTorch device to use.

    Returns:
        torch.Tensor: A boolean PyTorch tensor representing the validated and processed binary mask.

    Raises:
        RuntimeError: If the input mask is invalid or if required information is missing from info.
        ShapeMismatchError: If the resulting binary mask doesn't match the shape of the reference image.
        TypeError: If the input binary mask is not in a correct format - a numpy array, PyTorch tensor, or string.
    """
    if mask is not None and not isinstance(mask, (torch.Tensor, np.ndarray, str)):
        raise TypeError("Binary mask must be None, a PyTorch tensor, a numpy array, or the string 'artificial'")

    if "image" not in info.data or "orientation" not in info.data or "device" not in info.data:
        raise RuntimeError(
            "Missing required information in ValidationInfo. Required fields: 'image', 'orientation', 'device', ValidationInfo data: "
            + str(info.data)
        )

    image: torch.Tensor = info.data["image"]
    spectral_axis: int = get_channel_axis(info.data["orientation"])
    device: torch.device = info.data["device"]

    if mask is None:
        binary_mask = torch.ones_like(image, dtype=torch.bool, device=device)
    elif isinstance(mask, np.ndarray):
        binary_mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    elif isinstance(mask, str):
        if mask == "artificial":
            binary_mask = torch.index_select(image, 0, torch.tensor([0], device=device)).bool()[0]
            binary_mask = torch.repeat_interleave(
                binary_mask.unsqueeze(dim=spectral_axis),
                repeats=image.shape[spectral_axis],
                dim=spectral_axis,
            )
        else:
            raise ValueError("Unsupported binary_mask field for HSI. Mask specification must be 'artificial'")
    else:
        binary_mask = mask.bool().to(device)

    if binary_mask.shape != image.shape:
        try:
            binary_mask = binary_mask.expand_as(image)
        except RuntimeError:
            raise ShapeMismatchError(
                f"Mismatch in shapes of binary mask and HSI. Passed shapes are respectively: {binary_mask.shape}, {image.shape}"
            )

    return binary_mask


######################################################################
########################## IMAGE DATACLASS ###########################
######################################################################


class HSI(BaseModel):
    """A dataclass for hyperspectral image data, including the image, wavelengths, and binary mask.

    Attributes:
        image (torch.Tensor): The hyperspectral image data as a PyTorch tensor.
        wavelengths (torch.Tensor): The wavelengths present in the image.
        orientation (tuple[str, str, str]): The orientation of the image data.
        device (torch.device): The device to be used for inference.
        binary_mask (torch.Tensor): A binary mask used to cover unimportant parts of the image.
    """

    image: Annotated[  # Should always be a first field
        torch.Tensor,
        PlainValidator(ensure_image_tensor),
        Field(description="Hyperspectral image. Converted to torch tensor."),
    ]
    wavelengths: Annotated[
        torch.Tensor,
        PlainValidator(ensure_wavelengths_tensor),
        Field(description="Wavelengths present in the image. Defaults to None."),
    ]
    orientation: Annotated[
        tuple[str, str, str],
        PlainValidator(validate_orientation),
        Field(
            description=(
                'Orientation of the image - sequence of three one-letter strings in any order: "C", "H", "W" '
                'meaning respectively channels, height and width of the image. Defaults to ("C", "H", "W").'
            ),
        ),
    ] = ("C", "H", "W")
    device: Annotated[
        torch.device,
        PlainValidator(resolve_inference_device_hsi),
        Field(
            validate_default=True,
            exclude=True,
            description="Device to be used for inference. If None, the device of the input image will be used. Defaults to None.",
        ),
    ] = None
    binary_mask: Annotated[
        torch.Tensor,
        PlainValidator(process_and_validate_binary_mask),
        Field(
            validate_default=True,
            description=(
                "Binary mask used to cover not important parts of the base image, masked parts have values equals to 0. "
                "Converted to torch tensor. Defaults to None."
            ),
        ),
    ] = None

    @property
    def spectral_axis(self) -> int:
        """Returns the index of the spectral (wavelength) axis based on the current data orientation.

        In hyperspectral imaging, the spectral axis represents the dimension along which
        different spectral bands or wavelengths are arranged. This property dynamically
        determines the index of this axis based on the current orientation of the data.

        Returns:
            int: The index of the spectral axis in the current data structure.
                - 0 for 'CHW' or 'CWH' orientations (Channel/Wavelength first)
                - 2 for 'HWC' or 'WHC' orientations (Channel/Wavelength last)
                - 1 for 'HCW' or 'WCH' orientations (Channel/Wavelength in the middle)

        Note:
            The orientation is typically represented as a string where:
            - 'C' represents the spectral/wavelength dimension
            - 'H' represents the height (rows) of the image
            - 'W' represents the width (columns) of the image

        Examples:
            >>> hsi_image = HSI()
            >>> hsi_image.orientation = "CHW"
            >>> hsi_image.spectral_axis
            0
            >>> hsi_image.orientation = "HWC"
            >>> hsi_image.spectral_axis
            2
        """
        return get_channel_axis(self.orientation)

    @property
    def spatial_binary_mask(self) -> torch.Tensor:
        """Returns a 2D spatial representation of the binary mask.

        This property extracts a single 2D slice from the 3D binary mask, assuming that
        the mask is identical across all spectral bands. It handles different data
        orientations by first ensuring the spectral dimension is the last dimension
        before extracting the 2D spatial mask.

        Returns:
            torch.Tensor: A 2D tensor representing the spatial binary mask.
                The shape will be (H, W) where H is height and W is width of the image.

        Note:
            - This assumes that the binary mask is consistent across all spectral bands.
            - The returned mask is always 2D, regardless of the original data orientation.

        Examples:
            >>> # If self.binary_mask has shape (100, 100, 5) with spectral_axis=2:
            >>> hsi_image = HSI(binary_mask=torch.rand(100, 100, 5), orientation=("H", "W", "C"))
            >>> hsi_image.spatial_binary_mask.shape
            torch.Size([100, 100])
            >>> If self.binary_mask has shape (5, 100, 100) with spectral_axis=0:
            >>> hsi_image = HSI(binary_mask=torch.rand(5, 100, 100), orientation=("C", "H", "W"))
            >>> hsi_image.spatial_binary_mask.shape
            torch.Size([100, 100])
        """
        mask = self.binary_mask if self.binary_mask is not None else torch.ones_like(self.image)
        return mask.select(dim=self.spectral_axis, index=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_image_data(self) -> Self:
        """Validates the image data by checking the shape of the wavelengths, image, and spectral_axis.

        Returns:
            Self: The instance of the class.
        """
        validate_shapes(self.wavelengths, self.image, self.spectral_axis)
        return self

    def to(self, device: str | torch.device) -> Self:
        """Moves the image and binary mask (if available) to the specified device.

        Args:
            device (str or torch.device): The device to move the image and binary mask to.

        Returns:
            Self: The updated HSI object.

        Examples:
            >>> # Create an HSI object
            >>> hsi_image = HSI(image=torch.rand(10, 10, 10), wavelengths=np.arange(10))
            >>> # Move the image to cpu
            >>> hsi_image = hsi_image.to("cpu")
            >>> hsi_image.device
            device(type='cpu')
            >>> # Move the image to cuda
            >>> hsi_image = hsi_image.to("cuda")
            >>> hsi_image.device
            device(type='cuda', index=0)
        """
        self.image = self.image.to(device)
        self.binary_mask = self.binary_mask.to(device)
        self.device = self.image.device
        return self

    def get_image(self, apply_mask: bool = True) -> torch.Tensor:
        """Returns the hyperspectral image data with optional masking applied.

        Args:
            apply_mask (bool, optional): Whether to apply the binary mask to the image.
                Defaults to True.
        Returns:
            torch.Tensor: The hyperspectral image data.

        Notes:
            - If apply_mask is True, the binary mask will be applied to the image based on the `binary_mask` attribute.

        Examples:
            >>> hsi_image = HSI(image=torch.rand(10, 100, 100), wavelengths=np.linspace(400, 1000, 10))
            >>> image = hsi_image.get_image()
            >>> image.shape
            torch.Size([10, 100, 100])
            >>> image = hsi_image.get_image(apply_mask=False)
            >>> image.shape
            torch.Size([10, 100, 100])
        """
        if apply_mask and self.binary_mask is not None:
            return self.image * self.binary_mask
        return self.image

    def get_rgb_image(
        self,
        apply_mask: bool = True,
        apply_min_cutoff: bool = False,
        output_channel_axis: int | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extracts an RGB representation from the hyperspectral image data.

        This method creates a 3-channel RGB image by selecting appropriate bands
        corresponding to red, green, and blue wavelengths from the hyperspectral data.

        Args:
            apply_mask (bool, optional): Whether to apply the binary mask to the image.
                Defaults to True.
            apply_min_cutoff (bool, optional): Whether to apply a minimum intensity
                cutoff to the image. Defaults to False.
            output_channel_axis (int | None, optional): The axis where the RGB channels
                should be placed in the output tensor. If None, uses the current spectral
                axis of the hyperspectral data. Defaults to None.
            normalize (bool, optional): Whether to normalize the band values to the [0, 1] range.
                Defaults to True.

        Returns:
            torch.Tensor: The RGB representation of the hyperspectral image.
                Shape will be either (H, W, 3), (3, H, W), or (H, 3, W) depending on
                the specified output_channel_axis, where H is height and W is width.

        Notes:
            - The RGB bands are extracted using predefined wavelength ranges for R, G, and B.
            - Each band is normalized independently before combining into the RGB image.
            - If apply_mask is True, masked areas will be set to zero in the output.
            - If apply_min_cutoff is True, a minimum intensity threshold is applied to each band.

        Examples:
            >>> hsi_image = HSI(image=torch.rand(10, 100, 100), wavelengths=np.linspace(400, 1000, 10))
            >>> rgb_image = hsi_image.get_rgb_image()
            >>> rgb_image.shape
            torch.Size([100, 100, 3])

            >>> rgb_image = hsi_image.get_rgb_image(output_channel_axis=0)
            >>> rgb_image.shape
            torch.Size([3, 100, 100])

            >>> rgb_image = hsi_image.get_rgb_image(apply_mask=False, apply_min_cutoff=True)
            >>> rgb_image.shape
            torch.Size([100, 100, 3])
        """
        if output_channel_axis is None:
            output_channel_axis = self.spectral_axis

        rgb_img = torch.stack(
            [
                self.extract_band_by_name(
                    band, apply_mask=apply_mask, apply_min_cutoff=apply_min_cutoff, normalize=normalize
                )
                for band in ["R", "G", "B"]
            ],
            dim=self.spectral_axis,
        )

        return (
            rgb_img
            if output_channel_axis == self.spectral_axis
            else torch.moveaxis(rgb_img, self.spectral_axis, output_channel_axis)
        )

    def _extract_central_slice_from_band(
        self,
        band_wavelengths: torch.Tensor,
        apply_mask: bool = True,
        apply_min_cutoff: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extracts and processes the central wavelength band from a given range in the hyperspectral image.

        This method selects the central band from a specified range of wavelengths,
        applies optional processing steps (masking, normalization, and minimum cutoff),
        and returns the resulting 2D image slice.

        Args:
            band_wavelengths (torch.Tensor): The selected wavelengths that define the whole band
                from which the central slice will be extracted.
                All of the passed wavelengths must be present in the image.
            apply_mask (bool, optional): Whether to apply the binary mask to the extracted band.
                Defaults to True.
            apply_min_cutoff (bool, optional): Whether to apply a minimum intensity cutoff.
                If True, sets the minimum non-zero value to zero after normalization.
                Defaults to False.
            normalize (bool, optional): Whether to normalize the band values to [0, 1] range.
                Defaults to True.

        Returns:
            torch.Tensor: A 2D tensor representing the processed central wavelength band.
                Shape will be (H, W), where H is height and W is width of the image.

        Notes:
            - The central wavelength is determined as the middle index of the provided wavelengths list.
            - If normalization is applied, it's done before masking and cutoff operations.
            - The binary mask, if applied, is expected to have the same spatial dimensions as the image.

        Examples:
            >>> hsi_image = HSI(image=torch.rand(13, 100, 100), wavelengths=np.linspace(400, 1000, 13))
            >>> band_wavelengths = torch.tensor([500, 600, 650, 700])
            >>> central_slice = hsi_image._extract_central_slice_from_band(band_wavelengths)
            >>> central_slice.shape
            torch.Size([100, 100])

            >>> # Extract a slice without normalization or masking
            >>> raw_band = hsi_image._extract_central_slice_from_band(band_wavelengths, apply_mask=False, normalize=False)
        """
        # check if all wavelengths from the `band_wavelengths` are present in the image
        if not all(wave in self.wavelengths for wave in band_wavelengths):
            raise ValueError("All of the passed wavelengths must be present in the image")

        # sort the `band_wavelengths` to ensure the central band is selected
        band_wavelengths = torch.sort(band_wavelengths).values

        start_index = np.where(self.wavelengths == band_wavelengths[0])[0][0]
        relative_center_band_index = len(band_wavelengths) // 2
        central_band_index = start_index + relative_center_band_index

        # Ensure the spectral dimension is the last
        image = self.image if self.spectral_axis == 2 else torch.moveaxis(self.image, self.spectral_axis, 2)

        slice = image[..., central_band_index]

        if normalize:
            if apply_min_cutoff:
                slice_min = slice[slice != 0].min()
            else:
                slice_min = slice.min()

            slice_max = slice.max()
            if slice_max > slice_min:  # Avoid division by zero
                slice = (slice - slice_min) / (slice_max - slice_min)

            if apply_min_cutoff:
                slice[slice == slice.min()] = 0  # Set minimum values to zero

        if apply_mask:
            mask = (
                self.binary_mask if self.spectral_axis == 2 else torch.moveaxis(self.binary_mask, self.spectral_axis, 2)
            )
            slice = slice * mask[..., central_band_index]

        return slice

    def extract_band_by_name(
        self,
        band_name: str,
        selection_method: str = "center",
        apply_mask: bool = True,
        apply_min_cutoff: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extracts a single spectral band from the hyperspectral image based on a standardized band name.

        This method uses the spyndex library to map standardized band names to wavelength ranges,
        then extracts the corresponding band from the hyperspectral data.

        Args:
            band_name (str): The standardized name of the band to extract (e.g., "Red", "NIR", "SWIR1").
            selection_method (str, optional): The method to use for selecting the band within the wavelength range.
                Currently, only "center" is supported, which selects the central wavelength.
                Defaults to "center".
            apply_mask (bool, optional): Whether to apply the binary mask to the extracted band.
                Defaults to True.
            apply_min_cutoff (bool, optional): Whether to apply a minimum intensity cutoff after normalization.
                If True, sets the minimum non-zero value to zero. Defaults to False.
            normalize (bool, optional): Whether to normalize the band values to the [0, 1] range.
                Defaults to True.

        Returns:
            torch.Tensor: A 2D tensor representing the extracted and processed spectral band.
                Shape will be (H, W), where H is height and W is width of the image.

        Raises:
            BandSelectionError: If the specified band name is not found in the spyndex library.
            NotImplementedError: If a selection method other than "center" is specified.

        Notes:
            - The spyndex library is used to map band names to wavelength ranges.
            - Currently, only the "center" selection method is implemented, which chooses
            the central wavelength within the specified range.
            - Processing steps are applied in the order: normalization, cutoff, masking.

        Examples:
            >>> hsi_image = HSI(image=torch.rand(200, 100, 100), wavelengths=np.linspace(400, 2500, 200))
            >>> red_band = hsi_image.extract_band_by_name("Red")
            >>> red_band.shape
            torch.Size([100, 100])

            >>> # Extract NIR band without normalization or masking
            >>> nir_band = hsi_image.extract_band_by_name("NIR", apply_mask=False, normalize=False)
        """
        band_info = spyndex.bands.get(band_name)
        if band_info is None:
            raise BandSelectionError(f"Band name '{band_name}' not found in the spyndex library")

        min_wave, max_wave = band_info.min_wavelength, band_info.max_wavelength
        selected_wavelengths = self.wavelengths[(self.wavelengths >= min_wave) & (self.wavelengths <= max_wave)]

        if selection_method == "center":
            return self._extract_central_slice_from_band(
                selected_wavelengths, apply_mask=apply_mask, apply_min_cutoff=apply_min_cutoff, normalize=normalize
            )
        else:
            raise NotImplementedError(
                f"Selection method '{selection_method}' is not supported. Only 'center' is currently available."
            )

    def change_orientation(self, target_orientation: tuple[str, str, str] | list[str] | str, inplace=False) -> Self:
        """Changes the orientation of the hsi data to the target orientation.

        Args:
            target_orientation (tuple[str, str, str], list[str], str): The target orientation for the hsi data.
                This should be a tuple of three one-letter strings in any order: "C", "H", "W".
            inplace (bool, optional): Whether to modify the hsi data in place or return a new object.

        Returns:
            Self: The updated HSI object with the new orientation.

        Raises:
            ValueError: If the target orientation is not a valid tuple of three one-letter strings.
        """
        target_orientation = validate_orientation(target_orientation)

        if inplace:
            hsi = self
        else:
            hsi = self.model_copy()

        if target_orientation == self.orientation:
            return hsi

        permute_dims = [hsi.orientation.index(dim) for dim in target_orientation]

        # permute the image
        hsi.image = hsi.image.permute(permute_dims)

        # permute the binary mask
        if hsi.binary_mask is not None:
            hsi.binary_mask = hsi.binary_mask.permute(permute_dims)

        hsi.orientation = target_orientation

        return hsi
