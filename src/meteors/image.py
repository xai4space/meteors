from typing_extensions import Annotated, Self
from typing import Sequence

import torch
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, model_validator
from pydantic.functional_validators import BeforeValidator, AfterValidator


import spyndex

# TODO add better binary mask handling
# TODO wavelengths as dicts
# TODO width and height in orientation


def validate_orientation(value: tuple[str, str, str]) -> tuple[str, str, str]:
    if "H" not in value or "W" not in value or "C" not in value:
        raise ValueError("Orientation should contain 'W', 'H' and 'C' in any order")
    return value


def validate_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise ValueError("Image should be a numpy array or torch tensor")

    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)
    return image


def validate_device(
    device: str | torch.device | None, info: ValidationInfo
) -> torch.device:
    if device is None:
        image: torch.Tensor = info.data["image"]
        device = image.device
    elif isinstance(device, str):
        device = torch.device(device)
    if not isinstance(device, torch.device):
        raise ValueError("Device should be a string or torch device")
    return device


def validate_wavelengths(wavelengths: np.ndarray | Sequence[int | float]) -> np.ndarray:
    if not isinstance(wavelengths, np.ndarray):
        wavelengths = np.array(wavelengths)
    return wavelengths


class Image(BaseModel):
    image: Annotated[  # Should always be a first field
        torch.Tensor,
        BeforeValidator(validate_image),
        Field(
            kw_only=False,  # Why
            validate_default=True,  # There is no default value
            description="hyperspectral imaage. Converted to torch tensor.",
        ),
    ]
    wavelengths: Annotated[
        np.ndarray,
        BeforeValidator(validate_wavelengths),
        Field(
            kw_only=False,
            validate_default=True,
            description="wave lengths present in the image. Defaults to None.",
        ),
    ]
    binary_mask: Annotated[
        np.ndarray | torch.Tensor | None | str,
        Field(
            kw_only=False,
            validate_default=True,
            description="binary mask used to cover not important parts of the base image, masked parts have values equals to 0. Converted to torch tensor. Defaults to None.",
        ),
    ] = None
    orientation: Annotated[
        tuple[str, str, str],
        AfterValidator(validate_orientation),
        Field(
            kw_only=False,
            validate_default=True,
            description='orientation of the image - sequence of three one-letter strings in any order: "C", "H", "W" meaning respectively channels, height and width of the image. Defaults to ("C", "H", "W")',
        ),
    ] = ("C", "H", "W")

    _device: Annotated[torch.device, BeforeValidator(validate_device)] = None

    @property
    def band_axis(self) -> int:
        band_axis = self.orientation.index("C")
        return band_axis

    _rgb_img: torch.Tensor | None = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_wavelengths_shape(self) -> Self:
        if self.wavelengths.shape[0] != self.image.shape[self.band_axis]:
            raise ValueError(
                "Improper length of wavelengths - it should correspond to the number of channels"
            )
        return self

    @model_validator(mode="after")
    def validate_binary_mask(self) -> Self:
        if (
            not isinstance(self.binary_mask, torch.Tensor)
            and self.binary_mask is not None
            and not isinstance(self.binary_mask, str)
        ):
            self.binary_mask = torch.tensor(self.binary_mask, device=self._device).int()

        if isinstance(self.binary_mask, str) and self.binary_mask == "artificial":
            self.binary_mask = torch.index_select(
                self.image, 0, torch.tensor([0], device=self._device)
            ).bool()[0]  # get the first channel and encode it to bools
            self.binary_mask = torch.repeat_interleave(
                torch.unsqueeze(self.binary_mask, dim=self.band_axis),
                repeats=self.image.shape[self.band_axis],
                dim=self.band_axis,
            )
        elif isinstance(self.binary_mask, str):
            raise ValueError(
                "Mask should be a tensor, numpy ndarray or a string 'artificial' which will create an automatic mask"
            )

        if self.binary_mask is not None and self.binary_mask.shape != self.image.shape:
            raise ValueError("Binary mask should have the same shape as the image")
        return self

    def to(self, device: str | torch.device) -> Self:
        """Move the image to the device"""
        self._device = device
        self.image = self.image.to(device)
        if self.binary_mask is not None:
            if not isinstance(self.binary_mask, (np.ndarray, str)):
                self.binary_mask = self.binary_mask.to(device)
        return self

    def get_rgb_image(
        self, mask=True, cutoff_min=False, output_band_index: int | None = None
    ) -> torch.tensor:
        """get an RGB image from hyperspectral image. Useful for plotting

        Args:
            mask (bool, optional): whether to apply the mask of the image to the output, only used if the Hyper Image object actually contains the `binary_mask` field. Defaults to True.
            cutoff_min (bool, optional): In case the binary mask is pre-applied to the image stored in the Image memory, i.e. image has zero-valued fields, the image is scaled using the second smallest value in the image. By default the scaling uses the smallest value which in this case is artificial. Defaults to True.
            output_band_index (int, optional): index of the band axis to be used in the output image. Defaults to self.band_index.

        Returns:
            torch.tensor: Image with 3 bands - RGB
        """
        if output_band_index is None:
            output_band_index = self.band_axis

        if self._rgb_img is None:
            self._rgb_img_params = {"mask": mask, "cutoff_min": cutoff_min}
            self._rgb_img = self._get_rgb_img(mask=mask, cutoff_min=cutoff_min)

        # recalculate the image
        if (
            self._rgb_img_params["mask"] != mask
            or self._rgb_img_params["cutoff_min"] != cutoff_min
        ):
            self._rgb_img_params = {"mask": mask, "cutoff_min": cutoff_min}
            self._rgb_img = self._get_rgb_img(mask=mask, cutoff_min=cutoff_min)

        if output_band_index == self.band_axis:
            return self._rgb_img
        return torch.moveaxis(self._rgb_img, self.band_axis, output_band_index)

    @property
    def get_flattened_binary_mask(self) -> torch.tensor:
        # here we assume that mask is the same in each channel
        if self.binary_mask is None:
            return None

        transposed_binary_mask = (
            self.binary_mask
            if self.band_axis == 2
            else torch.moveaxis(self.binary_mask, self.band_axis, 2)
        )  # move channel axis to the back
        return transposed_binary_mask[:, :, 0]

    def select_single_band_from_name(
        self, band_name, method="center", mask=True, cutoff_min=False, normalize=True
    ):
        band = spyndex.bands.get(band_name)
        assert band is not None, f"Band {band_name} not found"

        min_wave = band.min_wavelength
        max_wave = band.max_wavelength

        selected_wavelengths = self.wavelengths[
            (self.wavelengths >= min_wave) & (self.wavelengths <= max_wave)
        ]
        if method == "center":
            start_index = np.where(self.wavelengths == selected_wavelengths[0])[0][0]

            if self.binary_mask is not None and mask:
                transposed_binary_mask = (
                    self.binary_mask
                    if self.band_axis == 2
                    else torch.moveaxis(self.binary_mask, self.band_axis, 2)
                )  # move channel axis to the back
            transposed_image = (
                self.image
                if self.band_axis == 2
                else torch.moveaxis(self.image, self.band_axis, 2)
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
                    wave[wave == wave.min] = (
                        0  # the values that equaled to 0 previously
                    )

            if self.binary_mask is not None and mask:
                wave = wave * (transposed_binary_mask[:, :, band_index])

            return wave
        else:
            raise NotImplementedError("Only center method is supported for now")

    def _get_rgb_img(self, mask=True, cutoff_min=False) -> torch.tensor:
        return torch.stack(
            [
                self.select_single_band_from_name(
                    band, normalize=True, mask=mask, cutoff_min=cutoff_min
                )
                for band in ["R", "G", "B"]
            ],
            axis=self.band_axis,
        )
