from __future__ import annotations

from typing import Literal, Callable, Iterable
from abc import ABC
from functools import cached_property

from meteors.utils.models import ExplainableModel, InterpretableModel
from meteors import Image

from meteors.lime_base import Lime as LimeBase

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    validate_call,
)
from typing_extensions import Annotated, Self

import torch
import numpy as np
import spyndex

from loguru import logger


try:
    from fast_slic import Slic as slic
except ImportError:
    from skimage.segmentation import slic

# important - specify the image orientation
# Width x Height x Channels seems to be the most appropriate
# but the model requires  (C, W, H) or (W, H C)


# explanations


class Explanation(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class ImageAttributes(Explanation):
    image: Annotated[
        Image,
        Field(
            kw_only=False,
            validate_default=True,
            description="Hyperspectral image object on which the attribution is performed.",
        ),
    ]
    attributes: Annotated[
        np.ndarray | torch.Tensor,
        Field(
            kw_only=False,
            validate_default=True,
            description="Attributions saved as a numpy array or torch tensor.",
        ),
    ]
    score: Annotated[
        float,
        Field(
            kw_only=False,
            validate_default=True,
            description="R^2 score of interpretable model used for the explanation.",
        ),
    ]

    _device: torch.device = None
    _flattened_attributes: torch.Tensor = None

    @model_validator(mode="before")
    def validate_attributes(cls, values):
        assert (
            values["attributes"].shape == values["image"].image.shape
        ), "Attributes must have the same shape as the image"

        device = values["image"].image.device
        values["_device"] = device

        if isinstance(values["attributes"], np.ndarray):
            values["attributes"] = torch.tensor(values["attributes"], device=device)
        return values

    @field_validator("score", mode="before")
    def validate_score(cls, value):
        assert 0 <= value <= 1, "R^2 must be between 0 and 1"
        return value

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def to(self, device: torch.device) -> Self:
        """function moves the image and attributes to the specified device"""
        self._device = device
        self.image.to(device)
        self.attributes = self.attributes.to(device)
        return self

    def get_flattened_attributes(self) -> torch.tensor:
        raise NotImplementedError("This method should be implemented in the subclass")


class ImageSpatialAttributes(ImageAttributes):
    segmentation_mask: Annotated[
        np.ndarray | torch.Tensor,
        Field(
            kw_only=False,
            validate_default=True,
            description="Segmentation mask used for the explanation.",
        ),
    ]

    _flattened_segmentation_mask: torch.Tensor = None

    @model_validator(mode="after")
    def validate_segmentation_mask(self) -> Self:
        if isinstance(self.segmentation_mask, np.ndarray):
            self.segmentation_mask = torch.tensor(self.segmentation_mask, device=self._device)

        if self.segmentation_mask.device != self._device:
            self.segmentation_mask = self.segmentation_mask.to(self._device)  # move to the device

        return self

    def to(self, device: torch.device) -> Self:
        super().to(device)
        self.segmentation_mask = self.segmentation_mask.to(device)
        return self

    def get_flattened_segmentation_mask(self) -> torch.tensor:
        """segmentation mask is after all only two dimensional tensor with some repeated values, this function returns only two-dimensional tensor"""
        if self._flattened_segmentation_mask is None:
            self._flattened_segmentation_mask = self.segmentation_mask.select(dim=self.image.band_axis, index=0)
        return self._flattened_segmentation_mask

    def get_flattened_attributes(self) -> torch.tensor:
        """attributions for spatial case are after all only two dimensional tensor with some repeated values, this function returns only two-dimensional tensor"""
        if self._flattened_attributes is None:
            self._flattened_attributes = self.attributes.select(dim=self.image.band_axis, index=0)
        return self._flattened_attributes


class ImageSpectralAttributes(ImageAttributes):
    band_mask: Annotated[
        np.ndarray | torch.Tensor,
        Field(
            kw_only=False,
            validate_default=True,
            description="Band mask used for the explanation.",
        ),
    ]
    band_names: Annotated[
        dict[str, int],
        Field(
            kw_only=False,
            validate_default=True,
            description="Dictionary that translates the band names into the segment values.",
        ),
    ]

    _flattened_band_mask = None

    @model_validator(mode="after")
    def validate_band_mask(self) -> Self:
        if isinstance(self.band_mask, np.ndarray):
            self.band_mask = torch.tensor(self.band_mask, device=self._device)

        if self.band_mask.device != self._device:
            self.band_mask = self.band_mask.to(self._device)

        if 0 not in self.band_names.values() and 0 in torch.unique(self.band_mask):
            self.band_names["not_included"] = 0

        return self

    def to(self, device: torch.device) -> Self:
        super().to(device)
        self.band_mask = self.band_mask.to(device)
        return self

    def get_flattened_band_mask(self) -> torch.tensor:
        """band mask is after all only one dimensional tensor with some repeated values, this function returns only one-dimensional tensor"""
        if self._flattened_band_mask is None:
            dims_to_select = [2, 1, 0]
            dims_to_select.remove(self.image.band_axis)
            self._flattened_band_mask = self.band_mask.select(dim=dims_to_select[0], index=0).select(
                dim=dims_to_select[1], index=0
            )
        return self._flattened_band_mask

    def get_flattened_attributes(self) -> torch.tensor:
        """attributions for spectral case are after all only one dimensional tensor with some repeated values, this function returns only one-dimensional tensor"""
        if self._flattened_attributes is None:
            dims_to_select = [2, 1, 0]
            dims_to_select.remove(self.image.band_axis)
            self._flattened_attributes = self.attributes.select(dim=dims_to_select[0], index=0).select(
                dim=dims_to_select[1], index=0
            )
        return self._flattened_attributes


# explainer itself


class Explainer(BaseModel, ABC):
    explainable_model: ExplainableModel
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @cached_property
    def _device(self) -> torch.device:
        try:
            device = next(self.explainable_model.forward_func.parameters()).device  # type: ignore
        except Exception:
            logger.debug("Not a torch model, setting device to cpu. Error: {e}")
            device = torch.device("cpu")
        return device

    def to(self, device: torch.device) -> Self:
        self.explainable_model.to(device)
        return self


class Lime(Explainer):
    # should it be any different than base lime?
    explainable_model: ExplainableModel
    interpretable_model: InterpretableModel
    similarity_func: Callable | None = None
    perturb_func: Callable | None = None

    _lime = None

    @model_validator(mode="after")
    def construct_lime(self) -> Self:
        self._lime = LimeBase(
            forward_func=self.explainable_model.forward_func,
            interpretable_model=self.interpretable_model,
            similarity_func=self.similarity_func,
            perturb_func=self.perturb_func,
        )

        return self

    def to(self, device: torch.device) -> Self:
        super().to(device)
        # self.interpretable_model.to(device)
        return self

    @staticmethod
    def get_segmentation_mask(
        image: Image,
        segmentation_method: Literal["patch", "slic"] = "slic",
        **segmentation_method_params,
    ) -> torch.Tensor:
        """function generates a segmentation mask - a tensor, for a given image; the segmentation mask can be used in spatial attribution. This tensor contains different integer values for each segments"""
        if segmentation_method == "slic":
            return Lime._get_slick_segmentation_mask(image, **segmentation_method_params)
        if segmentation_method == "patch":
            return Lime._get_patch_segmentation_mask(image, **segmentation_method_params)
        raise NotImplementedError("Only slic and patch methods are supported for now")

    @staticmethod
    def get_band_mask(
        image: Image,
        band_names: None | (list[str | list[str]] | dict[tuple[str, ...] | str, int]) = None,
        band_groups: dict[str | tuple[str, ...], list[int]] | None = None,
        band_ranges_indices: None | (dict[str | tuple[str, ...], list[tuple[int, int]] | tuple[int, int]]) = None,
        band_ranges_wavelengths: None
        | (
            dict[
                str | tuple[str, ...],
                Iterable[tuple[float, float]] | tuple[float, float],
            ]
        ) = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
    ) -> tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """function generates band mask - a tensor that corresponds to the image, which values are different segments. Can be used in spectral attribution. Each slice of the image - a single, one-dimensional tensor has identical values.
        Args:
            image (Image): A Hyperspectral image
            band_names ((List[str | List[str]]) | (Dict[str | List[str], Iterable[int]])): list of band names that should be treated as one segment or a dictionary containing band names as keys and segment values.
            Band names should either be included in `spyndex.indices` or `spyndex.bands`. For instance it could be have values as: `["R", "G", "B"]` or `["IPVI", "AFRI1600"]`. Alternatively, it could be a dictionary with band names as keys and segment ids as values, e.g. `{"R": 1, "G": 2, "B": 3}` or `{"IPVI": 1, "AFRI1600": 2}`.
            band_groups (Dict[str, Iterable[int]]): dictionary containing user-made band groups as keys and list of band indices as values. For instance it could be `{"A": [0, 1, 2, 3, 4], "B": [6, 7, 8, 20, 21, 22]}`. "A" and "B" would be the names of the segments, and [0, 1, 2, 3, 4] and [6, 7, 8, 20, 21, 22] would be the indices of the bands that belong to the segments. The indices should be in the range of the number of bands in the image.
            band_ranges_indices (Dict[str, Iterable[Tuple[int, int]] | Tuple[int, int]]): dictionary containing user-made band ranges as keys and tuple of start and end indices as values. For instance it could equal to `{"A": (0, 4), "B": [(6, 9), (20, 23)]}`. "A" and "B" would be the names of the segments, the created segment indices would equal to [0, 1, 2, 3, 4] and [6, 7, 8, 20, 21, 22]
            band_ranges_wavelengths (Dict[str, list[Tuple[float, float]] | Tuple[float, float]]): dictionary containing user-made band ranges as keys and tuple of start and end wavelengths as values. For instance, if our image contains wavelengths [400, 405, 410, 420, 450], then dictionary could equal to `{"A": (400, 410), "B": [(420, 450)]}`. "A" and "B" would be the names of the segments, the created segment indices would equal to [0, 1, 2] and [3, 4]

            device (torch.device | str | None): a device on which the band mask will be created. If None, it is created on the same device as the image. Defaults to None
            repeat_dimensions bool: Whether to repeat the band mask over all image dimensions For instance, if False the output mask could have shape (1, 1, 150). If True - (64, 64, 150). Defaults to False


        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: a tuple which consists of an actual band mask and a dictionary that translates the band names into the segment values
        """

        if not isinstance(image, Image):
            raise ValueError("Image should be an instance of Image class")

        # validate types

        dict_labels_to_segment_ids = None
        if band_names is not None:
            logger.debug("Getting band mask from band names of spectral bands")
            if not isinstance(band_names, (list, dict)):
                raise TypeError("Band names should be either a list or a dictionary")
            if isinstance(band_names, dict):
                if not all(isinstance(key, (tuple, str)) for key in band_names.keys()):
                    raise TypeError("Keys in band names dictionary should be either a tuple or a string")
                if not all(isinstance(value, int) for value in band_names.values()):
                    raise TypeError("Values in band names dictionary should be integers")
            if isinstance(band_names, list):
                if not all(isinstance(segment, (str, list)) for segment in band_names):
                    raise TypeError("Elements in band names list should be either a string or a list")

            band_ranges_wavelengths, dict_labels_to_segment_ids = Lime._get_band_ranges_wavelengths_from_band_names(  # type: ignore
                band_names
            )
        if band_ranges_wavelengths is not None:
            logger.debug("Getting band mask from band groups given by ranges of wavelengths")
            if not isinstance(band_ranges_wavelengths, dict):
                raise TypeError("Band ranges should be a dictionary")
            if not all(isinstance(key, (tuple, str)) for key in band_ranges_wavelengths.keys()):
                raise TypeError("Keys in band ranges dictionary should be either a tuple or a string")
            if not all(isinstance(value, (list, tuple)) for value in band_ranges_wavelengths.values()):
                raise TypeError("Values in band ranges dictionary should be either a list or a tuple")
            band_ranges_indices = Lime._get_band_range_indices_from_band_ranges_wavelengths(
                image,
                band_ranges_wavelengths,  # type: ignore
            )

        if band_ranges_indices is not None:
            logger.debug("Getting band mask from band groups given by ranges of indices")
            if not isinstance(band_ranges_indices, dict):
                raise TypeError("Band ranges should be a dictionary")
            if not all(isinstance(key, (tuple, str)) for key in band_ranges_indices.keys()):
                raise TypeError("Keys in band ranges dictionary should be either a tuple or a string")
            if not all(isinstance(value, (list, tuple)) for value in band_ranges_indices.values()):
                raise TypeError("Values in band ranges dictionary should be either a list or a tuple")

            band_groups = Lime._get_band_groups_from_band_ranges_indices(
                band_ranges_indices  # type: ignore
            )

        if band_groups is None:
            raise ValueError("No band names, groups, or ranges provided")

        if not isinstance(band_groups, dict):
            raise TypeError("Band groups should be a dictionary")
        if not all(isinstance(key, (tuple, str)) for key in band_groups.keys()):
            raise TypeError("Keys in band groups dictionary should be either a tuple or a string")
        if not all(isinstance(value, list) for value in band_groups.values()):
            raise TypeError("Values in band groups dictionary should be lists")

        return Lime._create_tensor_band_mask(
            image,
            band_groups,
            dict_labels_to_segment_ids=dict_labels_to_segment_ids,
            device=device,
            repeat_dimensions=repeat_dimensions,
            return_dict_labels_to_segment_ids=True,
        )

    @staticmethod
    def _make_band_names_indexable(
        segment_name: Iterable[str] | str,
    ) -> tuple[str, ...] | str:
        """some band names created by user may contain list of strings, that corresponds to different bands. This function converts list of strings into tuple of strings if necessary to make it indexable"""
        if isinstance(segment_name, str):
            return segment_name
        if isinstance(segment_name, list):
            return tuple(segment_name)
        raise ValueError(f"Incorrect segment {segment_name} type. Should be either a list or string")

    @staticmethod
    def _extract_bands_from_spyndex(
        segment_name: Iterable[str] | str,
    ) -> tuple[str, ...]:
        """Users may pass either band names or indices names, as in spyndex library. Index usually consists of multiple segments and needs to be expanded into a list of bands which can be then converted to waveband range."""
        band_names_segment: list[str] = []

        if isinstance(segment_name, str):
            segment_name = tuple([segment_name])
        for band_name in segment_name:
            if band_name in spyndex.indices:
                band_names_segment = band_names_segment + (spyndex.indices[band_name].bands)
            elif band_name in spyndex.bands:
                band_names_segment.append(band_name)
            else:
                raise ValueError(
                    f"Invalid band name {band_name}, band name must be either in `spyndex.indices` or `spyndex.bands`"
                )

        return tuple(band_names_segment)

    @staticmethod
    def _get_band_ranges_wavelengths_from_band_names(
        band_names: list[str | list[str]] | dict[tuple[str, ...] | str, int],
    ) -> tuple[
        dict[str | tuple[str, ...], list[tuple[float, float]]],
        dict[str | tuple[str, ...], int],
    ]:
        """function extracts ranges of wavelengths from the band names

        Args:
            band_names (list[str  |  list[str]] | dict[tuple[str, ...]  |  str, int]): list or dictionary with band names or segments

        Returns:
            dict[str | tuple[str, ...], list[tuple[float, float]]], dict[str | tuple[str, ...], int]: tuple containing the dictionary with mapping segment labels into wavelength ranges and mapping from segment labels into segment ids
        """

        if isinstance(band_names, list):
            logger.debug("band_names is a list of segments, creating a dictionary of segments")

            band_names_hashed = [Lime._make_band_names_indexable(segment) for segment in band_names]

            dict_labels_to_segment_ids = {segment: idx + 1 for idx, segment in enumerate(band_names_hashed)}
            segments_list = band_names_hashed
        elif isinstance(band_names, dict):
            logger.debug("band_names is a dictionary of segments")
            dict_labels_to_segment_ids = band_names
            segments_list = list(band_names.keys())
        else:
            raise ValueError("Incorrect band_names type. It should be a dict or a list")

        segments_list_after_mapping = [Lime._extract_bands_from_spyndex(segment) for segment in segments_list]
        band_ranges_wavelengths: dict[str | tuple[str, ...], list[tuple[float, float]]] = {}
        for segment in segments_list_after_mapping:
            band_ranges_wavelengths[segment] = []
            for band_name in segment:
                min_wavelength = spyndex.bands[band_name].min_wavelength
                max_wavelength = spyndex.bands[band_name].max_wavelength

                band_ranges_wavelengths[segment].append((min_wavelength, max_wavelength))

        return band_ranges_wavelengths, dict_labels_to_segment_ids

    @staticmethod
    def _get_band_range_indices_from_band_ranges_wavelengths(
        image: Image,
        band_ranges_wavelengths: dict[
            str | tuple[str, ...],
            list[tuple[float, float] | list[float]] | tuple[float, float] | list[float],
        ],
    ) -> dict[str | tuple[str, ...], list[tuple[int, int]]]:
        """function converts the ranges of wavelengths into ranges of indices"""
        logger.debug("Verifying and unifying format of band_ranges_indices")

        for segment_label, segment_range in band_ranges_wavelengths.items():
            if not isinstance(segment_range, Iterable):
                raise ValueError(f"Incorrect type for range of segment with label {segment_label}")
            if (
                len(segment_range) == 2
                and isinstance(segment_range[0], (int, float))  # type: ignore
                and isinstance(segment_range[1], (int, float))  # type: ignore
            ):
                # cast into list[Tuple[float, float]] in case type is Tuple[float, float]
                segment_range = [segment_range]  # type: ignore
                band_ranges_wavelengths[segment_label] = segment_range  # type: ignore

            for segment_part in segment_range:
                if (
                    (not isinstance(segment_part, list) and not isinstance(segment_part, tuple))
                    or len(segment_part) != 2
                    or not isinstance(segment_part[0], (int, float))
                    or not isinstance(segment_part[1], (int, float))
                ):
                    raise ValueError(
                        f"Segment {segment_label} has incorrect structure - it should be a tuple or list of length 2 or an Iterable with tuples or lists of length 2"
                    )
                if segment_part[0] >= segment_part[1]:
                    raise ValueError(f"Order of the range {segment_label} is incorrect")

        logger.debug("Mapping the wavelength ranges to get segment indices")

        band_range_indices: dict[str | tuple[str, ...], list[tuple[int, int]]] = {}
        wavelengths = np.array(image.wavelengths)
        for segment_label, segment_range in band_ranges_wavelengths.items():
            band_range_indices[segment_label] = []
            for segment_part_range in segment_range:
                min_index = np.searchsorted(
                    wavelengths,
                    segment_part_range[0],  # type: ignore[index]
                    side="right",
                )
                max_index = np.searchsorted(
                    wavelengths,
                    segment_part_range[1],  # type: ignore[index]
                    side="left",
                )
                band_range_indices[segment_label].append((min_index, max_index))

        return band_range_indices

    @staticmethod
    def _get_band_groups_from_band_ranges_indices(
        band_ranges_indices: dict[
            str | tuple[str, ...],
            list[tuple[int, int] | list[int]] | tuple[int, int] | list[int],
        ],
    ) -> dict[str | tuple[str, ...], list[int]]:
        """function converts the ranges of indices into actual indices of bands"""
        logger.debug("Verifying format of band_ranges_indices")
        for segment_label, segment_range in band_ranges_indices.items():
            if not isinstance(segment_range, Iterable):
                raise ValueError(f"Incorrect type for range of segment with label {segment_label}")
            if (
                len(segment_range) == 2
                and isinstance(segment_range[0], int)  # type: ignore
                and isinstance(segment_range[1], int)  # type: ignore
            ):
                # cast into list[Tuple[int, int]] in case type is Tuple[int, int]
                segment_range = [segment_range]  # type: ignore
                band_ranges_indices[segment_label] = segment_range  # type: ignore

            for segment_part in segment_range:
                if not (isinstance(segment_part, list) or isinstance(segment_part, tuple)) or len(segment_part) != 2:
                    raise ValueError(
                        f"Segment {segment_label} has incorrect structure - it should be a list or tuple of length 2 or an Iterable with list or tuples of length 2"
                    )
                if segment_part[0] > segment_part[1]:
                    raise ValueError(f"Order of the range {segment_label} is incorrect")

        logger.debug("Filling the ranges to get segment indices")

        band_groups: dict[tuple[str, ...] | str, list[int]] = {}
        for segment_label, segment_range in band_ranges_indices.items():
            band_groups[segment_label] = []
            for segment_part_range in segment_range:
                band_groups[segment_label] += list(
                    range(segment_part_range[0], segment_part_range[1])  # type: ignore[index]
                )

            band_groups[segment_label] = list(set(band_groups[segment_label]))

        return band_groups

    @staticmethod
    def _create_tensor_band_mask(
        image: Image,
        dict_labels_to_indices: dict[str | tuple[str, ...], list[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None = None,
        device: str | torch.device | None = None,
        repeat_dimensions: bool = False,
        return_dict_labels_to_segment_ids: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """create tensor band mask from dictionaries

        Args:
            image (Image): an Image for which the band mask is created.
            dict_labels_to_indices (dict[str | tuple[str, ...], list[int]]):
            dict_labels_to_segment_ids (dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None, optional): a dictionary containing the mapping from segment labels into the segment ids - integers that are used to denote the segment in the band_mask tensor. Defaults to None
            device (torch.device | str | None, optional): a device on which the band mask will be created. If None, it is created on the same device as the image. Defaults to None
            repeat_dimensions (bool, optional): Whether to repeat the band mask over all image dimensions For instance, if False the output mask could have shape (1, 1, 150). If True - (64, 64, 150). Defaults to False

            return_dict_labels_to_segment_ids (bool, optional): Whether to return also a dictionary mapping segment labels to segment ids.
        Returns:
            torch.Tensor | Tuple[torch.Tensor, Dict[str, int]]: an actual band mask or a tuple which consists of an actual band mask and a dictionary that translates the band names into the segment values
        """

        if device is None:
            device = image.image.device

        segment_labels = list(dict_labels_to_indices.keys())
        hashed = False

        logger.debug(f"Creating a band mask on the device {device} using {len(segment_labels)} segments")

        logger.debug("Checking if the bands are overlapping")
        overlapping_segments = {wavelength: None for wavelength in image.wavelengths}
        for segment_label, indices in dict_labels_to_indices.items():
            for idx in indices:
                if overlapping_segments[image.wavelengths[idx]] is not None:
                    logger.warning(
                        f"Bands {overlapping_segments[image.wavelengths[idx]]} and {segment_label} are overlapping"
                    )
                overlapping_segments[image.wavelengths[idx]] = segment_label  # type: ignore

        # check if there exists a dictionary dict_labels_to_segment_ids, if not, create it
        if dict_labels_to_segment_ids is None:
            logger.debug("Creating mapping from segment labels into ids")
            dict_labels_to_segment_ids = {segment: idx + 1 for idx, segment in enumerate(segment_labels)}
        else:
            logger.debug("Using existing mapping from segment labels into segment ids")
            logger.debug("Verifying if the passed mapping is correct")
            segment_labels_from_id_mapping = list(dict_labels_to_segment_ids)
            segment_labels_from_id_mapping_hashed = [
                Lime._extract_bands_from_spyndex(segment) for segment in segment_labels_from_id_mapping
            ]
            if len(segment_labels_from_id_mapping) != len(segment_labels):
                raise ValueError("Incorrect dict_labels_to_segment_ids - does not match length with segment labels")

            for segment_label in segment_labels:
                if (
                    segment_label not in segment_labels_from_id_mapping
                    and segment_label not in segment_labels_from_id_mapping_hashed
                ):
                    raise ValueError(f"{segment_label} not present in the dict_labels_to_segment_ids")
                if segment_label in segment_labels_from_id_mapping_hashed:
                    logger.debug(
                        "Detected that the segment labels are converted using hashing function. Hashed flag set to True"
                    )
                    hashed = True  # this type of verification might cause some problems in the future if someone wants to break the code from the inside
            unique_segment_ids = set(dict_labels_to_segment_ids.values())
            if len(unique_segment_ids) != len(segment_labels):
                raise ValueError("Non unique segment ids in the dict_labels_to_segment_ids")
            logger.debug("Passed mapping is correct")

        logger.debug("Creating a single dim band mask")

        band_mask_single_dim = torch.zeros(len(image.wavelengths), dtype=torch.int64, device=device)

        if hashed:
            segment_labels = segment_labels_from_id_mapping  # segment labels without hashing

        for segment_label in segment_labels[::-1]:
            segment_label_hashed = segment_label
            if hashed:
                segment_label_hashed = Lime._extract_bands_from_spyndex(segment_label)
            segment_indices = dict_labels_to_indices[segment_label_hashed]
            segment_id = dict_labels_to_segment_ids[segment_label]

            are_indices_valid = all(0 <= idx < dim for idx, dim in zip(segment_indices, band_mask_single_dim.shape))
            if not are_indices_valid:
                raise ValueError(
                    f"Indices for segment {segment_label} are out of bounds for the one-dimensional band mask of shape {band_mask_single_dim.shape}"
                )
            band_mask_single_dim[segment_indices] = segment_id

        logger.debug("Unsqueezing a band mask to match the image dimensions")
        axis = [0, 1, 2]
        axis.remove(image.band_axis)

        band_mask = band_mask_single_dim.unsqueeze(axis[0]).unsqueeze(axis[1])

        if repeat_dimensions:
            logger.debug(
                "Expanding a band mask with reduced reduced dimensions into the remaining dimensions to fit the image shape"
            )

            size_image = image.image.size()
            size_mask = band_mask.size()

            repeat_dims = [s2 // s1 for s1, s2 in zip(size_mask, size_image)]
            band_mask = band_mask.repeat(repeat_dims)

        if return_dict_labels_to_segment_ids:
            return band_mask, dict_labels_to_segment_ids
        return band_mask

    @validate_call(config=dict(arbitrary_types_allowed=True))  # type: ignore
    def get_spatial_attributes(
        self,
        image: Image,
        segmentation_mask: np.ndarray | torch.Tensor | None = None,
        target: int | None = None,
        segmentation_method: Literal["slic", "patch"] = "slic",
        **segmentation_method_params,
    ) -> ImageSpatialAttributes:
        """function attributes the image using LIME method for spatial data. The function returns ImageSpatialAttributes object that contains the image, the attributions, the segmentation mask, and the score of the interpretable model used for the explanation.

        Args:
            image (Image): an Image for which the attribution is performed
            segmentation_mask (np.ndarray | torch.Tensor | None, optional): A segmentation mask according to which the attribution should be performed. If None, new segmentation mask is created using the `segmentation_method`. Additional parameters for the segmentation method may be passed as kwargs Defaults to None.
            target (int, optional): If model creates more than one output, it analyses the given target. Defaults to None.
            segmentation_method (Literal[&quot;slic&quot;, &quot;patch&quot;], optional): Segmentation method, used only if `segmentation_mask` is None. Defaults to "slic".

        Returns:
            ImageSpatialAttributes: An object that contains the image, the attributions, the segmentation mask, and the score of the interpretable model used for the explanation.
        """
        if self._lime is None:
            raise ValueError("Lime object not initialized")

        if self.explainable_model.problem_type != "regression":
            raise ValueError("For now only the regression problem is supported")

        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(image, segmentation_method, **segmentation_method_params)

        if isinstance(image.image, np.ndarray):
            logger.debug("Converting numpy image to torch tensor")
            image.image = torch.tensor(image.image, device=self._device)
        elif isinstance(image.image, torch.Tensor) and image.image.device != self._device:
            logger.debug("Moving image to the device" + self._device)
            image.image = image.image.to(self._device)

        if isinstance(image.binary_mask, np.ndarray):
            logger.debug("Converting numpy binary mask to torch tensor")
            image.binary_mask = torch.tensor(image.image, device=self._device)
        elif isinstance(image.binary_mask, torch.Tensor) and image.binary_mask.device != self._device:
            logger.debug("Moving binary mask to the device" + self._device)
            image.binary_mask = image.binary_mask.to(self._device)

        if isinstance(segmentation_mask, np.ndarray):
            logger.debug("Converting numpy segmentation mask to torch tensor")
            segmentation_mask = torch.tensor(segmentation_mask, device=self._device)
        elif isinstance(segmentation_mask, torch.Tensor) and segmentation_mask.device != self._device:
            logger.debug("Moving segmentation mask to the device" + self._device)
            segmentation_mask = segmentation_mask.to(self._device)

        assert (
            segmentation_mask.device == self._device
        ), f"Segmentation mask should be on the same device as explainable model {self._device}"
        assert (
            image.image.device == self._device
        ), f"Image data should be on the same device as explainable model {self._device}"

        if not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")

        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=segmentation_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=True,
            return_input_shape=True,
        )

        if score < 0 or score > 1:
            logger.warning("Score is out of range [0, 1]. Clamping the score value to the range")
            score = torch.clamp(score, 0, 1)  # it seems that scikit learn sometimes returns negative values

        spatial_attribution = ImageSpatialAttributes(
            image=image,
            attributes=lime_attributes[0],
            segmentation_mask=segmentation_mask,
            score=score,
        )

        logger.debug("Spatial attribution created")

        return spatial_attribution

    @validate_call(config=dict(arbitrary_types_allowed=True))  # type: ignore
    def get_spectral_attributes(
        self,
        image: Image,
        band_mask: np.ndarray | torch.Tensor | None = None,
        target=None,
        band_names: list[str] | dict[str | tuple[str, ...], int] | None = None,
        verbose=False,
    ) -> ImageSpectralAttributes:
        """function attributes the image using LIME method for spectral data. The function returns ImageSpectralAttributes object that contains the image, the attributions, the band mask, the band names, and the score of the interpretable model used for the explanation.

        Args:
            image (Image): an Image for which the attribution is performed
            band_mask (np.ndarray | torch.Tensor | None, optional): band mask that is used for the spectral attribution, if equals to None, the band mask is created in scope of the function. Defaults to None.
            target (int, optional): If model creates more than one output, it analyses the given target. Defaults to None.
            band_names (list[str] | dict[str  |  tuple[str, ...], int] | None, optional): band names . Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            ImageSpectralAttributes: _description_
        """

        if self._lime is None:
            raise ValueError("Lime object not initialized")

        if self.explainable_model.problem_type != "regression":
            raise ValueError("For now only the regression problem is supported")

        if isinstance(image.image, np.ndarray):
            logger.debug("Converting numpy image to torch tensor")
            image.image = torch.tensor(image.image, device=self._device)
        elif isinstance(image.image, torch.Tensor) and image.image.device != self._device:
            logger.debug("Moving image to the device" + self._device)
            image.image = image.image.to(self._device)

        if isinstance(image.binary_mask, np.ndarray):
            logger.debug("Converting numpy binary mask to torch tensor")
            image.binary_mask = torch.tensor(image.image, device=self._device)
        elif isinstance(image.binary_mask, torch.Tensor) and image.binary_mask.device != self._device:
            logger.debug("Moving binary mask to the device" + self._device)
            image.binary_mask = image.binary_mask.to(self._device)

        assert (
            band_mask.device == self._device  # type: ignore
        ), f"Segmentation mask should be on the same device as explainable model {self._device}"
        assert (
            image.image.device == self._device
        ), f"Image data should be on the same device as explainable model {self._device}"

        if not isinstance(self._lime, LimeBase):
            raise ValueError("Lime object not initialized")

        assert (
            image.image.device == self._device
        ), f"Image data should be on the same device as explainable model {self._device}"

        if band_mask is None:
            band_mask, band_names = self.get_band_mask(image, band_names)  # type: ignore
        elif band_names is None:
            unique_segments = torch.unique(band_mask)
            band_names = {segment: idx for idx, segment in enumerate(unique_segments)}
        else:
            # checking consistency of names
            # unique_segments = torch.unique(band_mask)
            # if isinstance(band_names, dict):
            #     assert set(unique_segments).issubset(set(band_names.values())), "Incorrect band names"
            logger.debug("Band names are provided, using them. In future it there should be an option to validate them")

        if isinstance(band_mask, np.ndarray):
            band_mask = torch.tensor(band_mask, device=self._device)
        else:
            band_mask = band_mask.to(self._device)

        assert (
            band_mask.device == self._device
        ), f"Band mask should be on the same device as explainable model {self._device}"

        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=band_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=verbose,
            return_input_shape=True,
        )

        if score < 0 or score > 1:
            logger.warning("Score is out of range [0, 1]. Clamping the score value to the range")
            score = torch.clamp(score, 0, 1)  # it seems that scikit learn sometimes returns negative values

        lime_attributes = lime_attributes[0]

        spectral_attribution = ImageSpectralAttributes(
            image=image,
            attributes=lime_attributes,
            band_mask=band_mask,
            band_names=band_names,
            score=score,
        )

        return spectral_attribution

    @staticmethod
    def _get_slick_segmentation_mask(image: Image, num_interpret_features: int = 10, *args, **kwargs) -> torch.tensor:
        """creates a segmentation mask using SLIC method

        Args:
            image (Image): an Image for which the segmentation mask is created
            num_interpret_features (int, optional): number of segments. Defaults to 10.

        Returns:
            torch.tensor: an output segmentation mask
        """
        device = image.image.device
        numpy_image = np.array(image.image.to("cpu"))
        segmentation_mask = slic(
            numpy_image,
            n_segments=num_interpret_features,
            mask=np.array(image.get_flattened_binary_mask.to("cpu")),
            channel_axis=image.band_axis,
            *args,
            **kwargs,
        )

        if np.min(segmentation_mask) == 1:
            segmentation_mask -= 1

        # segmentation_mask = np.repeat(np.expand_dims(segmentation_mask, axis=image.band_axis), repeats=image.image.shape[image.band_axis], axis=image.band_axis)
        segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.int64, device=device)
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)
        # segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        return segmentation_mask

    @staticmethod
    def _get_patch_segmentation_mask(image: Image, patch_size=10, *args, **kwargs) -> torch.tensor:
        """creates a segmentation mask using patch method - creates small squares of the same size

        Args:
            image (Image): an Image for which the segmentation mask is created
            patch_size (int, optional): size of the patch, the image size should be divisible by this value. Defaults to 10.


        Returns:
            torch.tensor: an output segmentation mask
        """
        logger.warning("Patch segmentation only works for band_index = 0 now")

        device = image.image.device
        if image.image.shape[1] % patch_size != 0 or image.image.shape[2] % patch_size != 0:
            raise ValueError("Invalid patch_size. patch_size must be a factor of both width and height of the image")

        height, width = image.image.shape[1], image.image.shape[2]

        mask_zero = torch.tensor(image.image.bool()[0], device=device)
        idx_mask = torch.arange(height // patch_size * width // patch_size, device=device).reshape(
            height // patch_size, width // patch_size
        )
        idx_mask += 1
        segmentation_mask = torch.repeat_interleave(idx_mask, patch_size, dim=0)
        segmentation_mask = torch.repeat_interleave(segmentation_mask, patch_size, dim=1)
        segmentation_mask = segmentation_mask * mask_zero
        # segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx

        return segmentation_mask
