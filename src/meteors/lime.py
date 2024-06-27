from __future__ import annotations

from typing import Literal, Callable, Sequence
from abc import ABC
from functools import cached_property

from meteors.utils.models import ExplainableModel, InterpretableModel
from meteors import Image

from meteors.lime_base import Lime as LimeBase

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated, Self

import torch
import numpy as np
import spyndex

import loguru


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
            self.segmentation_mask = torch.tensor(
                self.segmentation_mask, device=self._device
            )

        if self.segmentation_mask.device != self._device:
            self.segmentation_mask = self.segmentation_mask.to(
                self._device
            )  # move to the device

        return self

    def to(self, device: torch.device) -> Self:
        super().to(device)
        self.segmentation_mask = self.segmentation_mask.to(device)
        return self

    def get_flattened_segmentation_mask(self) -> torch.tensor:
        """segmentation mask is after all only two dimensional tensor with some repeated values, this function returns only two-dimensional tensor"""
        if self._flattened_segmentation_mask is None:
            self._flattened_segmentation_mask = self.segmentation_mask.select(
                dim=self.image.band_axis, index=0
            )
        return self._flattened_segmentation_mask

    def get_flattened_attributes(self) -> torch.tensor:
        """attributions for spatial case are after all only two dimensional tensor with some repeated values, this function returns only two-dimensional tensor"""
        if self._flattened_attributes is None:
            self._flattened_attributes = self.attributes.select(
                dim=self.image.band_axis, index=0
            )
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
            self._flattened_band_mask = self.band_mask.select(
                dim=dims_to_select[0], index=0
            ).select(dim=dims_to_select[1], index=0)
        return self._flattened_band_mask

    def get_flattened_attributes(self) -> torch.tensor:
        """attributions for spectral case are after all only one dimensional tensor with some repeated values, this function returns only one-dimensional tensor"""
        if self._flattened_attributes is None:
            dims_to_select = [2, 1, 0]
            dims_to_select.remove(self.image.band_axis)
            self._flattened_attributes = self.attributes.select(
                dim=dims_to_select[0], index=0
            ).select(dim=dims_to_select[1], index=0)
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
            loguru.logger.debug("Not a torch model, setting device to cpu. Error: {e}")
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
        if segmentation_method == "slic":
            return Lime.__get_slick_segmentation_mask(
                image, **segmentation_method_params
            )
        if segmentation_method == "patch":
            return Lime.__get_patch_segmentation_mask(
                image, **segmentation_method_params
            )
        raise NotImplementedError("Only slic and patch methods are supported for now")

    @staticmethod
    def get_band_mask(
        image: Image,
        band_names: list[str | Sequence[str]]
        | dict[tuple[str, ...] | str, int]
        | None = None,
        band_groups: dict[str | tuple[str, ...], Sequence[int]] | None = None,
        band_ranges_indices: dict[
            str | tuple[str, ...], Sequence[tuple[int, int]] | tuple[int, int]
        ]
        | None = None,
        band_ranges_wavelengths: dict[
            str | tuple[str, ...], Sequence[tuple[float, float]] | tuple[float, float]
        ]
        | None = None,
        device: torch.device | str | None = None,
        repeat_dimensions: bool = False,
    ) -> tuple[torch.Tensor, dict[tuple[str, ...] | str, int]]:
        """function generates band mask - an array that corresponds to the image, which values are different segments.
        Args:
            image (Image): A Hyperspectral image
            band_names ((List[str | List[str]]) | (Dict[str | List[str], Iterable[int]])): list of band names that should be treated as one segment or a dictionary containing band names as keys and segment values.
            Band names should either be included in `spyndex.indices` or `spyndex.bands`. For instance it could be have values as: `["R", "G", "B"]` or `["IPVI", "AFRI1600"]`. Alternatively, it could be a dictionary with band names as keys and segment ids as values, e.g. `{"R": 1, "G": 2, "B": 3}` or `{"IPVI": 1, "AFRI1600": 2}`.
            band_groups (Dict[str, Iterable[int]]): dictionary containing user-made band groups as keys and list of band indices as values. For instance it could be `{"A": [0, 1, 2, 3, 4], "B": [6, 7, 8, 20, 21, 22]}`. "A" and "B" would be the names of the segments, and [0, 1, 2, 3, 4] and [6, 7, 8, 20, 21, 22] would be the indices of the bands that belong to the segments. The indices should be in the range of the number of bands in the image.
            band_ranges_indices (Dict[str, Sequence[Tuple[int, int]] | Tuple[int, int]]): dictionary containing user-made band ranges as keys and tuple of start and end indices as values. For instance it could equal to `{"A": (0, 4), "B": [(6, 9), (20, 23)]}`. "A" and "B" would be the names of the segments, the created segment indices would equal to [0, 1, 2, 3, 4] and [6, 7, 8, 20, 21, 22] 
            band_ranges_wavelengths (Dict[str, Sequence[Tuple[float, float]] | Tuple[float, float]]): dictionary containing user-made band ranges as keys and tuple of start and end wavelengths as values. For instance, if our image contains wavelengths [400, 405, 410, 420, 450], then dictionary could equal to `{"A": (400, 410), "B": [(420, 450)]}`. "A" and "B" would be the names of the segments, the created segment indices would equal to [0, 1, 2] and [3, 4]

            device (torch.device | str | None): a device on which the band mask will be created. If None, it is created on the same device as the image. Defaults to None
            repeat_dimensions (bool, optional): Whether to repeat the band mask over all image dimensions For instance, if False the output mask could have shape (1, 1, 150). If True - (64, 64, 150). Defaults to False


        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: a tuple which consists of an actual band mask and a dictionary that translates the band names into the segment values
        """
        dict_labels_to_segment_ids = None
        if band_names is not None:
            loguru.logger.debug("Getting band mask from band names of spectral bands")
            band_ranges_wavelengths, dict_labels_to_segment_ids = (
                Lime.__get_band_ranges_wavelengths_from_band_names(image, band_names)  # type: ignore
            )
        if band_ranges_wavelengths is not None:
            loguru.logger.debug(
                "Getting band mask from band groups given by ranges of wavelengths"
            )
            band_ranges_indices = (
                Lime.__get_band_range_indices_from_band_ranges_wavelengths(
                    image,
                    band_ranges_wavelengths,  # type: ignore
                )
            )

        if band_ranges_indices is not None:
            loguru.logger.debug(
                "Getting band mask from band groups given by ranges of indices"
            )
            band_groups = Lime.__get_band_groups_from_band_ranges_indices(
                band_ranges_indices
            )  # type: ignore

        if band_groups is None:
            raise ValueError("No band names, groups, or ranges provided")

        return Lime.__create_tensor_band_mask(
            image,
            band_groups,
            dict_labels_to_segment_ids=dict_labels_to_segment_ids,
            device=device,
            repeat_dimensions=repeat_dimensions,
            return_dict_labels_to_segment_ids=True,
        )

    @staticmethod
    def __make_band_names_hashable(
        segment_name: Sequence[str] | str,
    ) -> tuple[str, ...] | str:
        if isinstance(segment_name, str):
            return segment_name
        if isinstance(segment_name, Sequence):
            return tuple(segment_name)
        raise ValueError(
            f"Incorrect segment {segment_name} type. Should be either a sequence or string"
        )

    @staticmethod
    def __extract_bands_from_spyndex(
        segment_name: Sequence[str] | str,
    ) -> tuple[str, ...]:
        band_names_segment: list[str] = []

        if isinstance(segment_name, str):
            segment_name = tuple([segment_name])
        for band_name in segment_name:
            if band_name in spyndex.indices:
                band_names_segment = band_names_segment + (
                    spyndex.indices[band_name].bands
                )
            elif band_name in spyndex.bands:
                band_names_segment.append(band_name)
            else:
                raise ValueError(
                    f"Invalid band name {band_name}, band name must be either in `spyndex.indices` or `spyndex.bands`"
                )

        return tuple(band_names_segment)

    @staticmethod
    def __get_band_ranges_wavelengths_from_band_names(
        image: Image,
        band_names: list[str | Sequence[str]] | dict[tuple[str, ...] | str, int],
    ) -> tuple[
        dict[str | tuple[str, ...], list[tuple[float, float]]],
        dict[str | tuple[str, ...], int],
    ]:
        """function extracts ranges of wavelengths from the band names

        Args:
            image (Image): an image for which the band mask is created. The only important element of image are the wavelengths
            band_names (list[str  |  Sequence[str]] | dict[tuple[str, ...]  |  str, int]): list or dictionary with band names or segments

        Returns:
            dict[str | tuple[str, ...], Sequence[tuple[float, float]]], dict[str | tuple[str, ...], int]: tuple containing the dictionary with mapping segment labels into wavelength ranges and mapping from segment labels into segment ids
        """

        segments_list = []
        if isinstance(band_names, Sequence):
            loguru.logger.debug(
                "band_names is a list of segments, creating a dictionary of segments"
            )

            band_names_hashed = [
                Lime.__make_band_names_hashable(segment) for segment in band_names
            ]

            dict_labels_to_segment_ids = {
                segment: idx + 1 for idx, segment in enumerate(band_names_hashed)
            }
            segments_list = band_names_hashed
        elif isinstance(band_names, dict):
            loguru.logger.debug("band_names is a dictionary of segments")
            dict_labels_to_segment_ids = band_names
            segments_list = list(band_names.keys())
        else:
            raise ValueError(
                "Incorrect band_names type. It should be a dict or a Sequence"
            )

        segments_list_after_mapping = [
            Lime.__extract_bands_from_spyndex(segment) for segment in segments_list
        ]
        band_ranges_wavelengths: dict[
            str | tuple[str, ...], list[tuple[float, float]]
        ] = {}
        for segment in segments_list_after_mapping:
            band_ranges_wavelengths[segment] = []
            for band_name in segment:
                min_wavelength = spyndex.bands[band_name].min_wavelength
                max_wavelength = spyndex.bands[band_name].max_wavelength

                band_ranges_wavelengths[segment].append(
                    (min_wavelength, max_wavelength)
                )

        return band_ranges_wavelengths, dict_labels_to_segment_ids

    @staticmethod
    def __get_band_range_indices_from_band_ranges_wavelengths(
        image: Image,
        band_ranges_wavelengths: dict[
            str | tuple[str, ...], list[tuple[float, float]] | tuple[float, float]
        ],
    ) -> dict[str | tuple[str, ...], list[tuple[int, int]]]:
        loguru.logger.debug("Verifying and unifying format of band_ranges_indices")

        for segment_label, segment_range in band_ranges_wavelengths.items():
            if not isinstance(segment_range, Sequence):
                raise ValueError(
                    f"Incorrect type for range of segment with label {segment_label}"
                )
            if (
                len(segment_range) == 2
                and isinstance(segment_range[0], (int, float))
                and isinstance(segment_range[1], (int, float))
            ):
                # cast into Sequence[Tuple[float, float]] in case type is Tuple[float, float]
                segment_range = [segment_range]  # type: ignore
                band_ranges_wavelengths[segment_label] = segment_range  # type: ignore

            for segment_part in segment_range:
                if (
                    not isinstance(segment_part, Sequence)
                    or len(segment_part) != 2
                    or not isinstance(segment_part[0], (int, float))
                    or not isinstance(segment_part[1], (int, float))
                ):
                    raise ValueError(
                        f"Segment {segment_label} has incorrect structure - it should be a Tuple of length 2 or a Sequence with Tuples of length 2"
                    )
                if segment_part[0] >= segment_part[1]:
                    raise ValueError(f"Order of the range {segment_label} is incorrect")

        loguru.logger.debug("Mapping the wavelength ranges to get segment indices")

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
    def __get_band_groups_from_band_ranges_indices(
        band_ranges_indices: dict[
            str | tuple[str, ...], Sequence[tuple[int, int]] | tuple[int, int]
        ],
    ) -> dict[str | tuple[str, ...], list[int]]:
        loguru.logger.debug("Verifying format of band_ranges_indices")
        for segment_label, segment_range in band_ranges_indices.items():
            if not isinstance(segment_range, Sequence):
                raise ValueError(
                    f"Incorrect type for range of segment with label {segment_label}"
                )
            if (
                len(segment_range) == 2
                and isinstance(segment_range[0], int)
                and isinstance(segment_range[1], int)
            ):
                # cast into Sequence[Tuple[int, int]] in case type is Tuple[int, int]
                segment_range = [segment_range]  # type: ignore
                band_ranges_indices[segment_label] = segment_range  # type: ignore

            for segment_part in segment_range:
                if not isinstance(segment_part, Sequence) or len(segment_part) != 2:
                    raise ValueError(
                        f"Segment {segment_label} has incorrect structure - it should be a Tuple of length 2 or a Sequence with Tuples of length 2"
                    )
                if segment_part[0] > segment_part[1]:
                    raise ValueError(f"Order of the range {segment_label} is incorrect")

        loguru.logger.debug("Filling the ranges to get segment indices")

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
    def __create_tensor_band_mask(
        image: Image,
        dict_labels_to_indices: dict[str | tuple[str, ...], Sequence[int]],
        dict_labels_to_segment_ids: dict[str | tuple[str, ...], int] | None = None,
        device: torch.device | str | None = None,
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

        loguru.logger.debug(
            f"Creating a band mask on the device {device} using {len(segment_labels)} segments"
        )

        loguru.logger.debug("Checking if the bands are overlapping")
        overlapping_segments = {wavelength: None for wavelength in image.wavelengths}
        for segment_label, indices in dict_labels_to_indices.items():
            for idx in indices:
                if overlapping_segments[image.wavelengths[idx]] is not None:
                    loguru.logger.warning(
                        f"Bands {overlapping_segments[image.wavelengths[idx]]} and {segment_label} are overlapping"
                    )
                overlapping_segments[image.wavelengths[idx]] = segment_label  # type: ignore

        # check if there exists a dictionary dict_labels_to_segment_ids, if not, create it
        if dict_labels_to_segment_ids is None:
            loguru.logger.debug("Creating mapping from segment labels into ids")
            dict_labels_to_segment_ids = {
                segment: idx + 1 for idx, segment in enumerate(segment_labels)
            }
        else:
            loguru.logger.debug(
                "Using existing mapping from segment labels into segment ids"
            )
            loguru.logger.debug("Verifying if the passed mapping is correct")
            segment_labels_from_id_mapping = list(dict_labels_to_segment_ids)
            segment_labels_from_id_mapping_hashed = [
                Lime.__extract_bands_from_spyndex(segment)
                for segment in segment_labels_from_id_mapping
            ]
            if len(segment_labels_from_id_mapping) != len(segment_labels):
                raise ValueError(
                    "Incorrect dict_labels_to_segment_ids - does not match length with segment labels"
                )

            for segment_label in segment_labels:
                if (
                    segment_label not in segment_labels_from_id_mapping
                    and segment_label not in segment_labels_from_id_mapping_hashed
                ):
                    raise ValueError(
                        f"{segment_label} not present in the dict_labels_to_segment_ids"
                    )
                if segment_label in segment_labels_from_id_mapping_hashed:
                    loguru.logger.debug(
                        "Detected that the segment labels are converted using hashing function. Hashed flag set to True"
                    )
                    hashed = True  # this type of verification might cause some problems in the future if someone wants to break the code from the inside
            unique_segment_ids = set(dict_labels_to_segment_ids.values())
            if len(unique_segment_ids) != len(segment_labels):
                raise ValueError(
                    "Non unique segment ids in the dict_labels_to_segment_ids"
                )
            loguru.logger.debug("Passed mapping is correct")

        loguru.logger.debug("Creating a single dim band mask")

        band_mask_single_dim = torch.zeros(
            len(image.wavelengths), dtype=torch.int64, device=device
        )

        if hashed:
            segment_labels = (
                segment_labels_from_id_mapping  # segment labels without hashing
            )

        for segment_label in segment_labels[::-1]:
            segment_label_hashed = segment_label
            if hashed:
                segment_label_hashed = Lime.__extract_bands_from_spyndex(segment_label)
            segment_indices = dict_labels_to_indices[segment_label_hashed]
            segment_id = dict_labels_to_segment_ids[segment_label]

            are_indices_valid = all(
                0 <= idx < dim
                for idx, dim in zip(segment_indices, band_mask_single_dim.shape)
            )
            if not are_indices_valid:
                raise ValueError(
                    f"Indices for segment {segment_label} are out of bounds for the one-dimensional band mask of shape {band_mask_single_dim.shape}"
                )
            band_mask_single_dim[segment_indices] = segment_id

        loguru.logger.debug("Unsqueezing a band mask to match the image dimensions")
        axis = [0, 1, 2]
        axis.remove(image.band_axis)

        band_mask = band_mask_single_dim.unsqueeze(axis[0]).unsqueeze(axis[1])

        if repeat_dimensions:
            loguru.logger.debug(
                "Expanding a band mask with reduced reduced dimensions into the remaining dimensions to fit the image shape"
            )

            size_image = image.image.size()
            size_mask = band_mask.size()

            repeat_dims = [s2 // s1 for s1, s2 in zip(size_mask, size_image)]
            band_mask = band_mask.repeat(repeat_dims)

        if return_dict_labels_to_segment_ids:
            return band_mask, dict_labels_to_segment_ids
        return band_mask

    def get_spatial_attributes(
        self,
        image: Image,
        segmentation_mask: np.ndarray | torch.Tensor | None = None,
        target=None,
        segmentation_method: Literal["slic", "patch"] = "slic",
        **segmentation_method_params,
    ) -> ImageSpatialAttributes:
        assert self._lime is not None, "Lime object not initialized"

        assert (
            self.explainable_model.problem_type == "regression"
        ), "For now only the regression problem is supported"

        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(
                image, segmentation_method, **segmentation_method_params
            )

        if isinstance(image.image, np.ndarray):
            image.image = torch.tensor(image.image, device=self._device)
        elif isinstance(image.image, torch.Tensor):
            image.image = image.image.to(self._device)

        if isinstance(image.binary_mask, np.ndarray):
            image.binary_mask = torch.tensor(image.image, device=self._device)
        elif isinstance(image.binary_mask, torch.Tensor):
            image.binary_mask = image.binary_mask.to(self._device)

        if isinstance(segmentation_mask, np.ndarray):
            segmentation_mask = torch.tensor(segmentation_mask, device=self._device)
        elif isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = segmentation_mask.to(self._device)

        assert (
            segmentation_mask.device == self._device
        ), f"Segmentation mask should be on the same device as explainable model {self._device}"
        assert (
            image.image.device == self._device
        ), f"Image data should be on the same device as explainable model {self._device}"

        assert isinstance(self._lime, LimeBase), "Lime object not initialized"

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
            loguru.logger.warning(
                "Score is out of range [0, 1]. Clamping the score value to the range"
            )
            score = torch.clamp(
                score, 0, 1
            )  # it seems that scikit learn sometimes returns negative values

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
        band_names: list[str] | dict[str | tuple[str, ...], int] | None = None,
        verbose=False,
    ) -> ImageSpectralAttributes:
        assert self._lime is not None, "Lime object not initialized"

        assert (
            self.explainable_model.problem_type == "regression"
        ), "For now only the regression problem is supported"

        if isinstance(image.image, np.ndarray):
            image.image = torch.tensor(image.image, device=self._device)
        elif isinstance(image.image, torch.Tensor):
            image.image = image.image.to(self._device)

        if isinstance(image.binary_mask, np.ndarray):
            image.binary_mask = torch.tensor(image.image, device=self._device)
        elif isinstance(image.binary_mask, torch.Tensor):
            image.binary_mask = image.binary_mask.to(self._device)

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
            pass

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
            loguru.logger.warning(
                "Score is out of range [0, 1]. Clamping the score value to the range"
            )
            score = torch.clamp(
                score, 0, 1
            )  # it seems that scikit learn sometimes returns negative values

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
    def __get_slick_segmentation_mask(
        image: Image, num_interpret_features: int = 10, *args, **kwargs
    ) -> torch.tensor:
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
        segmentation_mask = torch.tensor(
            segmentation_mask, dtype=torch.int64, device=device
        )
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)
        # segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        return segmentation_mask

    @staticmethod
    def __get_patch_segmentation_mask(
        image: Image, patch_size=10, *args, **kwargs
    ) -> torch.tensor:
        print("Patch segmentation only works for band_index = 0 now")

        device = image.image.device
        if (
            image.image.shape[1] % patch_size != 0
            or image.image.shape[2] % patch_size != 0
        ):
            raise ValueError(
                "Invalid patch_size. patch_size must be a factor of both width and height of the image"
            )

        height, width = image.image.shape[1], image.image.shape[2]

        mask_zero = torch.tensor(image.image.bool()[0], device=device)
        idx_mask = torch.arange(
            height // patch_size * width // patch_size, device=device
        ).reshape(height // patch_size, width // patch_size)
        idx_mask += 1
        segmentation_mask = torch.repeat_interleave(idx_mask, patch_size, dim=0)
        segmentation_mask = torch.repeat_interleave(
            segmentation_mask, patch_size, dim=1
        )
        segmentation_mask = segmentation_mask * mask_zero
        # segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx

        return segmentation_mask
