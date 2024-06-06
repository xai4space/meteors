from typing import Literal, List, Callable, Dict, Tuple, Iterable, Sequence
from abc import ABC, abstractmethod

from meteors.utils.models import ExplainableModel, InterpretableModel
from meteors import Image

from meteors.lime_base import Lime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated, Union, Self

import torch
import numpy as np
import pandas as pd
import spyndex


try:
    from fast_slic import Slic as slic
except ImportError:
    from skimage.segmentation import slic

# important - specify the image orientation 
# Width x Height x Channels seems to be the most apropiate
# but the model requires  (C, W, H) or (W, H C)


# explanations

class Explanation(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    
        
class ImageAttributes(Explanation):
    
    image: Annotated[Image, Field(kw_only=False, validate_default=True, description='Hyperspectral image object on which the attribution is performed.')]
    attributes: Annotated[np.ndarray | torch.Tensor, Field(kw_only=False, validate_default=True, description='Attributions saved as a numpy array or torch tensor.')]
    score: Annotated[float, Field(kw_only=False, validate_default=True, description='R^2 score of interpretable model used for the explanation.')]
    
    _device: torch.device = None
    _flattened_attributes: torch.Tensor = None
    
    
    
    @model_validator(mode="before")
    def validate_attributes(cls, values):
        assert values["attributes"].shape == values["image"].image.shape, "Attributes must have the same shape as the image"
        
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
    segmentation_mask: Annotated[np.ndarray | torch.Tensor, Field(kw_only=False, validate_default=True, description='Segmentation mask used for the explanation.')]
    
    _flattened_segmentation_mask: torch.Tensor = None
    
    @model_validator(mode="after")
    def validate_segmentation_mask(self) -> Self:
        if isinstance(self.segmentation_mask, np.ndarray):
            self.segmentation_mask = torch.tensor(self.segmentation_mask, device=self._device)
        
        if self.segmentation_mask.device != self._device:
            self.segmentation_mask = self.segmentation_mask.to(self._device) # move to the device
            
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
    
    band_mask: Annotated[np.ndarray | torch.Tensor, Field(kw_only=False, validate_default=True, description='Band mask used for the explanation.')]
    band_names: Annotated[Dict[str, int], Field(kw_only=False, validate_default=True, description='Dictionary that translates the band names into the segment values.')]
    
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
            dims_to_select = [2,1,0]
            dims_to_select.remove(self.image.band_axis)
            self._flattened_band_mask = self.band_mask.select(dim=dims_to_select[0], index=0).select(dim=dims_to_select[1], index=0)            
        return self._flattened_band_mask
    
    def get_flattened_attributes(self) -> torch.tensor:
        """attributions for spectral case are after all only one dimensional tensor with some repeated values, this function returns only one-dimensional tensor"""
        if self._flattened_attributes is None:
            dims_to_select = [2,1,0]
            dims_to_select.remove(self.image.band_axis)
            self._flattened_attributes = self.attributes.select(dim=dims_to_select[0], index=0).select(dim=dims_to_select[1], index=0)            
        return self._flattened_attributes
    
    
        
        

# explainer itself

class Explainer(BaseModel, ABC):
    explainable_model: ExplainableModel
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    
    _device: torch.device = None
    
    
    @model_validator(mode="after")
    def validate_device(self) -> Self:
        self._device = next(self.explainable_model.forward_func.parameters()).device
        return self
    
    
    def to(self, device: torch.device) -> Self:
        self.explainable_model.to(device)
        return self


class Lime(Explainer):
    # should it be any different than base lime?
    explainable_model: ExplainableModel
    interpretable_model: InterpretableModel
    similarity_func: Callable = None
    perturb_func: Callable | None = None
    
    _lime = None
    
    @model_validator(mode="after")
    def construct_lime(self) -> Self:
        self._lime = Lime(
            forward_func=self.explainable_model.forward_func,
            interpretable_model=self.interpretable_model,
            similarity_func=self.similarity_func,
            perturb_func=self.perturb_func
        )

        return self
    
    
    def to(self, device: torch.device) -> Self:
        super().to(device)
        #self.interpretable_model.to(device)
        return self
        

    @staticmethod
    def get_segmentation_mask(image: Image, segmentation_method: Literal["patch", "slic"] | None = None, segmentation_method_params = {}) -> torch.Tensor:
        if segmentation_method == "slic":
            return Lime.__get_slick_segmentation_mask(image, **segmentation_method_params)
        if segmentation_method == "patch":
            return Lime.__get_patch_segmentation_mask(image, **segmentation_method_params)
        raise NotImplementedError("Only slic method is supported for now")

    @staticmethod
    def get_band_mask(image: Image, band_names: Sequence[str | Sequence[str]] | Dict[str | Tuple[str], Iterable[int]]) -> Tuple[torch.Tensor, Dict[str | Tuple[str], int]]:
        """function generates band mask - an array that corresponds to the image, which values are different segments. 
        Args:
            image (Image): A Hyperspectral image
            band_names ((List[str | List[str]]) | (Dict[str | List[str], Iterable[int]])): list of band names that should be treated as one segment or dictionary containing

        Returns:
            Tuple[torch.Tensor, Dict[str, int]]: a tuple which consists of an actual band mask and a dictionary that translates the band names into the segment values
        """
        
        if isinstance(band_names, Sequence):
            band_names = Lime.__get_band_dict_from_list(band_names)
        
        band_names_simplified = {segment[0] if len(segment) == 1 else str(segment): band_names[segment]  for segment in band_names}
        
        return (Lime.__get_band_mask_from_names_dict(image, band_names), band_names_simplified)
    
    
    @staticmethod
    def __get_band_dict_from_list(band_names_list: List[str | Sequence[str]]) -> Dict[Tuple[str], int]:
        band_names_dict = {}
        for idx, segment in enumerate(band_names_list):
            if isinstance(segment, str):
                segment = [segment]
            segment = tuple(segment)
            
            band_names_dict[segment] = idx + 1
        return band_names_dict
    
    
    @staticmethod
    def __get_band_mask_from_names_dict(image: Image, band_names: Dict[str | Tuple[str], int] = None) ->  Tuple[torch.Tensor, Dict[Tuple[str], int]]:
        grouped_band_names = Lime.__get_grouped_band_names(band_names)
        
        device = image.image.device
        resolution_segments = Lime.__get_resolution_segments(image.wavelengths, grouped_band_names, device=image.image.device)
        
        
        axis = [0,1,2]
        axis.remove(image.band_axis)
                
        band_mask = resolution_segments.unsqueeze(axis[0]).unsqueeze(axis[1])
        size_image = image.image.size()
        size_mask = band_mask.size()

        repeat_dims = [s2 // s1 for s1, s2 in zip(size_mask, size_image)]
        band_mask = band_mask.repeat(repeat_dims)
        
        return band_mask
        
    @staticmethod
    def __get_grouped_band_names(band_names: Dict[str | Tuple[str], int]) -> Dict[Tuple[str], int]:
        # function extracts band names or indices based on the spyndex library
        # also checks if the given names are valid
        
        grouped_band_names = {}

        for segment in band_names.keys():
            band_names_segment = []
            if not isinstance(segment, Sequence) or isinstance(segment, str):
                segment = [segment]
                
            segment = tuple(segment)
                
            for band_name in segment:    
                if band_name in spyndex.indices:
                    band_names_segment = band_names_segment + (spyndex.indices[band_name].bands)
                elif band_name in spyndex.bands:
                    band_names_segment.append(band_name)
                else:
                    raise ValueError(f'Invalid band name {band_name}, band name must be either in `spyndex.indices` or `spyndex.bands`')
                
            band_names_segment = tuple(band_names_segment)
            grouped_band_names[band_names_segment] = band_names[segment]
            
        return grouped_band_names        
        
    @staticmethod
    def __get_resolution_segments(wavelengths: Iterable, band_names: Dict[Iterable[str], int], device="cpu") -> Iterable[int]:
        
        resolution_segments = torch.zeros(len(wavelengths), dtype=torch.int64, device=device)

        segments = list(band_names.keys())
        for segment in segments[::-1]:
            for band_name in segment:
                min_wavelength = spyndex.bands[band_name].min_wavelength
                max_wavelength = spyndex.bands[band_name].max_wavelength
                
                for wave_idx, wave_val in enumerate(wavelengths):
                    if min_wavelength <= wave_val <= max_wavelength:
                        resolution_segments[wave_idx] = band_names[segment]
                        
        unique_segments = torch.unique(resolution_segments)
        for segment in band_names.keys():
            if band_names[segment] not in unique_segments:
                display_name = segment
                if len(segment) == 1:
                    display_name = segment[0]
                print(f"bands {display_name} not found in the wavelengths or bands are overlapping")      
        return resolution_segments
    

    def get_spatial_attributes(self, image: Image, segmentation_mask: np.ndarray | torch.Tensor | None = None, target = None, segmentation_method: Literal["slic", "patch"] | None = "slic", segmentation_method_params: dict| None = {}) -> ImageSpatialAttributes:
        assert self.explainable_model.problem_type == "regression", "For now only the regression problem is supported" 
        if segmentation_mask is None:
            segmentation_mask = self.get_segmentation_mask(image, segmentation_method, segmentation_method_params)
        

        if isinstance(image.image, np.ndarray):
            image.image = torch.tensor(image.image, device = self._device)
        
        if isinstance(image.binary_mask, np.ndarray):
            image.binary_mask = torch.tensor(image.image, device = self._device)
        
        if isinstance(segmentation_mask, np.ndarray):
            segmentation_mask = torch.tensor(segmentation_mask, device = self._device)
            
        assert segmentation_mask.device == self._device, f"Segmentation mask should be on the same device as explainable model {self._device}"
        assert image.binary_mask.device == self._device, f"Image binary mask should be on the same device as explainable model {self._device}"
        assert image.image.device == self._device, f"Image data should be on the same device as explainable model {self._device}"
        
        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=segmentation_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=True,
            return_input_shape=True,
        )
        
        spatial_attribution = ImageSpatialAttributes(image=image, attributes=lime_attributes[0], segmentation_mask=segmentation_mask, score=score)
        
        return spatial_attribution
        

    def get_spectral_attributes(self, image: Image, band_mask: np.ndarray | torch.Tensor | None = None, target = None, band_names: List["str"] | Dict[str | Tuple[str], int] = None, verbose = False) -> ImageSpectralAttributes:
        assert self.explainable_model.problem_type == "regression", "For now only the regression problem is supported" 
        
        if isinstance(image.image, np.ndarray):
            image.image = torch.tensor(image.image, device = self._device)
        
        if isinstance(image.binary_mask, np.ndarray):
            image.binary_mask = torch.tensor(image.image, device = self._device)
        
        assert image.image.device == self._device, f"Image data should be on the same device as explainable model {self._device}"
        assert image.binary_mask.device == self._device, f"Image binary mask should be on the same device as explainable model {self._device}"
        
        
        
        if band_mask is None:
            band_mask, band_names = self.get_band_mask(image, band_names)
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
            band_mask = torch.tensor(band_mask, device = self._device)
            
        assert band_mask.device == self._device, f"Band mask should be on the same device as explainable model {self._device}"
        
        lime_attributes, score = self._lime.attribute(
            inputs=image.image.unsqueeze(0),
            target=target,
            feature_mask=band_mask.unsqueeze(0),
            n_samples=10,
            perturbations_per_eval=4,
            show_progress=verbose,
            return_input_shape=True,
        )
        
        lime_attributes = lime_attributes[0]
        
        spectral_attribution = ImageSpectralAttributes(image=image, attributes=lime_attributes, band_mask=band_mask, band_names=band_names, score=score)
        
        return spectral_attribution        

    @staticmethod
    def __get_slick_segmentation_mask(image: Image, num_interpret_features: int, *args, **kwargs) -> torch.tensor:
        device = image.image.device
        numpy_image = np.array(image.image.to("cpu"))
        segmentation_mask = slic(numpy_image, n_segments=num_interpret_features, mask = np.array(image.get_flattened_binary_mask().to("cpu")), channel_axis = image.band_axis, *args, **kwargs)
        
        if np.min(segmentation_mask) == 1:
            segmentation_mask -= 1
        
        # segmentation_mask = np.repeat(np.expand_dims(segmentation_mask, axis=image.band_axis), repeats=image.image.shape[image.band_axis], axis=image.band_axis)
        segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.int64, device=device)
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)
        #segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        return segmentation_mask
    
    @staticmethod
    def __get_patch_segmentation_mask(image: Image, patch_size = 10, *args, **kwargs) -> torch.tensor:
        print("Patch segmentation only works for band_index = 0 now")
        
        device = image.image.device
        if image.image.shape[1] % patch_size != 0 or image.image.shape[2] % patch_size != 0:
            raise ValueError('Invalid patch_size. patch_size must be a factor of both width and height of the image')

        height, width = image.image.shape[1], image.image.shape[2]

        mask_zero = torch.tensor(image.image.bool()[0], device = device)
        idx_mask = torch.arange(height // patch_size * width // patch_size, device=device).reshape(height // patch_size, width // patch_size)
        idx_mask += 1
        segmentation_mask = torch.repeat_interleave(idx_mask, patch_size, dim=0)
        segmentation_mask = torch.repeat_interleave(segmentation_mask, patch_size, dim=1)
        segmentation_mask = segmentation_mask * mask_zero
        #segmentation_mask = torch.repeat_interleave(torch.unsqueeze(segmentation_mask, dim=image.band_axis), repeats=image.image.shape[image.band_axis], dim=image.band_axis)
        segmentation_mask = torch.unsqueeze(segmentation_mask, dim=image.band_axis)

        mask_idx = np.unique(segmentation_mask).tolist()
        for idx, mask_val in enumerate(mask_idx):
            segmentation_mask[segmentation_mask == mask_val] = idx
            
        return segmentation_mask
        
