import pytest
import warnings
from loguru import logger

import torch
import numpy as np

from pydantic import ValidationError

import meteors as mt
import meteors.lime as mt_lime
import meteors.lime_base as mt_lime_base
from meteors.utils.models import ExplainableModel, SkLearnLasso
from meteors.utils.utils import agg_segmentation_postprocessing

# Temporary solution for wavelengths
wavelengths = [
    462.08,
    465.27,
    468.47,
    471.67,
    474.86,
    478.06,
    481.26,
    484.45,
    487.65,
    490.85,
    494.04,
    497.24,
    500.43,
    503.63,
    506.83,
    510.03,
    513.22,
    516.42,
    519.61,
    522.81,
    526.01,
    529.2,
    532.4,
    535.6,
    538.79,
    541.99,
    545.19,
    548.38,
    551.58,
    554.78,
    557.97,
    561.17,
    564.37,
    567.56,
    570.76,
    573.96,
    577.15,
    580.35,
    583.55,
    586.74,
    589.94,
    593.14,
    596.33,
    599.53,
    602.73,
    605.92,
    609.12,
    612.32,
    615.51,
    618.71,
    621.91,
    625.1,
    628.3,
    631.5,
    634.69,
    637.89,
    641.09,
    644.28,
    647.48,
    650.67,
    653.87,
    657.07,
    660.27,
    663.46,
    666.66,
    669.85,
    673.05,
    676.25,
    679.45,
    682.64,
    685.84,
    689.03,
    692.23,
    695.43,
    698.62,
]


class ValidationInfoMock:
    def __init__(self, data):
        self.data = data


#####################################################################
############################ VALIDATIONS ############################
#####################################################################


def test_ensure_torch_tensor():
    # Test with numpy array
    np_array = np.array([1, 2, 3])
    result = mt_lime.ensure_torch_tensor(np_array, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([1, 2, 3])))

    # Test with torch tensor
    torch_tensor = torch.tensor([4, 5, 6])
    result = mt_lime.ensure_torch_tensor(torch_tensor, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([4, 5, 6])))

    # Test with invalid input type
    with pytest.raises(TypeError):
        mt_lime.ensure_torch_tensor("invalid", "Input must be a numpy array or torch tensor")


def test_validate_and_convert_attributes():
    # Test with numpy array
    attributes_np = np.ones((3, 4))
    attributes_torch = mt_lime.validate_and_convert_attributes(attributes_np)
    assert isinstance(attributes_torch, torch.Tensor)
    assert torch.all(attributes_torch.eq(torch.tensor(attributes_np)))

    # Test with torch tensor
    attributes_torch = torch.ones((3, 4))
    attributes_torch_validated = mt_lime.validate_and_convert_attributes(attributes_torch)
    assert isinstance(attributes_torch_validated, torch.Tensor)
    assert torch.all(attributes_torch_validated.eq(attributes_torch))

    # Test with invalid type
    with pytest.raises(TypeError):
        mt_lime.validate_and_convert_attributes(123)


def test_validate_and_convert_mask():
    # Test case 1: Valid mask (numpy array)
    segmentation_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_lime.validate_and_convert_mask(segmentation_mask_np)
    assert isinstance(result_np, torch.Tensor)
    assert torch.all(torch.eq(result_np, torch.tensor(segmentation_mask_np)))

    # Test case 2: Valid mask (torch tensor)
    segmentation_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_lime.validate_and_convert_mask(segmentation_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.all(torch.eq(result_tensor, segmentation_mask_tensor))

    # Test case 3: Invalid mask (wrong type)
    invalid_segmentation_mask = "invalid"
    with pytest.raises(TypeError):
        mt_lime.validate_and_convert_mask(invalid_segmentation_mask)

    # Test case 4: Invalid mask (wrong type)
    invalid_segmentation_mask = 123
    with pytest.raises(TypeError):
        mt_lime.validate_and_convert_mask(invalid_segmentation_mask)

    # Test case 5: None mask
    segmentation_mask_none = None
    result_none = mt_lime.validate_and_convert_mask(segmentation_mask_none)
    assert result_none is None


def test_validate_shapes():
    shape = (len(wavelengths), 240, 240)
    attributes = torch.ones(shape)
    hsi = mt.HSI(image=torch.ones(shape), wavelengths=wavelengths)

    # Test case 1: Valid shapes
    mt_lime.validate_shapes(attributes, hsi)  # No exception should be raised

    # Test case 2: Invalid shapes
    invalid_attributes = torch.ones((150, 240, 241))
    with pytest.raises(ValueError):
        mt_lime.validate_shapes(invalid_attributes, hsi)

    invalid_hsi = mt.HSI(image=torch.ones((len(wavelengths), 240, 241)), wavelengths=wavelengths)
    with pytest.raises(ValueError):
        mt_lime.validate_shapes(attributes, invalid_hsi)


def test_align_band_names_with_mask():
    # Test case 1: Changed band names
    band_names = {"R": 1, "G": 2, "B": 3}
    band_mask = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 3]])

    with pytest.warns(UserWarning):
        updated_band_names = mt_lime.align_band_names_with_mask(band_names, band_mask)

    assert updated_band_names == {
        "R": 1,
        "G": 2,
        "B": 3,
        "not_included": 0,
    }

    # Test case 2: Not changed band names
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.tensor([[1, 0, 1], [1, 2, 1], [1, 0, 1]])

    not_updated_band_names = mt_lime.align_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {
        "R": 0,
        "G": 1,
        "B": 2,
    }

    band_names = {"R": 0}
    band_mask = torch.tensor([[0], [0], [0]])

    not_updated_band_names = mt_lime.align_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {"R": 0}

    # Test case 3: Invalid band names
    band_names = {"R": 1, "G": 2, "B": 3}
    invalid_band_mask = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        mt_lime.align_band_names_with_mask(band_names, invalid_band_mask)


def test_validate_band_names():
    # Test case 1: Valid band names (list of strings)
    band_names_list = ["R", "G", "B"]
    mt_lime.validate_band_names(band_names_list)  # No exception should be raised

    # Test case 2: Valid band names (list of lists of strings)
    band_names_list_of_lists = [["R", "G"], ["B"]]
    mt_lime.validate_band_names(band_names_list_of_lists)  # No exception should be raised

    # Test case 3: Valid band names (dictionary with tuples as keys and integers as values)
    band_names_dict = {("R", "G"): 1, "B": 2}
    mt_lime.validate_band_names(band_names_dict)  # No exception should be raised

    # Test case 4: Invalid band names (wrong type)
    invalid_band_names = 123
    with pytest.raises(TypeError):
        mt_lime.validate_band_names(invalid_band_names)

    # Test case 5: Invalid band names (wrong type in list)
    invalid_band_names_list = ["R", 123, "B"]
    with pytest.raises(TypeError):
        mt_lime.validate_band_names(invalid_band_names_list)

    # Test case 6: Invalid band names (wrong type in list of lists)
    invalid_band_names_list_of_lists = [["R", "G"], ["B", 123]]
    with pytest.raises(TypeError):
        mt_lime.validate_band_names(invalid_band_names_list_of_lists)

    # Test case 7: Invalid band names (wrong type in dictionary keys)
    invalid_band_names_dict = {("R", 123): 1, "B": 2}
    with pytest.raises(TypeError):
        mt_lime.validate_band_names(invalid_band_names_dict)

    # Test case 8: Invalid band names (wrong type in dictionary values)
    invalid_band_names_dict = {("R", "G"): 1, "B": "2"}
    with pytest.raises(TypeError):
        mt_lime.validate_band_names(invalid_band_names_dict)


def test_validate_band_format():
    dict_band_names = "test"
    # Test case 1: Valid band ranges (tuple)
    band_ranges_tuple = (400, 700)
    mt_lime.validate_band_format({dict_band_names: band_ranges_tuple}, variable_name="test")

    band_ranges_tuple = (400.0, 700.0)
    mt_lime.validate_band_format({dict_band_names: band_ranges_tuple}, variable_name="test")

    dict_band_names_tuple = ("test1", "test2")
    band_ranges_tuple = (400, 700)
    mt_lime.validate_band_format({dict_band_names_tuple: band_ranges_tuple}, variable_name="test")

    # Test case 2: Valid band ranges (list of tuples)
    band_ranges_list = [(400, 500), (600, 700)]
    mt_lime.validate_band_format({dict_band_names: band_ranges_list}, variable_name="test")

    band_ranges_list = [(400.0, 500.0), (600.0, 700.0)]
    mt_lime.validate_band_format({dict_band_names: band_ranges_list}, variable_name="test")

    # Test case 3: Valid band value (int)
    band_value = 400
    mt_lime.validate_band_format({dict_band_names: band_value}, variable_name="test")

    band_value = 400.0
    mt_lime.validate_band_format({dict_band_names: band_value}, variable_name="test")

    # Test case 4: Valid band list (list of ints)
    band_list = [400, 500, 600]
    mt_lime.validate_band_format({dict_band_names: band_list}, variable_name="test")

    band_list = [400.0, 500.0, 600.0]
    mt_lime.validate_band_format({dict_band_names: band_list}, variable_name="test")

    # Test case 5: Invalid band ranges (wrong type)
    invalid_band_ranges = "invalid"
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value)

    # Test case 6: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400, 500, 600)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value)

    # Test case 7: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400, 500), 600]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value)

    # Test case 8: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400,)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value)

    # Test case 9: Different variable name
    invalid_band_ranges = [(400,)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format(
            {dict_band_names: invalid_band_ranges}, variable_name="pizza with a pineapple is not a pizza"
        )
    assert "pizza with a pineapple is not a pizza" in str(excinfo.value)
    assert "test" not in str(excinfo.value)
    # Test case 10: Invalid keys
    dict_band_names = 123
    band_value = 400
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: band_value}, variable_name="test")
    assert "keys should be string or tuple of strings" in str(excinfo.value)

    # Test case 11: Invalid keys tuple
    dict_band_names_tuple = (123, "test2")
    band_value = 400
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_format({dict_band_names: band_value}, variable_name="test")
    assert "keys should be string or tuple of strings" in str(excinfo.value)


def test_validate_segment_format():
    # Test case 1: Valid segment range (tuple of ints)
    segment_range_tuple = (0, 10)
    result_tuple = mt_lime.validate_segment_format(segment_range_tuple)
    assert isinstance(result_tuple, list)
    assert len(result_tuple) == 1
    assert result_tuple[0] == segment_range_tuple
    assert isinstance(result_tuple[0][0], int) and isinstance(result_tuple[0][1], int)

    # Test case 2: Valid segment range (list of tuples of ints)
    segment_range_list = [(0, 10), (20, 30)]
    result_list = mt_lime.validate_segment_format(segment_range_list)
    assert isinstance(result_list, list)
    assert len(result_list) == len(segment_range_list)
    assert all(
        result_tuple == segment_range_tuple
        for result_tuple, segment_range_tuple in zip(result_list, segment_range_list)
    )

    # Test case 3: Valid segment range (tuple of ints) of floats
    segment_range_tuple = (0.0, 10.0)
    result_tuple = mt_lime.validate_segment_format(segment_range_tuple, dtype=float)
    assert isinstance(result_tuple, list)
    assert len(result_tuple) == 1
    assert result_tuple[0] == segment_range_tuple
    assert isinstance(result_tuple[0][0], float) and isinstance(result_tuple[0][1], float)
    # Test case 4: Valid segment range (list of tuples of ints) of floats
    segment_range_list = [(0.0, 10.0), (20.0, 30.0)]
    result_list = mt_lime.validate_segment_format(segment_range_list, dtype=float)
    assert isinstance(result_list, list)
    assert len(result_list) == len(segment_range_list)
    assert all(
        result_tuple == segment_range_tuple
        for result_tuple, segment_range_tuple in zip(result_list, segment_range_list)
    )

    # Test case 5: Invalid segment range (wrong type)
    invalid_segment_range = "invalid"
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 6: Invalid segment range (wrong type)
    invalid_segment_range = 123
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 7: Invalid segment range (tuple of floats)
    invalid_segment_range = (0.5, 10.5)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 8: Invalid segment range (list of tuples of floats)
    invalid_segment_range = [(0.5, 10.5), (20.5, 30.5)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 9: Invalid segment range (tuple of ints with wrong order)
    invalid_segment_range = (10, 0)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 10: Invalid segment range (list of tuples of ints with wrong order)
    invalid_segment_range = [(10, 0), (30, 20)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range)

    # Test case 11: Invalid segment range (tuple of floats with wrong order)
    invalid_segment_range = (10.5, 0.5)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range, dtype=float)

    # Test case 12: Invalid segment range (list of tuples of floats with wrong order)
    invalid_segment_range = [(10.5, 0.5), (30.5, 20.5)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range, dtype=float)

    # Test case 13: Invalid segment dtype
    invalid_segment_range = (0, 10)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format(invalid_segment_range, dtype=float)


def test_adjust_and_validate_segment_ranges():
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])

    # Test case 1: Valid segment range
    segment_range = [(2, 5), (7, 9)]
    result = mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)
    assert result == [(2, 5), (7, 9)]

    # Test case 2: Segment range with end exceeding wavelengths max index
    segment_range = [(0, 3), (6, 10)]
    with pytest.warns(UserWarning):
        result = mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)
    assert result == [(0, 3), (6, 9)]

    # Test case 3: Segment range with negative start
    segment_range = [(-1, 4), (7, 9)]
    with pytest.warns(UserWarning):
        result = mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)
    assert result == [(0, 4), (7, 9)]

    # Test case 4: Not Valid segment range
    segment_range = [(-1, 0), (7, 9)]
    with pytest.raises(ValueError):
        mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)

    segment_range = [(2, 5), (9, 10)]
    with pytest.raises(ValueError):
        mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)


def test_validate_tensor():
    # Test with numpy array
    np_array = np.array([1, 2, 3])
    result = mt_lime.validate_tensor(np_array, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([1, 2, 3])))

    # Test with torch tensor
    torch_tensor = torch.tensor([4, 5, 6])
    result = mt_lime.validate_tensor(torch_tensor, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([4, 5, 6])))

    # Test with invalid input type
    with pytest.raises(TypeError):
        mt_lime.validate_tensor("invalid", "Input must be a numpy array or torch tensor")


def test_validate_segment_range():
    # Test case 1: Valid segment range within bounds
    wavelengths = torch.tensor([400, 500, 600, 700])
    segment_range = [(1, 2), (2, 3)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(1, 2), (2, 3)]

    # Test case 2: Valid segment range with adjustments
    wavelengths = torch.tensor([400, 500, 600, 700])
    segment_range = [(0, 3), (2, 5)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(0, 3), (2, 4)]

    # Test case 3: Valid segment range with adjustments
    wavelengths = torch.tensor([400, 500, 600, 700])
    segment_range = [(-1, 3), (2, 4)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(0, 3), (2, 4)]

    # Test case 4: Invalid segment range out of bounds
    wavelengths = torch.tensor([400, 500, 600, 700])
    segment_range = [(0, 5), (4, 8)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)

    # Test case 5: Invalid segment range out of bounds
    wavelengths = torch.tensor([400, 500, 600, 700])
    segment_range = [(-1, 0), (2, 4)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)


def test_resolve_inference_device():
    # Test device as string
    device = "cpu"
    info = ValidationInfoMock(data={"hsi": torch.randn(5, 5)})
    result = mt_lime.resolve_inference_device(device, info)
    assert isinstance(result, torch.device)
    assert str(result) == device

    # Test device as torch.device
    device = torch.device("cpu")
    result = mt_lime.resolve_inference_device(device, info)
    assert isinstance(result, torch.device)
    assert result == device

    # Test device as None
    device = None
    info = ValidationInfoMock(data={"hsi": torch.randn(5, 5)})
    result = mt_lime.resolve_inference_device(device, info)
    assert isinstance(result, torch.device)
    assert result == info.data["hsi"].device

    # Test invalid device type
    device = 123
    info = ValidationInfoMock(data={"hsi": torch.randn(5, 5)})
    with pytest.raises(ValueError):
        mt_lime.resolve_inference_device("device", info)

    # Test no image in the info
    device = None
    info = ValidationInfoMock(data={})
    with pytest.raises(ValueError):
        mt_lime.resolve_inference_device(device, info)

    # Test wrong type device
    device = 0
    info = ValidationInfoMock(data={"hsi": torch.randn(5, 5)})
    with pytest.raises(TypeError):
        mt_lime.resolve_inference_device(device, info)


######################################################################
############################ EXPLANATIONS ############################
######################################################################


def test_validate_hsi_attributions():
    # Create a sample HSIAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    device = torch.device("cpu")
    hsi_attributes = mt.HSIAttributes(hsi=hsi, attributes=attributes, score=score, device=device)

    # Assert that the attributes tensor has been moved to the specified device
    assert hsi_attributes.attributes.device == device

    # Assert that the hsi tensor has been moved to the specified device
    assert hsi_attributes.hsi.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert hsi_attributes.attributes.shape == hsi_attributes.hsi.image.shape

    # Assert that the device of the hsi object has been updated
    assert hsi_attributes.hsi.device == device

    # Assert that the mask is None
    assert hsi_attributes.mask is None

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.HSIAttributes(hsi=hsi, attributes=invalid_attributes, score=score, device=device)

    # Not implemented yet functions
    with pytest.raises(NotImplementedError):
        hsi_attributes.flattened_attributes

    # Validate mask
    mask = torch.randint(0, 2, (3, 4, 4))
    hsi_attributes = mt.HSIAttributes(hsi=hsi, attributes=attributes, score=score, device=device, mask=mask)

    # Assert that the mask is not None
    assert torch.equal(hsi_attributes.mask, mask)

    # Assert that the mask is moved to the specified device
    assert hsi_attributes.mask.device == device

    # Not Valid mask
    invalid_mask = torch.randint(0, 2, (1, 4, 4))
    with pytest.raises(ValueError):
        mt.HSIAttributes(hsi=hsi, attributes=attributes, score=score, device=device, mask=invalid_mask)


def test_spatial_attributes():
    # Create a sample HSISpatialAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    spatial_attributes = mt.HSISpatialAttributes(
        hsi=hsi, attributes=attributes, score=score, mask=segmentation_mask, device=device
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert spatial_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert spatial_attributes.hsi.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spatial_attributes.segmentation_mask.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert spatial_attributes.attributes.shape == spatial_attributes.hsi.image.shape

    # Assert that the device of the image object has been updated
    assert spatial_attributes.hsi.device == device

    # Assert that the mask is the same as the segmentation mask
    assert torch.equal(spatial_attributes.mask, segmentation_mask)

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.HSISpatialAttributes(
            hsi=hsi, attributes=invalid_attributes, score=score, mask=segmentation_mask, device=device
        )


def test_change_orientation_spatial_attributes():
    # Create a sample HSI and attributes
    hsi = mt.HSI(image=torch.randn(3, 4, 5), wavelengths=[400, 500, 600], orientation=("C", "H", "W"))
    assert hsi.orientation == ("C", "H", "W")

    attributes = torch.randn(3, 4, 5)
    mask = torch.randint(0, 2, (3, 4, 5))

    # Create HSISpatialAttributes object
    attrs = mt.HSISpatialAttributes(hsi=hsi, attributes=attributes, score=0.5, mask=mask)
    assert attrs.hsi.orientation == ("C", "H", "W")

    # Change orientation to ('H', 'W', 'C')
    new_orientation = ("H", "W", "C")
    attrs_changed = attrs.change_orientation(new_orientation, False)
    assert attrs.hsi.orientation != attrs_changed.hsi.orientation
    assert attrs_changed.hsi.orientation == new_orientation
    assert attrs_changed.hsi.image.shape == (4, 5, 3)
    assert attrs_changed.attributes.shape == (4, 5, 3)
    assert attrs_changed.mask.shape == (4, 5, 3)

    # change of orientation inplace
    new_orientation = ("H", "C", "W")
    attrs.change_orientation(new_orientation, True)
    assert attrs.hsi.orientation == new_orientation
    assert attrs.hsi.orientation == new_orientation
    assert attrs.hsi.image.shape == (4, 3, 5)
    assert attrs.attributes.shape == (4, 3, 5)
    assert attrs.mask.shape == (4, 3, 5)

    # test the case where the orientation is the same
    new_new_orientation = ("H", "C", "W")
    attrs.change_orientation(new_new_orientation, True)
    assert new_new_orientation == new_orientation
    assert attrs.hsi.orientation == new_new_orientation
    assert attrs.hsi.orientation == new_new_orientation
    assert attrs.hsi.image.shape == (4, 3, 5)
    assert attrs.attributes.shape == (4, 3, 5)
    assert attrs.mask.shape == (4, 3, 5)

    # test case with invalid orientation
    new_orientation = ("H", "C", "A")
    with pytest.raises(ValueError):
        attrs.change_orientation(new_orientation, True)


def test_to_hsi_spatial_attributes():
    # Create dummy data
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))

    # Create HSISpatialAttributes object
    spatial_attributes = mt.HSISpatialAttributes(
        hsi=hsi,
        attributes=attributes,
        score=0.8,
        mask=segmentation_mask,
    )

    # Move to a different device
    device = torch.device("cpu")
    spatial_attributes.to(device)

    # Check if the hsi, attributes, and segmentation mask are moved to the new device
    assert spatial_attributes.hsi.device == device
    assert spatial_attributes.attributes.device == device
    assert spatial_attributes.segmentation_mask.device == device

    if torch.cuda.is_available():
        # Move back to the original device
        device = torch.device("cuda")
        spatial_attributes.to(device)

        # Check if the hsi, attributes, and segmentation mask are moved to the original device
        assert spatial_attributes.hsi.device == device
        assert spatial_attributes.attributes.device == device
        assert spatial_attributes.segmentation_mask.device == device


def test_spatial_segmentation_mask_spacial_attributes():
    # Create a sample HSISpatialAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones_like(hsi.image)
    score = 0.8
    segmentation_mask_like_img = torch.randint(0, 2, attributes.shape)
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt.HSISpatialAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        mask=segmentation_mask_like_img,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask_like_img[0])

    spatial_attributes = mt.HSISpatialAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        device=device,
        model_config=model_config,
    )

    with pytest.raises(ValueError):
        spatial_attributes.segmentation_mask


def test_flattened_attributes_spacial_attributes():
    # Create a sample HSISpatialAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones_like(hsi.image)
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (attributes.shape[1:]))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt.HSISpatialAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spatial_attributes.flattened_attributes, attributes[0])


def test_spectral_attributes():
    # Create a sample HSISpectralAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.zeros_like(attributes)
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi, attributes=attributes, score=score, band_names=band_names, mask=band_mask, device=device
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert spectral_attributes.attributes.device == device

    # Assert that the hsi tensor has been moved to the specified device
    assert spectral_attributes.hsi.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spectral_attributes.band_mask.device == device

    # Assert that the shapes of the attributes and hsi tensors match
    assert spectral_attributes.attributes.shape == spectral_attributes.hsi.image.shape

    # Assert that the device of the hsi object has been updated
    assert spectral_attributes.hsi.device == device

    # Assert that the mask is the same as the band mask
    assert torch.equal(spectral_attributes.mask, band_mask)

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.HSISpectralAttributes(
            hsi=hsi, attributes=invalid_attributes, score=score, band_names=band_names, mask=band_mask, device=device
        )

    # Add `_not_included`
    band_mask = torch.zeros_like(attributes)
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"G": 1, "B": 2}

    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi, attributes=attributes, score=score, band_names=band_names, mask=band_mask, device=device
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert spectral_attributes.attributes.device == device

    # Assert that the hsi tensor has been moved to the specified device
    assert spectral_attributes.hsi.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spectral_attributes.band_mask.device == device

    # Assert that the shapes of the attributes and hsi tensors match
    assert spectral_attributes.attributes.shape == spectral_attributes.hsi.image.shape

    # Assert that the device of the hsi object has been updated
    assert spectral_attributes.hsi.device == device

    # Assert that the mask is the same as the band mask
    assert torch.equal(spectral_attributes.mask, band_mask)

    # Assert `not_included` band name
    assert spectral_attributes.band_names == {
        "not_included": 0,
        "G": 1,
        "B": 2,
    }

    # Test `not_included` added but there is not covered ids
    band_mask = torch.zeros_like(attributes)
    band_mask[1] = 1
    band_mask[2] = 2
    band_mask[2, 0] = 3
    band_names = {"G": 1, "B": 2, "not_included": 3}

    with pytest.raises(ValidationError):
        mt.HSISpectralAttributes(
            hsi=hsi, attributes=attributes, score=score, band_names=band_names, mask=band_mask, device=device
        )


def test_to_hsi_spectral_attributes():
    # Create dummy data
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create HSISpectralAttributes object
    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi,
        attributes=attributes,
        score=0.8,
        mask=band_mask,
        band_names=band_names,
    )

    # Move to a different device
    device = torch.device("cpu")
    spectral_attributes.to(device)

    # Check if the hsi, attributes, band mask, and band names are moved to the new device
    assert spectral_attributes.hsi.device == device
    assert spectral_attributes.attributes.device == device
    assert spectral_attributes.band_mask.device == device

    if torch.cuda.is_available():
        # Move back to the original device
        device = torch.device("cuda")
        spectral_attributes.to(device)

        # Check if the hsi, attributes, and segmentation mask are moved to the original device
        assert spectral_attributes.hsi.device == device
        assert spectral_attributes.attributes.device == device
        assert spectral_attributes.band_mask.device == device


def test_band_mask_spectral_attributes():
    # Create a sample HSISpectralAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        band_names=band_names,
        mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])

    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        band_names=band_names,
        device=device,
        model_config=model_config,
    )
    with pytest.raises(ValueError):
        spectral_attributes.band_mask


def test_flattened_attributes_spectral_attributes():
    # Create a sample HSISpectralAttributes object
    hsi = mt.HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spectral_attributes = mt.HSISpectralAttributes(
        hsi=hsi,
        attributes=attributes,
        score=score,
        band_names=band_names,
        mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spectral_attributes.flattened_attributes, torch.ones((band_mask.shape[0],)))


###################################################################
############################ EXPLAINER ############################
###################################################################


def test_explainer():
    # Create mock objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")
    interpretable_model = SkLearnLasso()

    # Create a Explainer object
    explainer = mt_lime.Explainer(explainable_model, interpretable_model)

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert explainer.explainable_model == explainable_model
    assert explainer.interpretable_model == interpretable_model

    # Test case 1: Valid input
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    explainable_model = ExplainableModel(forward_func=dumb_model, problem_type="regression")
    interpretable_model = SkLearnLasso(alpha=0.1)
    # Create a sample Lime object
    lime = mt_lime.Explainer(explainable_model, interpretable_model)

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert lime.explainable_model == explainable_model
    assert lime.interpretable_model == interpretable_model

    # Test case 2: different device
    device = torch.device("cpu")
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")
    interpretable_model = SkLearnLasso()
    explainer = mt_lime.Explainer(explainable_model, interpretable_model)

    # Call the to method
    explainer.to(device)


def test_lime_explainer():
    # Create mock objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")
    interpretable_model = SkLearnLasso()

    # Create a Lime object
    lime = mt_lime.Lime(explainable_model, interpretable_model)

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert lime.explainable_model == explainable_model
    assert lime.interpretable_model == interpretable_model
    assert lime._lime is not None

    # Test case 1: Valid input
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    explainable_model = ExplainableModel(forward_func=dumb_model, problem_type="regression")
    interpretable_model = SkLearnLasso(alpha=0.1)
    # Create a sample Lime object
    lime = mt_lime.Lime(explainable_model, interpretable_model, None)

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert lime.explainable_model == explainable_model
    assert lime.interpretable_model == interpretable_model
    assert lime._lime is not None

    # Test case 2: different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    explainable_model = ExplainableModel(forward_func=dumb_model, problem_type="regression")
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=explainable_model,
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert lime.explainable_model == explainable_model
    assert lime.interpretable_model == interpretable_model
    assert lime._lime is not None


def test__get_slick_segmentation_mask():
    # Create a sample hsi
    hsi = mt.HSI(image=torch.randn(3, 240, 240), wavelengths=[462.08, 465.27, 468.47])

    # Call the method
    segmentation_mask = mt_lime.Lime._get_slick_segmentation_mask(hsi, num_interpret_features=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert torch.all(segmentation_mask >= 0)
    assert torch.all(segmentation_mask < 10)

    # Check mask slic
    mask = torch.ones_like(hsi.image)
    mask[0, 10, 10] = 0
    mask[0, 20, 20] = 0

    hsi = mt.HSI(image=torch.randn(3, 240, 240), wavelengths=[462.08, 465.27, 468.47], binary_mask=mask)
    segmentation_mask = mt_lime.Lime._get_slick_segmentation_mask(hsi, num_interpret_features=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert torch.all(segmentation_mask >= 0)
    assert torch.all(segmentation_mask < 10)
    assert segmentation_mask[0, 10, 10] == 0
    assert segmentation_mask[0, 20, 20] == 0


def test__get_patch_segmentation_mask():
    # Create a sample hsi
    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])

    # Call the _get_patch_segmentation_mask method
    patch_size = 10
    segmentation_mask = mt_lime.Lime._get_patch_segmentation_mask(hsi, patch_size=patch_size)

    # Check the shape of the segmentation mask
    assert segmentation_mask.shape == (1, 240, 240)

    # Check the unique values in the segmentation mask
    unique_values = torch.unique(segmentation_mask)
    expected_number_of_elements = (240 // patch_size) * (240 // patch_size)
    assert unique_values.numel() == expected_number_of_elements

    # Check that the segmentation mask is created correctly
    for i in range(1, unique_values.numel()):
        mask_value = unique_values[i]
        mask_indices = torch.nonzero(segmentation_mask == mask_value)
        assert mask_indices.shape[0] == patch_size * patch_size

        # Check that the mask indices are within the hsi dimensions
        assert torch.all(mask_indices[:, 1] < 240)
        assert torch.all(mask_indices[:, 2] < 240)

    # Test case 2: Invalid patch size
    patch_size = 0
    with pytest.raises(ValueError):
        mt_lime.Lime._get_patch_segmentation_mask(hsi, patch_size=patch_size)

    # Test case 3: Invalid patch size
    patch_size = 241
    with pytest.raises(ValueError):
        mt_lime.Lime._get_patch_segmentation_mask(hsi, patch_size=patch_size)

    # Test case 4: Mask
    mask = torch.ones_like(hsi.image)
    mask[0, 10, 10] = 0
    mask[0, 20, 20] = 0

    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47], binary_mask=mask)
    segmentation_mask = mt_lime.Lime._get_patch_segmentation_mask(hsi, patch_size=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert segmentation_mask[0, 10, 10] == 0
    assert segmentation_mask[0, 20, 20] == 0


def test_get_segmentation_mask():
    # Test case 1: Valid segmentation method (slic)
    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="slic")
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)

    # Test case 2: Valid segmentation method (patch)
    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="patch")
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)

    # Test case 3: Invalid segmentation method
    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    with pytest.raises(ValueError):
        mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="invalid")

    # Test case 4: Invalid segmentation hsi
    hsi = torch.ones((3, 240, 240))
    with pytest.raises(ValueError):
        mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="slic")


def test__make_band_names_indexable():
    # Test case 1: List of strings
    segment_name_list = ["R", "G", "B"]
    result_list = mt_lime.Lime._make_band_names_indexable(segment_name_list)
    assert isinstance(result_list, tuple)
    assert result_list == ("R", "G", "B")

    # Test case 2: Tuple of strings
    segment_name_tuple = ("R", "G", "B")
    result_tuple = mt_lime.Lime._make_band_names_indexable(segment_name_tuple)
    assert isinstance(result_tuple, tuple)
    assert result_tuple == ("R", "G", "B")

    # Test case 3: String
    segment_name_string = "R"
    result_string = mt_lime.Lime._make_band_names_indexable(segment_name_string)
    assert isinstance(result_string, str)
    assert result_string == "R"

    # Test case 4: Incorrect segment name type
    segment_name_invalid = 123
    with pytest.raises(ValueError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)

    # Test case 5: Incorrect segment name type
    segment_name_invalid = [123]
    with pytest.raises(ValueError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)

    # Test case 6: Incorrect segment name type
    segment_name_invalid = (123,)
    with pytest.raises(ValueError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)


def test__extract_bands_from_spyndex():
    # Test case 1: Single band name
    segment_name = "R"
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert result == "R"

    # Test case 2: Multiple band names
    segment_name = ["R", "G"]
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert isinstance(result, tuple)
    assert list(result).sort() == ["R", "G"].sort()

    # Test case 3: Invalid band name
    segment_name = "D"
    with pytest.raises(ValueError):
        mt_lime.Lime._extract_bands_from_spyndex(segment_name)

    # Test case 4: Spatial Index
    segment_name = "AVI"
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert isinstance(result, tuple)
    assert list(result).sort() == ["N", "R"].sort()
    # Test case 5: Multiple spatial indices
    segment_name = ["AVI", "AVI"]
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert isinstance(result, tuple)
    assert list(result).sort() == ["N", "R"].sort()

    # Test case 5: Invalid spatial index
    segment_name = "AV"
    with pytest.raises(ValueError):
        mt_lime.Lime._extract_bands_from_spyndex(segment_name)

    # Test case 6: band name with spatial index
    segment_name = ["G", "AVI"]
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert isinstance(result, tuple)
    assert list(result).sort() == ["G", "N", "R"].sort()


def test__convert_wavelengths_to_indices():
    wavelengths = torch.tensor([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
    ranges = [(450.0, 550.0), (600.0, 700.0)]

    expected_indices = [(1, 4), (4, 7)]
    indices = mt_lime.Lime._convert_wavelengths_to_indices(wavelengths, ranges)

    assert indices == expected_indices

    # Test case with single range
    single_range = (500.0, 650.0)

    expected_single_indices = [(2, 6)]
    single_indices = mt_lime.Lime._convert_wavelengths_to_indices(wavelengths, single_range)

    assert single_indices == expected_single_indices

    # Test case with empty range
    empty_range = (550.0, 550.0)

    expected_empty_indices = [(3, 4)]
    empty_indices = mt_lime.Lime._convert_wavelengths_to_indices(wavelengths, empty_range)

    assert empty_indices == expected_empty_indices

    # Test case with range out of bounds
    empty_range = (720.0, 750.0)

    expected_single_indices = [(7, 7)]
    single_indices = mt_lime.Lime._convert_wavelengths_to_indices(wavelengths, empty_range)

    assert single_indices == expected_single_indices

    # Test case with negative range out of bounds
    empty_range = (350.0, 370.0)

    expected_single_indices = [(0, 0)]
    single_indices = mt_lime.Lime._convert_wavelengths_to_indices(wavelengths, empty_range)

    assert single_indices == expected_single_indices


def test__get_indices_from_wavelength_indices_range():
    # Test case 1: Single range
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [(1, 4)]
    expected_indices = [1, 2, 3]
    indices = mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)
    assert indices == expected_indices

    # Test case 2: Multiple ranges
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [(1, 3), (4, 6)]
    expected_indices = [1, 2, 4, 5]
    indices = mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)
    assert indices == expected_indices

    # Test case 3: Empty range
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = []
    expected_indices = []
    indices = mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)
    assert indices == expected_indices

    # Test case 4: Range with single index
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [(3, 3)]
    with pytest.raises(ValueError):
        mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)

    # Test case 5: Range with duplicate indices
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [(2, 4), (4, 6)]
    expected_indices = [2, 3, 4, 5]
    indices = mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)
    assert indices == expected_indices

    # Test case 6: Range with overlapping indices
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [(2, 4), (3, 6)]
    expected_indices = [2, 3, 4, 5]
    indices = mt_lime.Lime._get_indices_from_wavelength_indices_range(wavelengths, ranges)
    assert indices == expected_indices


def test__convert_wavelengths_list_to_indices():
    # Working example
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [450, 550, 650]
    expected_indices = [1, 3, 5]

    indices = mt_lime.Lime._convert_wavelengths_list_to_indices(wavelengths, ranges)
    assert indices == expected_indices

    # Test case with empty range
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = []
    expected_indices = []

    indices = mt_lime.Lime._convert_wavelengths_list_to_indices(wavelengths, ranges)
    assert indices == expected_indices

    # Test case with multiple ranges
    wavelengths = torch.tensor([400, 450, 450, 500, 550, 600, 650, 700])
    ranges = [450, 550, 700]

    with pytest.raises(ValueError):
        mt_lime.Lime._convert_wavelengths_list_to_indices(wavelengths, ranges)

    # Test case with not overalapping indices
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    ranges = [800, 900, 1000]

    with pytest.raises(ValueError):
        mt_lime.Lime._convert_wavelengths_list_to_indices(wavelengths, ranges)


def test__get_band_indices_from_input_band_indices():
    # Test lists
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_indices = {
        "segment1": [1, 3],
        "segment2": [4, 5],
        "segment3": 2,
    }

    expected_result = {
        "segment1": [1, 3],
        "segment2": [4, 5],
        "segment3": [2],
    }

    result = mt_lime.Lime._get_band_indices_from_input_band_indices(wavelengths, band_indices)
    assert result == expected_result

    # Test tuples
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_indices = {
        "segment1": (1, 3),
        "segment2": (4, 5),
        "segment3": [(2, 3), (4, 5)],
    }

    expected_result = {
        "segment1": [1, 2],
        "segment2": [4],
        "segment3": [2, 4],
    }

    result = mt_lime.Lime._get_band_indices_from_input_band_indices(wavelengths, band_indices)
    assert result == expected_result

    # Test single value
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_indices = {
        "segment1": 1,
        "segment2": 4,
    }

    expected_result = {
        "segment1": [1],
        "segment2": [4],
    }

    result = mt_lime.Lime._get_band_indices_from_input_band_indices(wavelengths, band_indices)
    assert result == expected_result

    # Test mix
    band_indices = {
        "segment1": 1,
        "segment2": (4, 5),
        "segment3": [2, 3],
        "segment4": [(4, 5), (5, 6)],
    }

    expected_result = {
        "segment1": [1],
        "segment2": [4],
        "segment3": [2, 3],
        "segment4": [4, 5],
    }
    result = mt_lime.Lime._get_band_indices_from_input_band_indices(wavelengths, band_indices)
    assert result == expected_result

    # Test invalid band_ranges_indices
    band_indices = 123
    with pytest.raises(ValueError):
        mt_lime.Lime._get_band_indices_from_input_band_indices(wavelengths, band_indices)


def test__get_band_indices_from_band_wavelengths():
    # Test case 1: Ranges
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_wavelengths = {
        "segment1": (450, 550),
        "segment2": [(400, 500), (600, 700)],
    }

    expected_band_indices = {
        "segment1": [1, 2, 3],
        "segment2": [0, 1, 2, 4, 5, 6],
    }

    band_indices = mt_lime.Lime._get_band_indices_from_band_wavelengths(wavelengths, band_wavelengths)
    assert band_indices == expected_band_indices

    # Test case 2: Lists
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_wavelengths = {
        "segment1": [450, 550],
        "segment2": [400, 500],
    }

    expected_band_indices = {
        "segment1": [1, 3],
        "segment2": [0, 2],
    }

    band_indices = mt_lime.Lime._get_band_indices_from_band_wavelengths(wavelengths, band_wavelengths)
    assert band_indices == expected_band_indices

    # Test case 3: Single value
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_wavelengths = {
        "segment1": 450,
        "segment2": 500,
    }

    expected_band_indices = {
        "segment1": [1],
        "segment2": [2],
    }

    band_indices = mt_lime.Lime._get_band_indices_from_band_wavelengths(wavelengths, band_wavelengths)
    assert band_indices == expected_band_indices

    # Test case 4: Mixed types
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])

    band_wavelengths = {
        "segment1": 450,
        "segment2": [400, 500],
        "segment3": (600, 700),
        "segment4": [(500, 600), (600, 700)],
    }

    expected_band_indices = {
        "segment1": [1],
        "segment2": [0, 2],
        "segment3": [4, 5, 6],
        "segment4": [2, 3, 4, 5, 6],
    }

    band_indices = mt_lime.Lime._get_band_indices_from_band_wavelengths(wavelengths, band_wavelengths)
    assert band_indices == expected_band_indices

    # Test invalid band_ranges_wavelengths
    band_wavelengths = 123
    with pytest.raises(ValueError):
        mt_lime.Lime._get_band_indices_from_band_wavelengths(wavelengths, band_wavelengths)


def test__get_band_wavelengths_indices_from_band_names():
    # Test bands
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])
    band_names = ["R", "G", "B"]
    band_indices, dict_labels_to_segment_ids = mt_lime.Lime._get_band_wavelengths_indices_from_band_names(
        wavelengths, band_names
    )

    expected_band_indices = {
        "R": [5],
        "G": [3, 4],
        "B": [1, 2],
    }
    expected_dict_labels_to_segment_ids = {
        "R": 1,
        "G": 2,
        "B": 3,
    }

    assert band_indices == expected_band_indices
    assert dict_labels_to_segment_ids == expected_dict_labels_to_segment_ids

    # Test indices
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])
    band_names = ["AVI"]
    band_indices, dict_labels_to_segment_ids = mt_lime.Lime._get_band_wavelengths_indices_from_band_names(
        wavelengths, band_names
    )

    expected_band_indices = {
        ("AVI"): [8, 5],
    }

    expected_dict_labels_to_segment_ids = {
        ("AVI"): 1,
    }

    # Test string
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])
    band_names = "AVI"
    band_indices, dict_labels_to_segment_ids = mt_lime.Lime._get_band_wavelengths_indices_from_band_names(
        wavelengths, band_names
    )

    expected_band_indices = {
        ("AVI"): [8, 5],
    }

    expected_dict_labels_to_segment_ids = {
        ("AVI"): 1,
    }

    assert band_indices == expected_band_indices
    assert dict_labels_to_segment_ids == expected_dict_labels_to_segment_ids

    # Test bands and indices
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])
    band_names = ["G", "AVI"]
    band_indices, dict_labels_to_segment_ids = mt_lime.Lime._get_band_wavelengths_indices_from_band_names(
        wavelengths, band_names
    )

    expected_band_indices = {
        "G": [3, 4],
        "AVI": [8, 5],
    }

    expected_dict_labels_to_segment_ids = {
        "G": 1,
        "AVI": 2,
    }

    assert band_indices == expected_band_indices
    assert dict_labels_to_segment_ids == expected_dict_labels_to_segment_ids

    # Test bands out of bounds
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_names = ["G", "AVI"]
    with pytest.raises(ValueError):
        mt_lime.Lime._get_band_wavelengths_indices_from_band_names(wavelengths, band_names)

    # Test invialid band names
    band_names = 123
    with pytest.raises(ValueError):
        mt_lime.Lime._get_band_wavelengths_indices_from_band_names(wavelengths, band_names)


def test__check_overlapping_segments(caplog):
    # Create a sample hsi
    wavelengths = torch.tensor([400, 500, 600, 700])
    hsi = mt.HSI(image=torch.ones((4, 4, 4)), wavelengths=wavelengths)

    # Create a sample dictionary mapping segment labels to indices
    dict_labels_to_indices = {
        "segment1": [0, 1],
        "segment2": [1, 2],
        "segment3": [2, 3],
    }

    # modify the loguru warnings to be captured by pytest
    def custom_sink(message):
        warnings.warn(message, UserWarning)

    logger.add(custom_sink, level="WARNING")

    with pytest.warns(UserWarning):
        mt_lime.Lime._check_overlapping_segments(hsi, dict_labels_to_indices)

    non_overlapping_dict_labels_to_indices = {
        "segment1": [0, 1],
        "segment2": [2, 3],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mt_lime.Lime._check_overlapping_segments(hsi, non_overlapping_dict_labels_to_indices)

    logger.remove()
    # Create a sample dictionary with index out of range segments
    dict_labels_to_indices = {
        "segment1": [0, 1],
        "segment2": [1, 2],
        "segment3": [2, 3],
        "segment4": [3, 4],
    }

    with pytest.raises(IndexError):
        mt_lime.Lime._check_overlapping_segments(hsi, dict_labels_to_indices)


def test__validate_and_create_dict_labels_to_segment_ids():
    # Test case 1: Existing mapping is None
    dict_labels_to_segment_ids = None
    segment_labels = ["segment1", "segment2", "segment3"]

    result = mt_lime.Lime._validate_and_create_dict_labels_to_segment_ids(dict_labels_to_segment_ids, segment_labels)

    expected_result = {
        "segment1": 1,
        "segment2": 2,
        "segment3": 3,
    }
    assert result == expected_result

    # Test case 2: Existing mapping is valid
    dict_labels_to_segment_ids = {
        "segment1": 1,
        "segment2": 2,
        "segment3": 3,
    }
    segment_labels = ["segment1", "segment2", "segment3"]

    result = mt_lime.Lime._validate_and_create_dict_labels_to_segment_ids(dict_labels_to_segment_ids, segment_labels)

    assert result == dict_labels_to_segment_ids

    # Test case 3: Existing mapping length mismatch
    dict_labels_to_segment_ids = {
        "segment1": 1,
        "segment2": 2,
    }
    segment_labels = ["segment1", "segment2", "segment3"]

    with pytest.raises(ValueError):
        mt_lime.Lime._validate_and_create_dict_labels_to_segment_ids(dict_labels_to_segment_ids, segment_labels)

    # Test case 4: Non-unique segment IDs
    dict_labels_to_segment_ids = {
        "segment1": 1,
        "segment2": 1,
        "segment3": 3,
    }
    segment_labels = ["segment1", "segment2", "segment3"]

    with pytest.raises(ValueError):
        mt_lime.Lime._validate_and_create_dict_labels_to_segment_ids(dict_labels_to_segment_ids, segment_labels)


def test__create_single_dim_band_mask():
    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 3, 3)), wavelengths=wavelengths)

    # Create sample segment labels and indices
    dict_labels_to_indices = {
        "segment1": [0, 2, 4],
        "segment2": [1, 3, 5],
    }

    # Create sample segment IDs
    dict_labels_to_segment_ids = {
        "segment1": 1,
        "segment2": 2,
    }

    # Create the expected band mask
    expected_band_mask = torch.tensor([1, 2, 1, 2, 1, 2, 0])

    # Call the method
    band_mask = mt_lime.Lime._create_single_dim_band_mask(
        hsi, dict_labels_to_indices, dict_labels_to_segment_ids, "cpu"
    )

    # Check if the band mask matches the expected result
    assert torch.all(torch.eq(band_mask, expected_band_mask))

    # Not valid segment IDs
    dict_labels_to_indices = {
        "segment1": [0, 2, 4, 10],
        "segment2": [1, 3, 5],
    }

    with pytest.raises(ValueError):
        mt_lime.Lime._create_single_dim_band_mask(hsi, dict_labels_to_indices, dict_labels_to_segment_ids, "cpu")


def test__create_tensor_band_mask():
    # Test case 1: Default parameters
    hsi = mt.HSI(image=torch.ones((3, 240, 240)), wavelengths=[400, 500, 600])
    dict_labels_to_indices = {"label1": [0], "label2": [1, 2]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime._create_tensor_band_mask(hsi, dict_labels_to_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (3, 1, 1)
    assert torch.equal(band_mask, torch.tensor([[[1]], [[2]], [[2]]]))
    assert dict_labels_to_segment_ids == {"label1": 1, "label2": 2}

    # Test case 2: With dict_labels_to_segment_ids
    dict_labels_to_segment_ids = {"label1": 1, "label2": 2}
    band_mask, dict_labels_to_segment_ids_out = mt_lime.Lime._create_tensor_band_mask(
        hsi, dict_labels_to_indices, dict_labels_to_segment_ids=dict_labels_to_segment_ids
    )
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (3, 1, 1)
    assert torch.equal(band_mask, torch.tensor([[[1]], [[2]], [[2]]]))
    assert dict_labels_to_segment_ids_out == dict_labels_to_segment_ids

    # Test case 3: With repeat_dimensions
    device = torch.device("cpu")
    band_mask, dict_labels_to_segment_ids_out = mt_lime.Lime._create_tensor_band_mask(
        hsi, dict_labels_to_indices, device=device, repeat_dimensions=True
    )
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (3, 240, 240)
    assert torch.equal(band_mask[:, 0, 0], torch.tensor([1, 2, 2]))
    assert torch.equal(band_mask[:, 10, 100], torch.tensor([1, 2, 2]))
    assert band_mask.device == device
    assert dict_labels_to_segment_ids == dict_labels_to_segment_ids_out

    # Test case 4: With return_dict_labels_to_segment_ids
    band_mask = mt_lime.Lime._create_tensor_band_mask(
        hsi, dict_labels_to_indices, return_dict_labels_to_segment_ids=False
    )
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (3, 1, 1)
    assert torch.equal(band_mask, torch.tensor([[[1]], [[2]], [[2]]]))


def test_get_band_mask():
    # Test case 1: Valid input with band names
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["R", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"R": 1, "G": 2}

    # Test case 2: Valid input with band indices
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": 0}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 3: Valid input with band indices list
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": [0, 1, 2]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 4: Valid input with band ranges (indices)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_ranges_indices = {"RGB": [(0, 2)]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_ranges_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 5: Valid input with band wavelengths
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": 500.43}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 6: Valid input with band wavelengths list
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": [500.43, 554.78]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 7: Valid input with band ranges (wavelengths)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_ranges_wavelengths = {"RGB": [(400, 600)]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_ranges_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 8: Invalid input (no band names, groups, or ranges provided)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    with pytest.raises(AssertionError):
        mt_lime.Lime.get_band_mask(hsi)

    # Test case 9: Invalid input (incorrect band names)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["R", "G", "B", "Invalid"]
    with pytest.raises(ValueError):
        mt_lime.Lime.get_band_mask(hsi, band_names=band_names)

    # Test case 10: Invalid input (incorrect band ranges wavelengths)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": [(wavelengths[-1] + 10, wavelengths[-1] + 20)]}
    with pytest.raises(ValueError):
        mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)

    # Test case 11: Invalid input (incorrect band ranges indices)
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": [(len(wavelengths), len(wavelengths) + 1)]}
    with pytest.raises(ValueError):
        mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)

    # Test case 12: Invalid input image
    hsi = torch.ones((len(wavelengths), 10, 10))
    with pytest.raises(ValueError):
        mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)


def test_get_spatial_attributes_regression():
    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600])
    hsi = mt.HSI(image=torch.randn(5, 10, 10), wavelengths=wavelengths)

    # Create a sample segmentation mask
    segmentation_mask = torch.randint(1, 4, (1, 10, 10))
    segmentation_mask_smaller = segmentation_mask[0]

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Assert smaller segmentation mask
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask_smaller, target=0)
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=1)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="slic", target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="patch", target=0, patch_size=5)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 4: different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "regression"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 5: provide a custom segmentation postprocessing function
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)
    spatial_attributes = lime.get_spatial_attributes(
        hsi, segmentation_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape


def test_get_spatial_attributes_classification():
    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        X = torch.rand(image.shape[0], 2)
        output = torch.bernoulli(X)
        return output

    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600])
    hsi = mt.HSI(image=torch.randn(5, 10, 10), wavelengths=wavelengths)

    # Create a sample segmentation mask
    segmentation_mask = torch.randint(1, 4, (1, 10, 10))

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "classification"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=1)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="slic", target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="patch", target=0, patch_size=5)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 4: different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "classification"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 5: provide a custom segmentation postprocessing function
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)
    spatial_attributes = lime.get_spatial_attributes(
        hsi, segmentation_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape


def test_get_spatial_attributes_segmentation():
    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = mt.HSI(image=torch.randn(len(wavelengths), 10, 10), wavelengths=wavelengths)

    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(image)
        if len(image.shape) == 3:
            output[0:2] = 1
            output[0:2] = 2
        else:
            output[:, 0:2] = 1
            output[:, 0:2] = 2
        return output

    # Create a sample segmentation mask
    segmentation_mask = torch.randint(1, 4, (1, 10, 10))

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Get postprocessingagg_segmentation_postprocessing
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(
        hsi,
        segmentation_mask,
        target=0,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(
        hsi,
        segmentation_mask,
        target=1,
        postprocessing_segmentation_output=postprocessing,
    )
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(
        hsi,
        segmentation_method="slic",
        target=0,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(
        hsi,
        segmentation_method="patch",
        target=0,
        patch_size=5,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 4: different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(
        hsi,
        segmentation_mask,
        target=0,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test No segmentation postprocessing
    with pytest.raises(AssertionError):
        spatial_attributes = lime.get_spatial_attributes(
            hsi, segmentation_mask, target=0, postprocessing_segmentation_output=None
        )


def test_get_spectral_attributes_regression():
    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = mt.HSI(image=torch.randn(len(wavelengths), 240, 240), wavelengths=wavelengths)

    # Create a sample band mask
    band_mask = torch.zeros_like(hsi.image, dtype=int)
    band_mask[0] = 1
    band_mask[1] = 2

    band_mask_smaller = band_mask[:, 0, 0]

    # Create a sample band names dictionary
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Call the get_spectral_attributes method
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Assert smaller band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask_smaller, band_names=band_names, target=0)
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=1)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "regression"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 5: provide a custom segmentation postprocessing function
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)
    spectral_attributes = lime.get_spectral_attributes(
        hsi, band_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape


def test_get_spectral_attributes_classification():
    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        X = torch.rand(image.shape[0], 2)
        output = torch.bernoulli(X)
        return output

    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = mt.HSI(image=torch.randn(len(wavelengths), 240, 240), wavelengths=wavelengths)

    # Create a sample band mask
    band_mask = torch.zeros_like(hsi.image, dtype=int)
    band_mask[0] = 1
    band_mask[1] = 2

    # Create a sample band names dictionary
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "classification"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Call the get_spectral_attributes method
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=1)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "classification"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 5: provide a custom segmentation postprocessing function
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)
    spectral_attributes = lime.get_spectral_attributes(
        hsi, band_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape


def test_get_spectral_attributes_segmentation():
    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = mt.HSI(image=torch.randn(len(wavelengths), 10, 10), wavelengths=wavelengths)

    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(image)
        if len(image.shape) == 3:
            output[:, 0:5] = 1
            output[:, 0:5] = 2
        else:
            output[:, :, 0:5] = 1
            output[:, :, 0:5] = 2
        return output

    # Create a sample band mask
    band_mask = torch.zeros_like(hsi.image, dtype=int)
    band_mask[0] = 1
    band_mask[1] = 2

    # Create a sample band names dictionary
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Get postprocessing
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)

    # Call the get_spectral_attributes method
    spectral_attributes = lime.get_spectral_attributes(
        hsi,
        band_mask,
        band_names=band_names,
        target=0,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(
        hsi,
        band_mask,
        band_names=band_names,
        target=1,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(
        hsi,
        band_names=band_names,
        target=0,
        postprocessing_segmentation_output=postprocessing,
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(
        hsi, band_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation"),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(
        hsi, band_mask, target=0, postprocessing_segmentation_output=postprocessing
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 5: No segmentation postprocessing
    with pytest.raises(AssertionError):
        spectral_attributes = lime.get_spectral_attributes(
            hsi,
            band_mask,
            band_names=band_names,
            target=0,
            postprocessing_segmentation_output=None,
        )
