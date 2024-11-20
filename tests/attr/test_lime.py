import pytest

import torch


import meteors as mt
import meteors.attr.lime as mt_lime
import meteors.attr.lime_base as mt_lime_base
from meteors.models import ExplainableModel, SkLearnLasso
from meteors.utils import agg_segmentation_postprocessing

from meteors import HSI
from meteors.attr import HSISpatialAttributes, HSISpectralAttributes
from meteors.exceptions import ShapeMismatchError, MaskCreationError, BandSelectionError

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
    result = mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)
    assert result == [(0, 3), (6, 9)]

    # Test case 3: Segment range with negative start
    segment_range = [(-1, 4), (7, 9)]
    result = mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)
    assert result == [(0, 4), (7, 9)]

    # Test case 4: Not Valid segment range
    segment_range = [(-1, 0), (7, 9)]
    with pytest.raises(ValueError):
        mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)

    segment_range = [(2, 5), (9, 10)]
    with pytest.raises(ValueError):
        mt_lime.adjust_and_validate_segment_ranges(wavelengths, segment_range)


def test_validate_mask_shape():
    # validation of the function

    hsi = HSI(image=torch.randn(3, 240, 240), wavelengths=[462.08, 465.27, 468.47])
    segmentation_mask = torch.randint(0, 3, (1, 240, 240))

    validated_segmentation_mask = mt_lime.validate_mask_shape("segmentation", hsi=hsi, mask=segmentation_mask)

    assert validated_segmentation_mask.shape == (3, 240, 240)
    assert torch.all(validated_segmentation_mask[0] == segmentation_mask)
    assert torch.all(validated_segmentation_mask[1] == segmentation_mask)
    assert torch.all(validated_segmentation_mask[2] == segmentation_mask)

    band_mask = torch.randint(0, 3, (3, 1, 1))

    validated_band_mask = mt_lime.validate_mask_shape("band", hsi=hsi, mask=band_mask)

    assert validated_band_mask.shape == (3, 240, 240)
    assert torch.all(validated_band_mask[:, 0, 0] == band_mask.squeeze(-1).squeeze(-1))
    for i in range(240):
        for j in range(240):
            assert torch.all(validated_band_mask[:, i, j] == band_mask.squeeze(-1).squeeze(-1))

    # incorrect mask type
    with pytest.raises(ValueError):
        mt_lime.validate_mask_shape("incorrect", hsi=hsi, mask=band_mask)

    # incorrect mask shape
    incorrect_band_mask = torch.randint(0, 3, (3, 240, 240, 1))
    with pytest.raises(ValueError):
        mt_lime.validate_mask_shape("band", hsi=hsi, mask=incorrect_band_mask)

    unbroadcastable_band_mask = torch.randint(0, 3, (3, 243, 240))
    with pytest.raises(ValueError):
        mt_lime.validate_mask_shape("band", hsi=hsi, mask=unbroadcastable_band_mask)

    incorrectly_broadcastable_band_mask = torch.randint(0, 3, (3, 240, 1))
    incorrectly_broadcastable_hsi = HSI(image=torch.randn(1, 240, 240), wavelengths=[462.08])
    with pytest.raises(ShapeMismatchError):
        mt_lime.validate_mask_shape("band", hsi=incorrectly_broadcastable_hsi, mask=incorrectly_broadcastable_band_mask)


###################################################################
############################ EXPLAINER ############################
###################################################################


def test_lime_explainer():
    # Create mock objects for ExplainableModel and InterpretableModel
    explainable_model = ExplainableModel(forward_func=lambda x: x.mean(dim=(1, 2, 3)), problem_type="regression")
    interpretable_model = SkLearnLasso()

    # Create a Lime object
    lime = mt_lime.Lime(explainable_model, interpretable_model)

    # Assert that the explainable_model and interpretable_model attributes are set correctly
    assert lime.explainable_model == explainable_model
    assert lime.interpretable_model == interpretable_model
    assert lime._attribution_method is not None
    with pytest.raises(ValueError):
        lime.attribute(hsi=HSI(image=torch.randn(3, 240, 240), wavelengths=[100, 200, 300]), target=0)

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
    assert lime._attribution_method is not None

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
    assert lime._attribution_method is not None


def test__get_slic_segmentation_mask():
    # Create a sample hsi
    hsi = HSI(image=torch.randn(3, 240, 240), wavelengths=[462.08, 465.27, 468.47])

    # Call the method
    segmentation_mask = mt_lime.Lime._get_slic_segmentation_mask(hsi, num_interpret_features=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert torch.all(segmentation_mask >= 0)
    assert torch.all(segmentation_mask < 10)

    # Check mask slic
    mask = torch.ones_like(hsi.image)
    mask[0, 10, 10] = 0
    mask[0, 20, 20] = 0

    hsi = HSI(image=torch.randn(3, 240, 240), wavelengths=[462.08, 465.27, 468.47], binary_mask=mask)
    segmentation_mask = mt_lime.Lime._get_slic_segmentation_mask(hsi, num_interpret_features=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert torch.all(segmentation_mask >= 0)
    assert torch.all(segmentation_mask < 10)
    assert segmentation_mask[0, 10, 10] == 0
    assert segmentation_mask[0, 20, 20] == 0


def test__get_patch_segmentation_mask():
    # Create a sample hsi
    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])

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

    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47], binary_mask=mask)
    segmentation_mask = mt_lime.Lime._get_patch_segmentation_mask(hsi, patch_size=10)

    # Check the output
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)
    assert segmentation_mask[0, 10, 10] == 0
    assert segmentation_mask[0, 20, 20] == 0


def test_get_segmentation_mask():
    # Test case 1: Valid segmentation method (slic)
    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="slic")
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)

    # Test case 2: Valid segmentation method (patch)
    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    segmentation_mask = mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="patch")
    assert isinstance(segmentation_mask, torch.Tensor)
    assert segmentation_mask.shape == (1, 240, 240)

    # Test case 3: Invalid segmentation method
    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[462.08, 465.27, 468.47])
    with pytest.raises(MaskCreationError):
        mt_lime.Lime.get_segmentation_mask(hsi, segmentation_method="invalid")

    # Test case 4: Invalid segmentation hsi
    hsi = torch.ones((3, 240, 240))
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)

    # Test case 5: Incorrect segment name type
    segment_name_invalid = [123]
    with pytest.raises(TypeError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)

    # Test case 6: Incorrect segment name type
    segment_name_invalid = (123,)
    with pytest.raises(TypeError):
        mt_lime.Lime._make_band_names_indexable(segment_name_invalid)


def test__extract_bands_from_spyndex():
    # Test case 1: Single band name
    segment_name = "R"
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert result == "R"

    segment_name = "G1"
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert result == "G1"

    # Test case 2: Multiple band names
    segment_name = ["R", "G"]
    result = mt_lime.Lime._extract_bands_from_spyndex(segment_name)
    assert isinstance(result, tuple)
    assert list(result).sort() == ["R", "G"].sort()

    # Test case 3: Invalid band name
    segment_name = "D"
    with pytest.raises(BandSelectionError):
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
    with pytest.raises(BandSelectionError):
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
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
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

    # Test bands out of bounds - a warning should be raised
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700])
    band_names = ["G", "AVI"]

    mt_lime.Lime._get_band_wavelengths_indices_from_band_names(wavelengths, band_names)

    # Test invialid band names
    band_names = 123
    with pytest.raises(TypeError):
        mt_lime.Lime._get_band_wavelengths_indices_from_band_names(wavelengths, band_names)

    band_names = "invalid band"
    with pytest.raises(BandSelectionError):
        mt_lime.Lime._get_band_wavelengths_indices_from_band_names(wavelengths, band_names)


def test__check_overlapping_segments():
    # Create a sample dictionary mapping segment labels to indices
    dict_labels_to_indices = {
        "segment1": [0, 1],
        "segment2": [1, 2],
        "segment3": [2, 3],
    }
    mt_lime.Lime._check_overlapping_segments(dict_labels_to_indices)

    non_overlapping_dict_labels_to_indices = {
        "segment1": [0, 1],
        "segment2": [2, 3],
    }
    mt_lime.Lime._check_overlapping_segments(non_overlapping_dict_labels_to_indices)


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
    hsi = HSI(image=torch.ones((len(wavelengths), 3, 3)), wavelengths=wavelengths)

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
    hsi = HSI(image=torch.ones((3, 240, 240)), wavelengths=[400, 500, 600])
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
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["R", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"R": 1, "G": 2}

    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["S2", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"S2": 1, "G": 2}

    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["BI", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"BI": 1, "G": 2}

    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["S2", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"S2": 1, "G": 2}

    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["BI", "G"]
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_names=band_names)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"BI": 1, "G": 2}

    # Test case 2: Valid input with band indices
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": 0}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 3: Valid input with band indices list
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": [0, 1, 2]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 4: Valid input with band ranges (indices)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_ranges_indices = {"RGB": [(0, 2)]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_indices=band_ranges_indices)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 5: Valid input with band wavelengths
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": 500.43}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 6: Valid input with band wavelengths list
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": [500.43, 554.78]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 7: Valid input with band ranges (wavelengths)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_ranges_wavelengths = {"RGB": [(400, 600)]}
    band_mask, dict_labels_to_segment_ids = mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_ranges_wavelengths)
    assert isinstance(band_mask, torch.Tensor)
    assert band_mask.shape == (len(wavelengths), 1, 1)
    assert dict_labels_to_segment_ids == {"RGB": 1}

    # Test case 7: Valid input with band ranges (wavelengths) and too many params
    mt_lime.Lime.get_band_mask(hsi, band_names=band_names, band_indices=band_ranges_indices)
    mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_ranges_wavelengths, band_indices=band_indices)

    # Test case 8: Invalid input (no band names, groups, or ranges provided)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    with pytest.raises(MaskCreationError):
        mt_lime.Lime.get_band_mask(hsi)

    # Test case 9: Invalid input (incorrect band names)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_names = ["R", "G", "B", "Invalid"]
    with pytest.raises(MaskCreationError):
        mt_lime.Lime.get_band_mask(hsi, band_names=band_names)

    # Test case 10: Invalid input (incorrect band ranges wavelengths)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": [(wavelengths[-1] + 10, wavelengths[-1] + 20)]}
    with pytest.raises(MaskCreationError):
        mt_lime.Lime.get_band_mask(hsi, band_wavelengths=band_wavelengths)

    # Test case 11: Invalid input (incorrect band ranges indices)
    hsi = HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_indices = {"RGB": [(len(wavelengths), len(wavelengths) + 1)]}
    with pytest.raises(MaskCreationError):
        mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)

    # Test case 12: Invalid input image
    hsi = torch.ones((len(wavelengths), 10, 10))
    with pytest.raises(TypeError):
        mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices)

    # Test case 13: Band mask with multiple inputs
    hsi = mt.HSI(image=torch.ones((len(wavelengths), 10, 10)), wavelengths=wavelengths)
    band_wavelengths = {"RGB": [500.43, 554.78]}
    band_names = ["BI", "G"]
    band_indices = {"RGB": [0, 1, 2]}

    mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices, band_names=band_names)

    mt_lime.Lime.get_band_mask(hsi, band_indices=band_indices, band_wavelengths=band_wavelengths)


def test_get_spatial_attributes_regression():
    # Dumb model
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600])
    hsi = HSI(image=torch.randn(5, 10, 10), wavelengths=wavelengths)

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
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Assert smaller segmentation mask
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask_smaller, target=0)
    assert isinstance(spatial_attributes, HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=1)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple segmentation
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], [segmentation_mask, segmentation_mask], target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Multiple images with multiple targets
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=[0, 1])
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Wrong number of masks
    with pytest.raises(ValueError):
        lime.get_spatial_attributes(hsi, [segmentation_mask, segmentation_mask], target=0)

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="slic", target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use slic with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="slic", target=0)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="patch", target=0, patch_size=5)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use patch with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="patch", target=0, patch_size=5)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

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
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
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
    hsi = HSI(image=torch.randn(5, 10, 10), wavelengths=wavelengths)

    # Create a sample segmentation mask
    segmentation_mask = torch.randint(1, 4, (1, 10, 10))

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "classification"), interpretable_model=SkLearnLasso(alpha=0.1)
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=1)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple segmentation
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], [segmentation_mask, segmentation_mask], target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Multiple images with multiple targets
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=[0, 1])
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Wrong number of masks
    with pytest.raises(ValueError):
        lime.get_spatial_attributes(hsi, [segmentation_mask, segmentation_mask], target=0)

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="slic", target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use slic with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="slic", target=0)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="patch", target=0, patch_size=5)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use patch with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="patch", target=0, patch_size=5)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

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
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape


def test_get_spatial_attributes_segmentation():
    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = HSI(image=torch.randn(len(wavelengths), 10, 10), wavelengths=wavelengths)

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

    # Get postprocessingagg_segmentation_postprocessing
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation", postprocessing),
        interpretable_model=SkLearnLasso(alpha=0.1),
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 1: Different target
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=1)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple segmentation
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], [segmentation_mask, segmentation_mask], target=0)
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Multiple images with multiple targets
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_mask, target=[0, 1])
    assert len(spatial_attributes) == 2
    assert torch.equal(spatial_attributes[0].segmentation_mask, segmentation_mask[0, :, :])
    assert torch.equal(spatial_attributes[1].segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Wrong number of masks
    with pytest.raises(ValueError):
        lime.get_spatial_attributes(hsi, [segmentation_mask, segmentation_mask], target=0)

    # Test case 2: Use slic for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="slic", target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use slic with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="slic", target=0)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 3: Use patch for segmentation
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_method="patch", target=0, patch_size=5)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert spatial_attributes.segmentation_mask is not None
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Use patch with multiple images
    spatial_attributes = lime.get_spatial_attributes([hsi, hsi], segmentation_method="patch", target=0, patch_size=5)
    assert len(spatial_attributes) == 2
    assert spatial_attributes[0].hsi == hsi
    assert spatial_attributes[0].segmentation_mask is not None
    assert spatial_attributes[0].score <= 1.0
    assert spatial_attributes[0].attributes.shape == hsi.image.shape
    assert spatial_attributes[1].hsi == hsi
    assert spatial_attributes[1].segmentation_mask is not None
    assert spatial_attributes[1].score <= 1.0
    assert spatial_attributes[1].attributes.shape == hsi.image.shape

    # Test case 4: different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation", postprocessing),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Call the get_spatial_attributes method
    spatial_attributes = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    # Test No segmentation postprocessing
    with pytest.raises(ValueError):
        lime = mt_lime.Lime(
            explainable_model=ExplainableModel(dumb_model, "segmentation"),
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
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
    hsi = HSI(image=torch.randn(len(wavelengths), 240, 240), wavelengths=wavelengths)

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
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Assert smaller band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask_smaller, band_names=band_names, target=0)
    assert isinstance(spectral_attributes, HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=1)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple band masks
    spectral_attributes = lime.get_spectral_attributes(
        [hsi, hsi], [band_mask, band_mask], band_names=band_names, target=0
    )
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Incorrect number of band masks
    with pytest.raises(ValueError):
        spectral_attributes = lime.get_spectral_attributes(hsi, [band_mask, band_mask], band_names=band_names, target=0)

    # Test case 3: Multiple images with multiple targets
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=[0, 1])
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert spectral_attributes[0].hsi == hsi
    assert spectral_attributes[0].band_mask is not None
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].hsi == hsi
    assert spectral_attributes[1].band_mask is not None
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

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
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
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
    hsi = HSI(image=torch.randn(len(wavelengths), 240, 240), wavelengths=wavelengths)

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
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=1)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple band masks
    spectral_attributes = lime.get_spectral_attributes(
        [hsi, hsi], [band_mask, band_mask], band_names=band_names, target=0
    )
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Incorrect number of band masks
    with pytest.raises(ValueError):
        spectral_attributes = lime.get_spectral_attributes(hsi, [band_mask, band_mask], band_names=band_names, target=0)

    # Test case 3: Multiple images with multiple targets
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=[0, 1])
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert spectral_attributes[0].hsi == hsi
    assert spectral_attributes[0].band_mask is not None
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].hsi == hsi
    assert spectral_attributes[1].band_mask is not None
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

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
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape


def test_get_spectral_attributes_segmentation():
    # Create a sample image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = HSI(image=torch.randn(len(wavelengths), 10, 10), wavelengths=wavelengths)

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

    # Get postprocessing
    postprocessing = agg_segmentation_postprocessing(classes_numb=3)

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation", postprocessing),
        interpretable_model=SkLearnLasso(alpha=0.1),
    )

    # Call the get_spectral_attributes method
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Different target
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=1)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 2: Multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Test case 2: Multiple images and multiple band masks
    spectral_attributes = lime.get_spectral_attributes(
        [hsi, hsi], [band_mask, band_mask], band_names=band_names, target=0
    )
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Incorrect number of band masks
    with pytest.raises(ValueError):
        spectral_attributes = lime.get_spectral_attributes(hsi, [band_mask, band_mask], band_names=band_names, target=0)

    # Test case 3: Multiple images with multiple targets
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, band_names=band_names, target=[0, 1])
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band names no Band mask
    spectral_attributes = lime.get_spectral_attributes(hsi, band_names=band_names, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert spectral_attributes.band_mask is not None
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band names with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_names=band_names, target=0)
    assert len(spectral_attributes) == 2
    assert spectral_attributes[0].hsi == hsi
    assert spectral_attributes[0].band_mask is not None
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].hsi == hsi
    assert spectral_attributes[1].band_mask is not None
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Use Band mask with multiple images
    spectral_attributes = lime.get_spectral_attributes([hsi, hsi], band_mask, target=0)
    assert len(spectral_attributes) == 2
    assert torch.equal(spectral_attributes[0].band_mask, band_mask[:, 0, 0])
    assert torch.equal(spectral_attributes[1].band_mask, band_mask[:, 0, 0])
    assert spectral_attributes[0].score <= 1.0
    assert spectral_attributes[1].score <= 1.0
    assert spectral_attributes[0].attributes.shape == hsi.image.shape
    assert spectral_attributes[1].attributes.shape == hsi.image.shape

    # Test case different lime parameters
    similarity_func = mt_lime_base.get_exp_kernel_similarity_function(distance_mode="cosine", kernel_width=1000)
    interpretable_model = SkLearnLasso(alpha=0.08)
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "segmentation", postprocessing),
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )

    # Use Band mask no Band names
    spectral_attributes = lime.get_spectral_attributes(hsi, band_mask, target=0)

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names is not None
    assert spectral_attributes.score <= 1.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    # Test case 5: No segmentation postprocessing
    with pytest.raises(ValueError):
        lime = mt_lime.Lime(
            explainable_model=ExplainableModel(dumb_model, "segmentation"),
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
        )


def test_attribute_wrapper():
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    # Create a sample hsi image
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    hsi = HSI(image=torch.randn(len(wavelengths), 240, 240), wavelengths=wavelengths)

    # Create a sample band mask
    band_mask = torch.zeros_like(hsi.image, dtype=int)
    band_mask[0] = 1
    band_mask[1] = 2

    # Create a sample band names dictionary
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create a sample Lime object
    lime = mt_lime.Lime(
        explainable_model=ExplainableModel(dumb_model, "regression"), interpretable_model=SkLearnLasso(alpha=0.9)
    )

    # Call the get_spectral_attributes method
    spectral_attributes = lime.attribute(
        attribution_type="spectral", hsi=hsi, band_mask=band_mask, band_names=band_names, target=0
    )

    # Assert the output type and properties
    assert isinstance(spectral_attributes, mt.attr.HSISpectralAttributes)
    assert spectral_attributes.hsi == hsi
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names
    assert spectral_attributes.score <= 1.0 and spectral_attributes.score >= 0.0
    assert isinstance(spectral_attributes.score, float)
    assert spectral_attributes.attributes.shape == hsi.image.shape

    spectral_attributes_get_method = lime.get_spectral_attributes(hsi, band_mask, band_names=band_names, target=0)
    assert torch.equal(spectral_attributes.attributes, spectral_attributes_get_method.attributes)
    assert spectral_attributes.score == spectral_attributes_get_method.score
    assert torch.equal(spectral_attributes.band_mask, spectral_attributes_get_method.band_mask)
    assert spectral_attributes.band_names == spectral_attributes_get_method.band_names

    # Test case 2: Spatial attributes
    def dumb_model(image: torch.Tensor) -> torch.Tensor:
        output = torch.empty((image.shape[0], 2))
        output[:, 0] = 0
        output[:, 1] = 1
        return output

    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600])
    hsi = HSI(image=torch.randn(5, 10, 10), wavelengths=wavelengths)

    # Create a sample segmentation mask
    segmentation_mask = torch.randint(1, 4, (1, 10, 10)).expand_as(hsi.image)

    spatial_attributes = lime.attribute(
        attribution_type="spatial", hsi=hsi, segmentation_mask=segmentation_mask, target=0
    )

    # Assert the output type and properties
    assert isinstance(spatial_attributes, mt.attr.HSISpatialAttributes)
    assert spatial_attributes.hsi == hsi
    assert torch.equal(spatial_attributes.segmentation_mask, segmentation_mask[0, :, :])
    assert spatial_attributes.score <= 1.0 and spatial_attributes.score >= 0.0
    assert spatial_attributes.attributes.shape == hsi.image.shape

    spatial_attributes_get_method = lime.get_spatial_attributes(hsi, segmentation_mask, target=0)
    assert torch.equal(spatial_attributes.attributes, spatial_attributes_get_method.attributes)
    assert spatial_attributes.score == spatial_attributes_get_method.score
    assert torch.equal(spatial_attributes.segmentation_mask, spatial_attributes_get_method.segmentation_mask)

    # Test case 3: Invalid attribute type
    with pytest.raises(ValueError):
        lime.attribute(attribution_type="invalid", hsi=hsi, band_mask=band_mask, band_names=band_names, target=0)
