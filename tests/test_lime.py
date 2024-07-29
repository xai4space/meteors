import pytest

import torch
import numpy as np

import meteors as mt
import meteors.lime as mt_lime

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

#####################################################################
############################ VALIDATIONS ############################
#####################################################################


def test_validate_torch_tensor_type():
    # Test with numpy array
    np_array = np.array([1, 2, 3])
    result = mt_lime.validate_torch_tensor_type(np_array, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([1, 2, 3])))

    # Test with torch tensor
    torch_tensor = torch.tensor([4, 5, 6])
    result = mt_lime.validate_torch_tensor_type(torch_tensor, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([4, 5, 6])))

    # Test with invalid input type
    with pytest.raises(TypeError):
        mt_lime.validate_torch_tensor_type("invalid", "Input must be a numpy array or torch tensor")


def test_validate_attributes():
    # Test with numpy array
    attributes_np = np.ones((3, 4))
    attributes_torch = mt_lime.validate_attributes(attributes_np)
    assert isinstance(attributes_torch, torch.Tensor)
    assert torch.all(attributes_torch.eq(torch.tensor(attributes_np)))

    # Test with torch tensor
    attributes_torch = torch.ones((3, 4))
    attributes_torch_validated = mt_lime.validate_attributes(attributes_torch)
    assert isinstance(attributes_torch_validated, torch.Tensor)
    assert torch.all(attributes_torch_validated.eq(attributes_torch))

    # Test with invalid type
    with pytest.raises(TypeError):
        mt_lime.validate_attributes(123)


def test_validate_segmentation_mask():
    # Test case 1: Valid segmentation mask (numpy array)
    segmentation_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_lime.validate_segmentation_mask(segmentation_mask_np)
    assert isinstance(result_np, torch.Tensor), "The result should be a torch.Tensor"
    assert torch.all(
        torch.eq(result_np, torch.tensor(segmentation_mask_np))
    ), "The result should be equal to the input segmentation mask"

    # Test case 2: Valid segmentation mask (torch tensor)
    segmentation_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_lime.validate_segmentation_mask(segmentation_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor), "The result should be a torch.Tensor"
    assert torch.all(
        torch.eq(result_tensor, segmentation_mask_tensor)
    ), "The result should be equal to the input segmentation mask"

    # Test case 3: Invalid segmentation mask (wrong type)
    invalid_segmentation_mask = "invalid"
    with pytest.raises(TypeError):
        mt_lime.validate_segmentation_mask(invalid_segmentation_mask)

    # Test case 4: Invalid segmentation mask (wrong type)
    invalid_segmentation_mask = 123
    with pytest.raises(TypeError):
        mt_lime.validate_segmentation_mask(invalid_segmentation_mask)


def test_validate_band_mask():
    # Test case 1: Valid band mask (numpy array)
    band_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_lime.validate_band_mask(band_mask_np)
    assert isinstance(result_np, torch.Tensor), "The result should be a torch.Tensor"
    assert torch.all(
        torch.eq(result_np, torch.tensor(band_mask_np))
    ), "The result should be equal to the input band mask"

    # Test case 2: Valid band mask (torch tensor)
    band_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_lime.validate_band_mask(band_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor), "The result should be a torch.Tensor"
    assert torch.all(torch.eq(result_tensor, band_mask_tensor)), "The result should be equal to the input band mask"

    # Test case 3: Invalid band mask (wrong type)
    invalid_band_mask = "invalid"
    with pytest.raises(TypeError):
        mt_lime.validate_band_mask(invalid_band_mask)

    # Test case 4: Invalid band mask (wrong type)
    invalid_band_mask = 123
    with pytest.raises(TypeError):
        mt_lime.validate_band_mask(invalid_band_mask)


def test_validate_shapes():
    shape = (len(wavelengths), 240, 240)
    attributes = torch.ones(shape)
    image = mt.Image(image=torch.ones(shape), wavelengths=wavelengths)

    # Test case 1: Valid shapes
    mt_lime.validate_shapes(attributes, image)  # No exception should be raised

    # Test case 2: Invalid shapes
    invalid_attributes = torch.ones((150, 240, 241))
    with pytest.raises(ValueError):
        mt_lime.validate_shapes(invalid_attributes, image)

    invalid_image = mt.Image(image=torch.ones((len(wavelengths), 240, 241)), wavelengths=wavelengths)
    with pytest.raises(ValueError):
        mt_lime.validate_shapes(attributes, invalid_image)


def test_validate_band_names_with_mask():
    # Test case 1: Changed band names
    band_names = {"R": 1, "G": 2, "B": 3}
    band_mask = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 3]])

    updated_band_names = mt_lime.validate_band_names_with_mask(band_names, band_mask)

    assert updated_band_names == {
        "R": 1,
        "G": 2,
        "B": 3,
        "not_included": 0,
    }, "The band names should be updated with 'not_included' key"

    # Test case 2: Not changed band names
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.tensor([[1, 0, 1], [1, 2, 1], [1, 0, 1]])

    not_updated_band_names = mt_lime.validate_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {
        "R": 0,
        "G": 1,
        "B": 2,
    }, "The band names should not be updated if all bands are included in the mask"

    band_names = {"R": 0}
    band_mask = torch.tensor([[0], [0], [0]])

    not_updated_band_names = mt_lime.validate_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {"R": 0}, "The band names should not be updated if the mask is empty"

    # Test case 3: Invalid band names
    band_names = {"R": 1, "G": 2, "B": 3}
    invalid_band_mask = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        mt_lime.validate_band_names_with_mask(band_names, invalid_band_mask)


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


def test_validate_band_ranges_or_list():
    dict_band_names = "test"
    # Test case 1: Valid band ranges (tuple)
    band_ranges_tuple = (400, 700)
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_ranges_tuple}, variable_name="test")

    band_ranges_tuple = (400.0, 700.0)
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_ranges_tuple}, variable_name="test")

    # Test case 2: Valid band ranges (list of tuples)
    band_ranges_list = [(400, 500), (600, 700)]
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_ranges_list}, variable_name="test")

    band_ranges_list = [(400.0, 500.0), (600.0, 700.0)]
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_ranges_list}, variable_name="test")

    # Test case 3: Valid band value (int)
    band_value = 400
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_value}, variable_name="test")

    band_value = 400.0
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_value}, variable_name="test")

    # Test case 4: Valid band list (list of ints)
    band_list = [400, 500, 600]
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_list}, variable_name="test")

    band_list = [400.0, 500.0, 600.0]
    mt_lime.validate_band_ranges_or_list({dict_band_names: band_list}, variable_name="test")

    # Test case 5: Invalid band ranges (wrong type)
    invalid_band_ranges = "invalid"
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_ranges_or_list({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value), "The error message should contain the variable name"

    # Test case 6: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400, 500, 600)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_ranges_or_list({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value), "The error message should contain the variable name"

    # Test case 7: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400, 500), 600]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_ranges_or_list({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value), "The error message should contain the variable name"

    # Test case 8: Invalid band ranges (wrong format)
    invalid_band_ranges = [(400,)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_ranges_or_list({dict_band_names: invalid_band_ranges}, variable_name="test")
    assert "test" in str(excinfo.value), "The error message should contain the variable name"

    # Test case 9: Different variable name
    invalid_band_ranges = [(400,)]
    with pytest.raises(TypeError) as excinfo:
        mt_lime.validate_band_ranges_or_list(
            {dict_band_names: invalid_band_ranges}, variable_name="pizza with a pineapple is not a pizza"
        )
    assert "pizza with a pineapple is not a pizza" in str(
        excinfo.value
    ), "The error message should contain the variable name"
    assert "test" not in str(excinfo.value), "The error message should not contain the default variable name"


def test_validate_segment_format_range():
    # Test case 1: Valid segment range (tuple of ints)
    segment_range_tuple = (0, 10)
    result_tuple = mt_lime.validate_segment_format_range(segment_range_tuple)
    assert isinstance(result_tuple, list), "The result should be a list"
    assert len(result_tuple) == 1, "The result should contain a single tuple"
    assert result_tuple[0] == segment_range_tuple, "The result should be equal to the input segment range"
    assert isinstance(result_tuple[0][0], int) and isinstance(
        result_tuple[0][1], int
    ), "The result should be a tuple of ints"

    # Test case 2: Valid segment range (list of tuples of ints)
    segment_range_list = [(0, 10), (20, 30)]
    result_list = mt_lime.validate_segment_format_range(segment_range_list)
    assert isinstance(result_list, list), "The result should be a list"
    assert len(result_list) == len(segment_range_list), "The result should contain the same number of tuples"
    assert all(
        result_tuple == segment_range_tuple
        for result_tuple, segment_range_tuple in zip(result_list, segment_range_list)
    ), "The result should be equal to the input segment range"

    # Test case 3: Valid segment range (tuple of ints) of floats
    segment_range_tuple = (0.0, 10.0)
    result_tuple = mt_lime.validate_segment_format_range(segment_range_tuple, dtype=float)
    assert isinstance(result_tuple, list), "The result should be a list"
    assert len(result_tuple) == 1, "The result should contain a single tuple"
    assert result_tuple[0] == segment_range_tuple, "The result should be equal to the input segment range"
    assert isinstance(result_tuple[0][0], float) and isinstance(
        result_tuple[0][1], float
    ), "The result should be a tuple of floats"

    # Test case 4: Valid segment range (list of tuples of ints) of floats
    segment_range_list = [(0.0, 10.0), (20.0, 30.0)]
    result_list = mt_lime.validate_segment_format_range(segment_range_list, dtype=float)
    assert isinstance(result_list, list), "The result should be a list"
    assert len(result_list) == len(segment_range_list), "The result should contain the same number of tuples"
    assert all(
        result_tuple == segment_range_tuple
        for result_tuple, segment_range_tuple in zip(result_list, segment_range_list)
    ), "The result should be equal to the input segment range"

    # Test case 5: Invalid segment range (wrong type)
    invalid_segment_range = "invalid"
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 6: Invalid segment range (wrong type)
    invalid_segment_range = 123
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 7: Invalid segment range (tuple of floats)
    invalid_segment_range = (0.5, 10.5)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 8: Invalid segment range (list of tuples of floats)
    invalid_segment_range = [(0.5, 10.5), (20.5, 30.5)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 9: Invalid segment range (tuple of ints with wrong order)
    invalid_segment_range = (10, 0)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 10: Invalid segment range (list of tuples of ints with wrong order)
    invalid_segment_range = [(10, 0), (30, 20)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range)

    # Test case 11: Invalid segment range (tuple of floats with wrong order)
    invalid_segment_range = (10.5, 0.5)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range, dtype=float)

    # Test case 12: Invalid segment range (list of tuples of floats with wrong order)
    invalid_segment_range = [(10.5, 0.5), (30.5, 20.5)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range, dtype=float)

    # Test case 13: Invalid segment dtype
    invalid_segment_range = (0, 10)
    with pytest.raises(ValueError):
        mt_lime.validate_segment_format_range(invalid_segment_range, dtype=float)


def test_validate_segment_range():
    wavelengths = torch.tensor([400, 450, 500, 550, 600, 650, 700, 750, 800])

    # Test case 1: Valid segment range
    segment_range = [(2, 5), (7, 9)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(2, 5), (7, 9)], "The segment range should remain unchanged"

    # Test case 2: Segment range with end exceeding wavelengths max index
    segment_range = [(0, 3), (6, 10)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(0, 3), (6, 9)], "The segment range should be adjusted to end at wavelengths max index"

    # Test case 3: Segment range with negative start
    segment_range = [(-1, 4), (7, 9)]
    result = mt_lime.validate_segment_range(wavelengths, segment_range)
    assert result == [(0, 4), (7, 9)], "The segment range should be adjusted to start at 0"

    # Test case 4: Not Valid segment range
    segment_range = [(0, 0), (7, 9)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)

    segment_range = [(-1, 0), (7, 9)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)

    segment_range = [(2, 5), (9, 9)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)

    segment_range = [(2, 5), (9, 10)]
    with pytest.raises(ValueError):
        mt_lime.validate_segment_range(wavelengths, segment_range)


# def test_band_mask_created_from_band_names():
#     shape = (150, 240, 240)
#     sample = torch.ones(shape)

#     image = mt.Image(
#         image=sample,
#         wavelengths=wavelengths,
#         orientation=("C", "H", "W"),
#         binary_mask="artificial",
#     )

#     band_names_list = [["R", "G"], "B"]

#     band_mask, band_names = mt.Lime.get_band_mask(image, band_names_list)

#     assert band_names == {
#         ("R", "G"): 1,
#         "B": 2,
#     }, "There should be only 3 values in the band mask corresponding to (R, G), B and background"
#     assert (
#         len(torch.unique(band_mask)) == 3
#     ), "There should be only 3 values in the band mask corresponding to (R, G), B and background"
#     assert torch.equal(
#         torch.unique(band_mask), torch.tensor([0, 1, 2])
#     ), "There should be only 3 values in the band mask corresponding to (R, G), B and background"
#     for c in range(shape[0]):
#         assert len(torch.unique(band_mask[c, :, :])) == 1, "On each channel there should be only one unique number"


# def test_band_mask_errors():
#     shape = (150, 240, 240)
#     sample = torch.ones(shape)
#     image = mt.Image(
#         image=sample,
#         wavelengths=wavelengths,
#         orientation=("C", "H", "W"),
#         binary_mask="artificial",
#     )

#     with pytest.raises(
#         AssertionError,
#         match="No band names, groups, or ranges provided",
#     ):
#         mt.Lime.get_band_mask(image)

#     with pytest.raises(
#         ValueError,
#         match="Incorrect band names provided",
#     ):
#         mt.Lime.get_band_mask(image, band_names=4)

#     with pytest.raises(
#         ValueError,
#         match="Incorrect band ranges wavelengths provided, please check if provided wavelengths are correct",
#     ):
#         band_ranges_wavelengths = {"bad_range": 1}
#         mt.Lime.get_band_mask(image, band_ranges_wavelengths=band_ranges_wavelengths)

#     with pytest.raises(
#         TypeError,
#     ):
#         band_ranges_wavelengths = {"bad_structure": (1, 2, 3)}
#         mt.Lime.get_band_mask(image, band_ranges_wavelengths=band_ranges_wavelengths)

#     with pytest.raises(
#         TypeError,
#     ):
#         band_ranges_wavelengths = {"bad_order": (2, 1)}
#         mt.Lime.get_band_mask(image, band_ranges_wavelengths=band_ranges_wavelengths)

#     # with pytest.raises(
#     #     TypeError,
#     # ):
#     #     band_ranges_indices = {"bad_range": 1}
#     #     mt.Lime.get_band_mask(image, band_ranges_indices=band_ranges_indices)
#     # Now we accept band range with one value as its mean that we want to have mask with only this value

#     with pytest.raises(
#         TypeError,
#     ):
#         band_ranges_indices = {"bad_structure": (1, 2, 3)}
#         mt.Lime.get_band_mask(image, band_ranges_indices=band_ranges_indices)

#     with pytest.raises(
#         TypeError,
#     ):
#         band_ranges_indices = {"bad_order": (2, 1)}
#         mt.Lime.get_band_mask(image, band_ranges_indices=band_ranges_indices)


# def test_dummy_explainer():
#     def forward_func(x: torch.tensor):
#         return x.mean(dim=(1, 2, 3))

#     explainable_model = mt.utils.models.ExplainableModel(forward_func=forward_func, problem_type="regression")

#     interpretable_model = mt.utils.models.SkLearnLasso()

#     lime = mt.Lime(explainable_model=explainable_model, interpretable_model=interpretable_model)

#     assert lime.device == torch.device("cpu"), "Device should be set to cpu by default"

#     shape = (150, 240, 240)
#     sample = torch.ones(shape)

#     image = mt.Image(
#         image=sample,
#         wavelengths=wavelengths,
#         orientation=("C", "H", "W"),
#         binary_mask="artificial",
#     )

#     # Test spectral attribution

#     band_names_list = ["AVI", "B"]
#     # TODO: decide what to do if `AFRI1600` some bands are out of wavelengths do we ignore or throw error?

#     band_mask, band_names = lime.get_band_mask(image, band_names_list)

#     lime.get_spectral_attributes(image=image, band_mask=band_mask, band_names=band_names)

#     segmentation_mask = lime.get_segmentation_mask(image, "patch")
#     segmentation_mask = lime.get_segmentation_mask(image, "slic")
#     segmentation_mask = lime.get_segmentation_mask(image, "slic")

#     lime.get_spatial_attributes(image=image, segmentation_mask=segmentation_mask)

#     lime.get_spatial_attributes(image=image, segmentation_method="slic", num_interpret_features=3)
#     lime.get_spatial_attributes(image=image, segmentation_method="patch")

######################################################################
############################ EXPLANATIONS ############################
######################################################################


def test_validate_image_attributions():
    # Create a sample ImageAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    image_attributes = mt.ImageAttributes(
        image=image, attributes=attributes, score=score, device=device, model_config=model_config
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert image_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert image_attributes.image.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert image_attributes.attributes.shape == image_attributes.image.image.shape

    # Assert that the device of the image object has been updated
    assert image_attributes.image.device == device

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.ImageAttributes(
            image=image, attributes=invalid_attributes, score=score, device=device, model_config=model_config
        )


def test_spatial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert spatial_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert spatial_attributes.image.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spatial_attributes.segmentation_mask.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert spatial_attributes.attributes.shape == spatial_attributes.image.image.shape

    # Assert that the device of the image object has been updated
    assert spatial_attributes.image.device == device

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.ImageSpatialAttributes(
            image=image,
            attributes=invalid_attributes,
            score=score,
            segmentation_mask=segmentation_mask,
            device=device,
            model_config=model_config,
        )


def test_to_image_spatial_attributes():
    # Create dummy data
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))

    # Create ImageSpatialAttributes object
    spatial_attributes = mt.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=0.8,
        segmentation_mask=segmentation_mask,
    )

    # Move to a different device
    device = torch.device("cpu")
    spatial_attributes.to(device)

    # Check if the image, attributes, and segmentation mask are moved to the new device
    assert spatial_attributes.image.device == device
    assert spatial_attributes.attributes.device == device
    assert spatial_attributes.segmentation_mask.device == device

    if torch.cuda.is_available():
        # Move back to the original device
        device = torch.device("cuda")
        spatial_attributes.to(device)

        # Check if the image, attributes, and segmentation mask are moved to the original device
        assert spatial_attributes.image.device == device
        assert spatial_attributes.attributes.device == device
        assert spatial_attributes.segmentation_mask.device == device


def test_flattened_segmentation_mask_spacial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    assert (
        spatial_attributes.flattened_segmentation_mask.shape == segmentation_mask.shape[1:]
    ), "The flattened segmentation mask should have shape equal to the segmentation mask shape"


def test_flattened_attributes_spacial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    assert (
        spatial_attributes.flattened_attributes.shape == segmentation_mask.shape[1:]
    ), "The flattened attributes should have shape equal to the segmentation mask shape"


def test_spectral_attributes():
    # Create a sample ImageSpectralAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spectral_attributes = mt.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
        device=device,
        model_config=model_config,
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert spectral_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert spectral_attributes.image.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spectral_attributes.band_mask.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert spectral_attributes.attributes.shape == spectral_attributes.image.image.shape

    # Assert that the device of the image object has been updated
    assert spectral_attributes.image.device == device

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValueError):
        mt.ImageSpectralAttributes(
            image=image,
            attributes=invalid_attributes,
            score=score,
            band_names=band_names,
            band_mask=band_mask,
            device=device,
            model_config=model_config,
        )


def test_to_image_spectral_attributes():
    # Create dummy data
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}

    # Create ImageSpectralAttributes object
    spectral_attributes = mt.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=0.8,
        band_mask=band_mask,
        band_names=band_names,
    )

    # Move to a different device
    device = torch.device("cpu")
    spectral_attributes.to(device)

    # Check if the image, attributes, band mask, and band names are moved to the new device
    assert spectral_attributes.image.device == device
    assert spectral_attributes.attributes.device == device
    assert spectral_attributes.band_mask.device == device


def test_flattened_band_mask_spectral_attributes():
    # Create a sample ImageSpectralAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spectral_attributes = mt.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(
        spectral_attributes.flattened_band_mask, torch.tensor([0, 1, 2])
    ), "The flattened band mask should be equal to [0, 1, 2]"


def test_flattened_attributes_spectral_attributes():
    # Create a sample ImageSpectralAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spectral_attributes = mt.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(
        spectral_attributes.flattened_attributes, torch.tensor([0, 1, 2])
    ), "The flattened attributes should be equal to [0, 1, 2]"


###################################################################
############################ EXPLAINER ############################
###################################################################

# def test_lime_explainer():
#     # Create mock objects for ExplainableModel and InterpretableModel
#     explainable_model = MagicMock(spec=ExplainableModel)
#     interpretable_model = MagicMock(spec=InterpretableModel)

#     # Create a Lime object
#     lime = Lime(explainable_model, interpretable_model)

#     # Assert that the explainable_model and interpretable_model attributes are set correctly
#     assert lime.explainable_model == explainable_model
#     assert lime.interpretable_model == interpretable_model
