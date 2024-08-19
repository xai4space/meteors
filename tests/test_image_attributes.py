import pytest

import torch
import numpy as np


import meteors as mt
import meteors.attr as mt_attr


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


def test_ensure_torch_tensor():
    # Test with numpy array
    np_array = np.array([1, 2, 3])
    result = mt_attr.attributes.ensure_torch_tensor(np_array, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([1, 2, 3])))

    # Test with torch tensor
    torch_tensor = torch.tensor([4, 5, 6])
    result = mt_attr.attributes.ensure_torch_tensor(torch_tensor, "Input must be a numpy array or torch tensor")
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor([4, 5, 6])))

    # Test with invalid input type
    with pytest.raises(TypeError):
        mt_attr.attributes.ensure_torch_tensor("invalid", "Input must be a numpy array or torch tensor")


def test_validate_and_convert_attributes():
    # Test with numpy array
    attributes_np = np.ones((3, 4))
    attributes_torch = mt_attr.attributes.validate_and_convert_attributes(attributes_np)
    assert isinstance(attributes_torch, torch.Tensor)
    assert torch.all(attributes_torch.eq(torch.tensor(attributes_np)))

    # Test with torch tensor
    attributes_torch = torch.ones((3, 4))
    attributes_torch_validated = mt_attr.attributes.validate_and_convert_attributes(attributes_torch)
    assert isinstance(attributes_torch_validated, torch.Tensor)
    assert torch.all(attributes_torch_validated.eq(attributes_torch))

    # Test with invalid type
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_attributes(123)


def test_validate_and_convert_segmentation_mask():
    # Test case 1: Valid segmentation mask (numpy array)
    segmentation_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_attr.attributes.validate_and_convert_segmentation_mask(segmentation_mask_np)
    assert isinstance(result_np, torch.Tensor)
    assert torch.all(torch.eq(result_np, torch.tensor(segmentation_mask_np)))

    # Test case 2: Valid segmentation mask (torch tensor)
    segmentation_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_attr.attributes.validate_and_convert_segmentation_mask(segmentation_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.all(torch.eq(result_tensor, segmentation_mask_tensor))

    # Test case 3: Invalid segmentation mask (wrong type)
    invalid_segmentation_mask = "invalid"
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_segmentation_mask(invalid_segmentation_mask)

    # Test case 4: Invalid segmentation mask (wrong type)
    invalid_segmentation_mask = 123
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_segmentation_mask(invalid_segmentation_mask)


def test_validate_and_convert_band_mask():
    # Test case 1: Valid band mask (numpy array)
    band_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_attr.attributes.validate_and_convert_band_mask(band_mask_np)
    assert isinstance(result_np, torch.Tensor)
    assert torch.all(torch.eq(result_np, torch.tensor(band_mask_np)))

    # Test case 2: Valid band mask (torch tensor)
    band_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_attr.attributes.validate_and_convert_band_mask(band_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.all(torch.eq(result_tensor, band_mask_tensor))

    # Test case 3: Invalid band mask (wrong type)
    invalid_band_mask = "invalid"
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_band_mask(invalid_band_mask)

    # Test case 4: Invalid band mask (wrong type)
    invalid_band_mask = 123
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_band_mask(invalid_band_mask)


def test_validate_shapes():
    shape = (len(wavelengths), 240, 240)
    attributes = torch.ones(shape)
    image = mt.Image(image=torch.ones(shape), wavelengths=wavelengths)

    # Test case 1: Valid shapes
    mt_attr.attributes.validate_shapes(attributes, image)  # No exception should be raised

    # Test case 2: Invalid shapes
    invalid_attributes = torch.ones((150, 240, 241))
    with pytest.raises(ValueError):
        mt_attr.attributes.validate_shapes(invalid_attributes, image)

    invalid_image = mt.Image(image=torch.ones((len(wavelengths), 240, 241)), wavelengths=wavelengths)
    with pytest.raises(ValueError):
        mt_attr.attributes.validate_shapes(attributes, invalid_image)


def test_align_band_names_with_mask():
    # Test case 1: Changed band names
    band_names = {"R": 1, "G": 2, "B": 3}
    band_mask = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 3]])

    with pytest.warns(UserWarning):
        updated_band_names = mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    assert updated_band_names == {
        "R": 1,
        "G": 2,
        "B": 3,
        "not_included": 0,
    }

    # Test case 2: Not changed band names
    band_names = {"R": 0, "G": 1, "B": 2}
    band_mask = torch.tensor([[1, 0, 1], [1, 2, 1], [1, 0, 1]])

    not_updated_band_names = mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {
        "R": 0,
        "G": 1,
        "B": 2,
    }

    band_names = {"R": 0}
    band_mask = torch.tensor([[0], [0], [0]])

    not_updated_band_names = mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    assert not_updated_band_names == {"R": 0}

    # Test case 3: Invalid band names
    band_names = {"R": 1, "G": 2, "B": 3}
    invalid_band_mask = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        mt_attr.attributes.align_band_names_with_mask(band_names, invalid_band_mask)


def test_validate_image_attributions():
    # Create a sample ImageAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    image_attributes = mt_attr.ImageAttributes(
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
        mt_attr.ImageAttributes(
            image=image, attributes=invalid_attributes, score=score, device=device, model_config=model_config
        )

    # Not implemented yet functions
    with pytest.raises(NotImplementedError):
        image_attributes.flattened_attributes


######################################################################
############################ EXPLANATIONS ############################
######################################################################


def test_spatial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt_attr.ImageSpatialAttributes(
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
        mt_attr.ImageSpatialAttributes(
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
    spatial_attributes = mt_attr.ImageSpatialAttributes(
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


def test_spatial_segmentation_mask_spacial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt_attr.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    assert spatial_attributes.spatial_segmentation_mask.shape == segmentation_mask.shape[1:]


def test_flattened_attributes_spacial_attributes():
    # Create a sample ImageSpatialAttributes object
    image = mt.Image(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    spatial_attributes = mt_attr.ImageSpatialAttributes(
        image=image,
        attributes=attributes,
        score=score,
        segmentation_mask=segmentation_mask,
        device=device,
        model_config=model_config,
    )

    assert spatial_attributes.flattened_attributes.shape == segmentation_mask.shape[1:]


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
    spectral_attributes = mt_attr.ImageSpectralAttributes(
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
        mt_attr.ImageSpectralAttributes(
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
    spectral_attributes = mt_attr.ImageSpectralAttributes(
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
    spectral_attributes = mt_attr.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spectral_attributes.spectral_band_mask, torch.tensor([0, 1, 2]))


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
    spectral_attributes = mt_attr.ImageSpectralAttributes(
        image=image,
        attributes=attributes,
        score=score,
        band_names=band_names,
        band_mask=band_mask,
        device=device,
        model_config=model_config,
    )

    assert torch.equal(spectral_attributes.flattened_attributes, torch.tensor([0, 1, 2]))
