import pytest

import torch
import numpy as np
from pydantic import ValidationError


from meteors import HSI
import meteors.attr as mt_attr
from meteors.attr import HSIAttributes, HSISpatialAttributes, HSISpectralAttributes
from meteors.exceptions import ShapeMismatchError, HSIAttributesError, MaskCreationError


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


def test_validate_and_convert_mask():
    # Test case 1: Valid mask (numpy array)
    segmentation_mask_np = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_np = mt_attr.attributes.validate_and_convert_mask(segmentation_mask_np)
    assert isinstance(result_np, torch.Tensor)
    assert torch.all(torch.eq(result_np, torch.tensor(segmentation_mask_np)))

    # Test case 2: Valid mask (torch tensor)
    segmentation_mask_tensor = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result_tensor = mt_attr.attributes.validate_and_convert_mask(segmentation_mask_tensor)
    assert isinstance(result_tensor, torch.Tensor)
    assert torch.all(torch.eq(result_tensor, segmentation_mask_tensor))

    # Test case 3: Invalid mask (wrong type)
    invalid_segmentation_mask = "invalid"
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_mask(invalid_segmentation_mask)

    # Test case 4: Invalid mask (wrong type)
    invalid_segmentation_mask = 123
    with pytest.raises(TypeError):
        mt_attr.attributes.validate_and_convert_mask(invalid_segmentation_mask)

    # Test case 5: None mask
    segmentation_mask_none = None
    result_none = mt_attr.attributes.validate_and_convert_mask(segmentation_mask_none)
    assert result_none is None


def test_validate_shapes():
    shape = (len(wavelengths), 240, 240)
    attributes = torch.ones(shape)
    image = HSI(image=torch.ones(shape), wavelengths=wavelengths)

    # Test case 1: Valid shapes
    mt_attr.attributes.validate_shapes(attributes, image)  # No exception should be raised

    # Test case 2: Invalid shapes
    invalid_attributes = torch.ones((150, 240, 241))
    with pytest.raises(ShapeMismatchError):
        mt_attr.attributes.validate_shapes(invalid_attributes, image)

    invalid_image = HSI(image=torch.ones((len(wavelengths), 240, 241)), wavelengths=wavelengths)
    with pytest.raises(ShapeMismatchError):
        mt_attr.attributes.validate_shapes(attributes, invalid_image)


def test_align_band_names_with_mask():
    # Test case 1: Changed band names
    band_names = {"R": 1, "G": 2, "B": 3}
    band_mask = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 3]])

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

    # Test case 3: One not included band name
    band_names = {"R": 0}
    band_mask = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]])

    updated_band_names = mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    assert updated_band_names == {
        "R": 0,
        "not_included": 1,
    }

    # Test case 4: Invalid band names
    band_names = {"R": 0}
    invalid_band_mask = torch.tensor([[0, 1, 0], [0, 2, 0], [0, 0, 0]])
    with pytest.raises(MaskCreationError):
        mt_attr.attributes.align_band_names_with_mask(band_names, invalid_band_mask)

    # Test case 5:  band names to much bands
    band_names = {"R": 0, "G": 1}
    band_mask = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    changed_band_names = mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    assert changed_band_names == {"R": 0}

    with pytest.raises(MaskCreationError):
        band_names = {"R": 0, "G": 1}
        band_mask = torch.tensor([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
        mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)

    # Test case 6: `not_included` already in band names but there is not covered bands
    band_names = {"R": 0, "not_included": 1}
    band_mask = torch.tensor([[0, 1, 0], [0, 2, 0], [0, 0, 0]])
    with pytest.raises(MaskCreationError):
        mt_attr.attributes.align_band_names_with_mask(band_names, band_mask)


def test_validate_hsi_attributions():
    # Create a sample HSIAttributes object
    image = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    score = 0.8
    device = torch.device("cpu")
    model_config = {"param1": 1, "param2": 2}
    image_attributes = HSIAttributes(
        hsi=image,
        attributes=attributes,
        score=score,
        attribution_method="Lime",
        device=device,
        model_config=model_config,
    )

    # Assert that the attributes tensor has been moved to the specified device
    assert image_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert image_attributes.hsi.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert image_attributes.attributes.shape == image_attributes.hsi.image.shape

    # Assert that the device of the image object has been updated
    assert image_attributes.hsi.device == device

    # Validate invalid shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValidationError):
        HSIAttributes(
            hsi=image,
            attributes=invalid_attributes,
            score=score,
            device=device,
            model_config=model_config,
            attribution_method="lime",
        )

    # Not implemented yet functions
    with pytest.raises(NotImplementedError):
        image_attributes.flattened_attributes


def test_resolve_inference_device_attributes():
    # Test device as string
    device = "cpu"
    hsi = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])

    info = ValidationInfoMock(data={"hsi": hsi})
    result = mt_attr.attributes.resolve_inference_device_attributes(device, info)
    assert isinstance(result, torch.device)
    assert str(result) == device

    # Test device as torch.device
    device = torch.device("cpu")
    result = mt_attr.attributes.resolve_inference_device_attributes(device, info)
    assert isinstance(result, torch.device)
    assert result == device

    # Test device as None
    device = None
    info = ValidationInfoMock(data={"hsi": hsi})
    result = mt_attr.attributes.resolve_inference_device_attributes(device, info)
    assert isinstance(result, torch.device)
    assert result == info.data["hsi"].device

    # Test invalid device type
    device = 123
    info = ValidationInfoMock(data={"hsi": hsi})
    with pytest.raises(ValueError):
        mt_attr.attributes.resolve_inference_device_attributes("device", info)

    # Test no image in the info
    device = None
    info = ValidationInfoMock(data={})
    with pytest.raises(HSIAttributesError):
        mt_attr.attributes.resolve_inference_device_attributes(device, info)

    # Test wrong type device
    device = 0
    info = ValidationInfoMock(data={"hsi": hsi})
    with pytest.raises(TypeError):
        mt_attr.attributes.resolve_inference_device_attributes(device, info)


def test_validate_attribution_method():
    # Test valid attribution method
    for method in mt_attr.attributes.AVAILABLE_ATTRIBUTION_METHODS:
        method_new = mt_attr.attributes.validate_attribution_method(method)  # No exception should be raised
        assert method == method_new

    method = mt_attr.attributes.validate_attribution_method(None)  # No exception should be raised
    assert method is None

    # this should raise a loguru warning, but pass
    method = mt_attr.attributes.validate_attribution_method("invalid")


######################################################################
############################ EXPLANATIONS ############################
######################################################################


def test_attributes():
    # Create a sample HSIAttributes object
    image = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    torch_attributes = torch.ones((3, 4, 4))
    device = torch.device("cpu")
    mask = torch.randint(0, 2, (3, 4, 4))
    attributes = HSIAttributes(
        hsi=image, attributes=torch_attributes, score=0.8, mask=mask, attribution_method="Lime", device=device
    )

    assert attributes.hsi == image
    assert torch.equal(attributes.attributes, torch.ones((3, 4, 4)))
    assert torch.equal(attributes.mask, mask)
    assert attributes.hsi.device == device
    assert attributes.attributes.device == device
    assert attributes.mask.device == device
    assert attributes.score == 0.8
    assert attributes.attribution_method == "Lime"
    assert attributes.device == device

    with pytest.raises(NotImplementedError):
        attributes.flattened_attributes

    # Not valid attribution shape
    with pytest.raises(ValidationError):
        HSIAttributes(
            hsi=image,
            attributes=torch.ones((1, 4, 4)),
            score=0.8,
            mask=torch.randint(0, 2, (3, 4, 4)),
            attribution_method="Lime",
            device=device,
        )

    # Not valid mask shape
    with pytest.raises(ValidationError):
        HSIAttributes(
            hsi=image,
            attributes=torch.ones((3, 4, 4)),
            score=0.8,
            mask=torch.randint(0, 2, (1, 4, 4)),
            attribution_method="Lime",
            device=device,
        )

    # No mask passed
    no_mask_attributes = HSIAttributes(
        hsi=image, attributes=torch_attributes, score=0.8, attribution_method="Lime", device=device
    )
    assert no_mask_attributes.mask is None

    # test to method
    attributes.to("cpu")
    assert attributes.device == torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        attributes.to(device)
        assert attributes.device == device
        assert attributes.hsi.device == device
        assert attributes.attributes.device == device
        assert attributes.mask.device == device


def test_validate_score():
    # Test valid score
    hsi = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    score = 0.8

    HSIAttributes(
        hsi=hsi,
        attributes=torch.ones((3, 4, 4)),
        score=score,
        mask=torch.randint(0, 2, (3, 4, 4)),
        attribution_method="Lime",
    )

    # No score
    HSIAttributes(
        hsi=hsi,
        attributes=torch.ones((3, 4, 4)),
        score=score,
        mask=torch.randint(0, 2, (3, 4, 4)),
        attribution_method="Lime",
    )


def test_change_orientation_attributes():
    image = HSI(image=torch.ones((3, 4, 5)), wavelengths=[400, 500, 600], orientation=("C", "H", "W"))
    attributes = torch.ones((3, 4, 5))
    device = torch.device("cpu")
    attrs = HSIAttributes(
        hsi=image,
        attributes=attributes,
        score=0.8,
        mask=torch.randint(0, 2, (3, 4, 5)),
        attribution_method="Lime",
        device=device,
    )

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


def test_spatial_attributes():
    # Create a sample HSISpatialAttributes object
    image = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    segmentation_mask = torch.randint(0, 2, (3, 4, 4))
    device = torch.device("cpu")
    spatial_attributes = HSISpatialAttributes(hsi=image, attributes=attributes, mask=segmentation_mask, device=device)

    assert spatial_attributes.hsi == image
    assert torch.equal(spatial_attributes.attributes, attributes)
    assert torch.equal(spatial_attributes.mask, segmentation_mask)

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

    # Assert segmentation mask
    assert spatial_attributes.segmentation_mask.ndim == 2
    assert spatial_attributes.segmentation_mask.shape == (4, 4)

    # Assert flattened attributes
    assert spatial_attributes.flattened_attributes.shape == (4, 4)
    assert torch.equal(spatial_attributes.flattened_attributes, torch.ones((4, 4)))
    assert spatial_attributes.flattened_attributes.shape == spatial_attributes.segmentation_mask.shape

    # not valid attributes shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValidationError):
        HSISpatialAttributes(
            hsi=image,
            attributes=invalid_attributes,
            mask=segmentation_mask,
            device=device,
        )

    # Not valid segmentation mask shape
    with pytest.raises(ValidationError):
        HSISpatialAttributes(
            hsi=image,
            attributes=attributes,
            mask=torch.randint(0, 2, (1, 4, 4)),
            device=device,
        )

    # no segmentation mask passed
    with pytest.raises(HSIAttributesError):
        attributes = HSISpatialAttributes(
            hsi=image,
            attributes=attributes,
            device=device,
        )
        attributes.segmentation_mask

    # test to method
    spatial_attributes.to("cpu")
    assert spatial_attributes.device == torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        spatial_attributes.to(device)
        assert spatial_attributes.device == device
        assert spatial_attributes.hsi.device == device
        assert spatial_attributes.attributes.device == device
        assert spatial_attributes.segmentation_mask.device == device

    with pytest.raises(HSIAttributesError):
        HSISpatialAttributes(
            hsi=image,
            attributes=attributes,
            mask=None,
            device=device,
        )

    with pytest.raises(HSIAttributesError):
        attributes_with_no_mask = HSISpatialAttributes(
            hsi=image,
            attributes=attributes,
            mask=segmentation_mask,
            device=device,
        )
        attributes_with_no_mask.mask = None
        attributes_with_no_mask.segmentation_mask


def test_spectral_attributes():
    # Create a sample HSISpectralAttributes object
    image = HSI(image=torch.ones((3, 4, 4)), wavelengths=[400, 500, 600])
    attributes = torch.ones((3, 4, 4))
    band_mask = torch.empty_like(attributes)
    band_mask[0] = 0
    band_mask[1] = 1
    band_mask[2] = 2
    band_names = {"R": 0, "G": 1, "B": 2}
    device = torch.device("cpu")
    spectral_attributes = HSISpectralAttributes(
        hsi=image,
        attributes=attributes,
        band_names=band_names,
        mask=band_mask,
        device=device,
    )

    assert spectral_attributes.hsi == image
    assert torch.equal(spectral_attributes.attributes, attributes)
    assert torch.equal(spectral_attributes.band_mask, band_mask[:, 0, 0])
    assert spectral_attributes.band_names == band_names

    # Assert that the attributes tensor has been moved to the specified device
    assert spectral_attributes.attributes.device == device

    # Assert that the image tensor has been moved to the specified device
    assert spectral_attributes.hsi.device == device

    # Assert that the segmentation mask tensor has been moved to the specified device
    assert spectral_attributes.band_mask.device == device

    # Assert that the shapes of the attributes and image tensors match
    assert spectral_attributes.attributes.shape == spectral_attributes.hsi.image.shape

    # Assert that the device of the image object has been updated
    assert spectral_attributes.hsi.device == device

    # Assert band mask
    assert spectral_attributes.band_mask.ndim == 1
    assert spectral_attributes.band_mask.shape == (3,)
    assert torch.equal(spectral_attributes.band_mask, torch.tensor([0, 1, 2]))

    # Assert flattened attributes
    assert spectral_attributes.flattened_attributes.shape == (3,)
    assert torch.equal(spectral_attributes.flattened_attributes, torch.ones((3,)))
    assert spectral_attributes.flattened_attributes.shape == spectral_attributes.band_mask.shape

    # not valid attributes shape
    invalid_attributes = torch.ones((1, 4, 4))
    with pytest.raises(ValidationError):
        HSISpectralAttributes(
            hsi=image,
            attributes=invalid_attributes,
            band_names=band_names,
            mask=band_mask,
            device=device,
        )

    # Not valid band mask shape
    with pytest.raises(ValidationError):
        HSISpectralAttributes(
            hsi=image,
            attributes=attributes,
            band_names=band_names,
            mask=torch.randint(0, 2, (3, 4, 5)),
            device=device,
        )

    # no band mask passed
    with pytest.raises(HSIAttributesError):
        attributes = HSISpectralAttributes(
            hsi=image,
            attributes=attributes,
            band_names=band_names,
            device=device,
        )
        attributes.band_mask

    # test to method
    spectral_attributes.to("cpu")
    assert spectral_attributes.device == torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        spectral_attributes.to(device)
        assert spectral_attributes.device == device
        assert spectral_attributes.hsi.device == device
        assert spectral_attributes.attributes.device == device
        assert spectral_attributes.band_mask.device == device

    with pytest.raises(HSIAttributesError):
        HSISpectralAttributes(
            hsi=image,
            attributes=attributes,
            band_names=band_names,
            mask=None,
            device=device,
        )

    with pytest.raises(ValueError):
        attributes_with_no_mask = HSISpectralAttributes(
            hsi=image,
            attributes=attributes,
            band_names=band_names,
            mask=band_mask,
            device=device,
        )
        attributes_with_no_mask.mask = None
        attributes_with_no_mask.band_mask
