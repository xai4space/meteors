import pytest

import torch
import numpy as np
from pydantic import ValidationError

import meteors.hsi as mt_image
from meteors import HSI

from meteors.exceptions import ShapeMismatchError, BandSelectionError

# Temporary solution for wavelengths
wavelengths_main = [
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


def test_get_channel_axis():
    # Test orientation ("C", "H", "W")
    orientation = ("C", "H", "W")
    expected_result = 0

    result = mt_image.get_channel_axis(orientation)
    assert result == expected_result

    # Test orientation ("H", "C", "W")
    orientation = ("H", "C", "W")
    expected_result = 1

    result = mt_image.get_channel_axis(orientation)
    assert result == expected_result

    # Test orientation ("W", "H", "C")
    orientation = ("W", "H", "C")
    expected_result = 2

    result = mt_image.get_channel_axis(orientation)
    assert result == expected_result


def test_validate_orientation():
    # Test valid orientation
    orientation = ("C", "H", "W")
    result = mt_image.validate_orientation(orientation)
    assert result == orientation

    # test conversion of string to tuple
    orientation = "CHW"
    result = mt_image.validate_orientation(orientation)
    assert result == ("C", "H", "W")

    # Test valid orientation with list
    orientation = ["C", "H", "W"]
    result = mt_image.validate_orientation(orientation)
    assert result == tuple(orientation)

    # Test invalid orientation with wrong length
    orientation = ("C", "H")
    with pytest.raises(ValueError):
        mt_image.validate_orientation(orientation)

    # Test invalid orientation with wrong elements
    orientation = ("C", "H", "A")
    with pytest.raises(ValueError):
        mt_image.validate_orientation(orientation)

    # test invalid orientation with repeated elements
    orientation = "HHH"
    with pytest.raises(ValueError):
        mt_image.validate_orientation(orientation)


def test_ensure_image_tensor():
    # Test ensure_image_tensor as numpy array
    image = np.random.rand(10, 10)
    result = mt_image.ensure_image_tensor(image)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([10, 10])

    # Test valid image as torch tensor
    image = torch.randn(5, 5)
    result = mt_image.ensure_image_tensor(image)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([5, 5])

    # Test invalid image type
    image = "invalid"
    with pytest.raises(TypeError):
        mt_image.ensure_image_tensor(image)

    # Test invalid image type
    image = 123
    with pytest.raises(TypeError):
        mt_image.ensure_image_tensor(image)


def test_resolve_inference_device_hsi():
    # Test device as string
    device = "cpu"
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    result = mt_image.resolve_inference_device_hsi(device, info)
    assert isinstance(result, torch.device)
    assert str(result) == device

    # Test device as torch.device
    device = torch.device("cpu")
    result = mt_image.resolve_inference_device_hsi(device, info)
    assert isinstance(result, torch.device)
    assert result == device

    # Test device as None
    device = None
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    result = mt_image.resolve_inference_device_hsi(device, info)
    assert isinstance(result, torch.device)
    assert result == info.data["image"].device

    # Test invalid device type
    device = 123
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    with pytest.raises(ValueError):
        mt_image.resolve_inference_device_hsi("device", info)

    # Test no image in the info
    device = None
    info = ValidationInfoMock(data={})
    with pytest.raises(RuntimeError):
        mt_image.resolve_inference_device_hsi(device, info)

    # Test wrong type device
    device = 0
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    with pytest.raises(TypeError):
        mt_image.resolve_inference_device_hsi(device, info)


def test_ensure_wavelengths_tensor():
    # Test valid wavelengths as torch tensor
    wavelengths = torch.tensor([400, 500, 600])
    result = mt_image.ensure_wavelengths_tensor(wavelengths)
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, wavelengths))

    # Test valid wavelengths as numpy array
    wavelengths = np.array([400, 500, 600])
    result = mt_image.ensure_wavelengths_tensor(wavelengths)
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor(wavelengths)))

    # Test valid wavelengths as list of integers
    wavelengths = [400, 500, 600]
    result = mt_image.ensure_wavelengths_tensor(wavelengths)
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor(wavelengths)))

    # Test valid wavelengths as tuple of floats
    wavelengths = (400.0, 500.0, 600.0)
    result = mt_image.ensure_wavelengths_tensor(wavelengths)
    assert isinstance(result, torch.Tensor)
    assert torch.all(torch.eq(result, torch.tensor(wavelengths)))

    # Test invalid wavelengths type
    wavelengths = "invalid"
    with pytest.raises(TypeError):
        mt_image.ensure_wavelengths_tensor(wavelengths)

    # Test invalid wavelengths type
    wavelengths = 123
    with pytest.raises(TypeError):
        mt_image.ensure_wavelengths_tensor(wavelengths)


def test_validate_shapes():
    # Test valid shapes
    wavelengths = torch.rand(10)
    image = torch.randn(5, 5, 10)
    band_axis = 2
    mt_image.validate_shapes(wavelengths, image, band_axis)  # No exception should be raised

    # Test invalid shapes
    wavelengths = torch.rand(10)
    image = torch.randn(5, 5, 10)
    band_axis = 1
    with pytest.raises(ShapeMismatchError):
        mt_image.validate_shapes(wavelengths, image, band_axis)

    wavelengths = torch.rand(10)
    image = torch.randn(5, 5, 5)
    band_axis = 2
    with pytest.raises(ShapeMismatchError):
        mt_image.validate_shapes(wavelengths, image, band_axis)


def test_process_and_validate_binary_mask():
    # Test valid binary mask as numpy array
    mask = np.random.randint(0, 2, size=(3, 5, 5))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.process_and_validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([3, 5, 5])

    # Test valid binary mask as torch tensor
    mask = torch.randint(0, 2, size=(3, 5, 5))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.process_and_validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([3, 5, 5])

    # Test valid binary mask as string 'artificial'
    mask = "artificial"
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.process_and_validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([3, 5, 5])
    assert torch.equal(result, torch.ones_like(info.data["image"]))

    # Test valid binary mask with dim image - 1
    mask = torch.ones(5, 5)
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.process_and_validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([3, 5, 5])
    assert torch.equal(result, mask.unsqueeze(0).repeat(3, 1, 1))

    # Test invalid binary mask type
    mask = 123
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(TypeError):
        mt_image.process_and_validate_binary_mask(mask, info)

    # Test invalid binary mask shape
    mask = np.random.randint(0, 2, size=(3, 5, 10))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(ShapeMismatchError):
        mt_image.process_and_validate_binary_mask(mask, info)

    # Test invalid binary mask shape
    mask = "Tymon where are you?"
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(ValueError):
        mt_image.process_and_validate_binary_mask(mask, info)

    # Test no image in the info
    mask = torch.ones(3, 5, 5)
    info = ValidationInfoMock(data={})
    with pytest.raises(RuntimeError):
        mt_image.process_and_validate_binary_mask(mask, info)


######################################################################
########################## IMAGE DATACLASS ###########################
######################################################################


def test_image():
    # Test valid image with default parameters
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    valid_image = HSI(image=sample, wavelengths=wavelengths)
    assert valid_image.image == sample
    assert valid_image.wavelengths == torch.tensor(wavelengths)
    assert valid_image.orientation == ("C", "H", "W")
    assert valid_image.binary_mask.shape == sample.shape
    assert all(
        [
            valid_image.device == sample.device,
            valid_image.binary_mask.device == sample.device,
            valid_image.image.device == sample.device,
        ]
    )

    # Test valid image with custom parameters
    orientation = ("C", "H", "W")
    binary_mask = "artificial"
    device = torch.device("cpu")
    valid_image = HSI(
        image=sample, wavelengths=wavelengths, orientation=orientation, binary_mask=binary_mask, device=device
    )
    assert valid_image.image == sample
    assert valid_image.wavelengths == torch.tensor(wavelengths)
    assert valid_image.orientation == orientation
    assert valid_image.binary_mask.shape == sample.shape
    assert all(
        [valid_image.device == device, valid_image.binary_mask.device == device, valid_image.image.device == device]
    )

    # Test invalid image type
    sample = "invalid"
    wavelengths = [0]
    with pytest.raises(TypeError):
        HSI(image=sample, wavelengths=wavelengths)

    # Test invalid wavelengths type
    sample = torch.tensor([[[0]]])
    wavelengths = "invalid"
    with pytest.raises(TypeError):
        HSI(image=sample, wavelengths=wavelengths)

    # Test invalid orientation type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    orientation = "invalid"
    with pytest.raises(RuntimeError):
        HSI(image=sample, wavelengths=wavelengths, orientation=orientation)

    # Test invalid binary mask type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    binary_mask = 123
    with pytest.raises(TypeError):
        HSI(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)

    # Test invalid device type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    device = "invalid"
    with pytest.raises(RuntimeError):
        HSI(image=sample, wavelengths=wavelengths, device=device)


def test_spectral_axis():
    # Test case with orientation ("C", "H", "W")
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    valid_image = HSI(image=sample, wavelengths=wavelengths)

    expected_result = 0
    result = valid_image.spectral_axis

    assert result == expected_result

    orientation = ("C", "H", "W")
    valid_image = HSI(image=sample, wavelengths=wavelengths, orientation=orientation)

    expected_result = 0
    result = valid_image.spectral_axis

    assert result == expected_result

    # Test case with orientation ("H", "C", "W")
    orientation = ("H", "C", "W")
    valid_image = HSI(image=sample, wavelengths=wavelengths, orientation=orientation)
    expected_result = 1

    result = valid_image.spectral_axis

    assert result == expected_result

    # Test case with orientation ("W", "H", "C")
    orientation = ("W", "H", "C")
    valid_image = HSI(image=sample, wavelengths=wavelengths, orientation=orientation)
    expected_result = 2

    result = valid_image.spectral_axis

    assert result == expected_result


def test_spatial_binary_mask():
    # Test with binary mask as None
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    binary_mask = None
    result = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask).spatial_binary_mask
    expected_result = torch.ones(5, 5, dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result))

    # Test with binary mask as numpy array
    binary_mask = np.random.randint(0, 2, size=(3, 5, 5))
    result = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask).spatial_binary_mask
    expected_result = torch.as_tensor(binary_mask[0, :, :], dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result))

    # Test with binary mask as torch tensor
    binary_mask = torch.randint(0, 2, size=(3, 5, 5))
    result = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask).spatial_binary_mask
    expected_result = torch.as_tensor(binary_mask[0, :, :], dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result))

    # Test with binary mask as string 'artificial'
    binary_mask = "artificial"
    result = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask).spatial_binary_mask
    expected_result = torch.ones(5, 5, dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result))


def test_validate_image_data():
    # Test valid image data
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    HSI(image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask)

    # Test invalid image data with mismatched wavelengths and image shape
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    with pytest.raises(ValidationError):
        HSI(image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask)

    # Test invalid image data with mismatched band axis
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("H", "C", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    with pytest.raises(ValidationError):
        HSI(image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask)

    # Test invalid mask shape
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 10)
    with pytest.raises(ValidationError):
        HSI(image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask)

    # Test invalid binary mask shape
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 6, 5)
    with pytest.raises(ValidationError):
        HSI(image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask)


def test_image_to():
    # Test moving image and binary mask to CPU device
    image = torch.randn(1, 5, 5)
    wavelengths = [0]
    device = torch.device("cpu")
    valid_image = HSI(image=image, wavelengths=wavelengths)
    result = valid_image.to(device)
    assert result.image.device == device
    assert result.binary_mask.device == device
    assert result.device == device

    # Test moving image and binary mask to CUDA device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        valid_image = HSI(image=image, wavelengths=wavelengths)
        result = valid_image.to(device)
        assert result.image.device == device
        assert result.binary_mask.device == device
        assert result.device == device

    # Test moving image and binary mask to a string device
    device = "cpu"
    valid_image = HSI(image=image, wavelengths=wavelengths)
    result = valid_image.to(device)
    assert result.image.device == torch.device(device)
    assert result.binary_mask.device == torch.device(device)
    assert result.device == torch.device(device)


def test_get_image():
    # Test with apply_mask=True and binary_mask is not None
    image = torch.randn((3, 5, 5))
    binary_mask = torch.ones((3, 5, 5), dtype=torch.bool)
    hsi_image = HSI(image=image, wavelengths=[0, 1, 2], binary_mask=binary_mask)
    result = hsi_image.get_image(apply_mask=True)
    assert torch.all(torch.eq(result, image))

    # Test with apply_mask=False and binary_mask is not None
    result = hsi_image.get_image(apply_mask=False)
    assert torch.all(torch.eq(result, image))

    # Test with apply_mask=True and binary_mask is None
    hsi_image = HSI(image=image, wavelengths=[0, 1, 2])
    result = hsi_image.get_image(apply_mask=True)
    assert torch.all(torch.eq(result, image))

    # Test with apply_mask=False and binary_mask is None
    result = hsi_image.get_image(apply_mask=False)
    assert torch.all(torch.eq(result, image))

    # Test with binary mask different:
    binary_mask[0, 0, 0] = False
    hsi_image = HSI(image=image, wavelengths=[0, 1, 2], binary_mask=binary_mask)
    result = hsi_image.get_image(apply_mask=True)
    assert not torch.all(torch.eq(result, image))
    assert torch.all(torch.eq(result[0, 0, 0], torch.tensor(0.0)))
    assert torch.all(torch.eq(result[1:, 1:, 1:], image[1:, 1:, 1:]))


def test__extract_central_slice_from_band():
    # Create a sample HSI object

    image = torch.randn(10, 5, 5)
    wavelengths = torch.arange(10)
    binary_mask = torch.ones(10, 5, 5, dtype=torch.bool)
    image_obj = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask)

    # Test selecting central band with mask and normalization
    selected_wavelengths = torch.tensor([3, 4, 5])
    expected_result = (image[4, :, :] - image[4, :, :].min()) / (image[4, :, :].max() - image[4, :, :].min())
    expected_result *= binary_mask[4, :, :]
    result = image_obj._extract_central_slice_from_band(
        selected_wavelengths, apply_mask=True, apply_min_cutoff=False, normalize=True
    )
    assert torch.allclose(result, expected_result)

    # Test selecting central band without mask and cutoff min
    selected_wavelengths = torch.tensor([7, 8, 9])
    expected_result = image[8, :, :]
    result = image_obj._extract_central_slice_from_band(
        selected_wavelengths, apply_mask=False, apply_min_cutoff=True, normalize=False
    )
    assert torch.allclose(result, expected_result)

    # Test selecting central band with mask and cutoff min
    selected_wavelengths = torch.tensor([1, 2, 3])
    expected_result = (image[2, :, :] - image[2, :, :].min()) / (image[2, :, :].max() - image[2, :, :].min())
    expected_result *= binary_mask[2, :, :]
    result = image_obj._extract_central_slice_from_band(
        selected_wavelengths, apply_mask=True, apply_min_cutoff=True, normalize=True
    )
    assert torch.allclose(result, expected_result)

    selected_wavelengths = [1, 90]
    with pytest.raises(ValueError):
        image_obj._extract_central_slice_from_band(selected_wavelengths)


def test_extract_band_by_name():
    # Test selecting a valid band using the center method
    image = torch.randn(len(wavelengths_main), 5, 5)
    wavelengths = wavelengths_main
    binary_mask = torch.ones(len(wavelengths_main), 5, 5, dtype=torch.bool)
    band_name = "R"
    method = "center"
    mask = True
    cutoff_min = False
    normalize = True

    hsi_image = HSI(image=image, wavelengths=wavelengths, binary_mask=binary_mask)
    result = hsi_image.extract_band_by_name(
        band_name=band_name, selection_method=method, apply_mask=mask, apply_min_cutoff=cutoff_min, normalize=normalize
    )

    assert result.shape == (5, 5)
    assert isinstance(result, torch.Tensor)

    # Test selecting an invalid band
    band_name = "InvalidBand"
    with pytest.raises(BandSelectionError):
        hsi_image.extract_band_by_name(
            band_name=band_name,
            selection_method=method,
            apply_mask=mask,
            apply_min_cutoff=cutoff_min,
            normalize=normalize,
        )

    # Test selecting a valid band using an unsupported method
    band_name = "R"
    method = "unsupported"
    with pytest.raises(NotImplementedError):
        hsi_image.extract_band_by_name(
            band_name=band_name,
            selection_method=method,
            apply_mask=mask,
            apply_min_cutoff=cutoff_min,
            normalize=normalize,
        )


def test_get_rgb_image():
    # Test with default settings
    image = torch.randn(len(wavelengths_main), 10, 10)
    wavelengths = wavelengths_main
    orientation = ("C", "H", "W")
    binary_mask = torch.ones(len(wavelengths_main), 10, 10, dtype=torch.bool)
    device = torch.device("cpu")
    test_image = HSI(
        image=image, wavelengths=wavelengths, orientation=orientation, binary_mask=binary_mask, device=device
    )

    result = test_image.get_rgb_image()

    assert result.shape == torch.Size([3, 10, 10])

    # Test with specific output band axis
    result = test_image.get_rgb_image(output_channel_axis=2)

    assert result.shape == torch.Size([10, 10, 3])

    # Test without applying a mask
    result = test_image.get_rgb_image(apply_mask=False)

    assert result.shape == torch.Size([3, 10, 10])


def test_orientation_change():
    tensor_image = torch.rand((4, 3, 2))
    image = HSI(image=tensor_image, wavelengths=[0, 1, 2, 3], orientation=("C", "H", "W"))

    assert image.orientation == ("C", "H", "W")

    # change of orientation with copy
    new_orientation = ("H", "W", "C")
    new_image = image.change_orientation(new_orientation, inplace=False)

    assert new_image.orientation == new_orientation
    assert new_image.image.shape == torch.Size([3, 2, 4])
    assert image.orientation == ("C", "H", "W")
    assert image.orientation != new_image.orientation

    # change of orientation inplace
    new_orientation = ("H", "C", "W")
    new_image = image.change_orientation(new_orientation, inplace=True)
    assert image.orientation == new_orientation
    assert image.image.shape == torch.Size([3, 4, 2])
    assert image == new_image

    # test the case where the orientation is the same
    new_orientation = ("H", "C", "W")
    new_image = image.change_orientation(new_orientation, inplace=True)
    assert image.orientation == new_orientation
    assert new_image == image

    # test case with binary mask
    tensor_image = torch.rand((4, 3, 2))
    binary_mask = torch.ones((4, 3, 2), dtype=torch.bool)
    image = HSI(image=tensor_image, wavelengths=[0, 1, 2, 3], orientation=("C", "H", "W"), binary_mask=binary_mask)

    assert image.orientation == ("C", "H", "W")

    image.change_orientation(new_orientation, inplace=True)
    assert image.orientation == new_orientation
    assert image.binary_mask.shape == torch.Size([3, 4, 2])

    # test case with invalid orientation
    new_orientation = ("H", "C", "A")
    with pytest.raises(ValueError):
        image.change_orientation(new_orientation, inplace=True)
