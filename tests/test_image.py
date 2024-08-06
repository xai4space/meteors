import pytest

import torch
import numpy as np
from pydantic import ValidationError

import meteors.image as mt_image


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


def test_validate_orientation():
    # Test valid orientation
    orientation = ("C", "H", "W")
    result = mt_image.validate_orientation(orientation)
    assert result == orientation, "Valid orientation should not raise an exception"

    # Test valid orientation with list
    orientation = ["C", "H", "W"]
    result = mt_image.validate_orientation(orientation)
    assert result == tuple(orientation), "Valid orientation should not raise an exception"

    # Test invalid orientation with wrong length
    orientation = ("C", "H")
    with pytest.raises(ValueError):
        mt_image.validate_orientation(orientation)

    # Test invalid orientation with wrong elements
    orientation = ("C", "H", "A")
    with pytest.raises(ValueError):
        mt_image.validate_orientation(orientation)


def test_validate_image():
    # Test valid image as numpy array
    image = np.random.rand(10, 10)
    result = mt_image.validate_image(image)
    assert isinstance(result, torch.Tensor), "Image should be converted to torch tensor"
    assert result.shape == torch.Size([10, 10]), "Image shape should be preserved"

    # Test valid image as torch tensor
    image = torch.randn(5, 5)
    result = mt_image.validate_image(image)
    assert isinstance(result, torch.Tensor), "Image should remain as torch tensor"
    assert result.shape == torch.Size([5, 5]), "Image shape should be preserved"

    # Test invalid image type
    image = "invalid"
    with pytest.raises(TypeError):
        mt_image.validate_image(image)

    # Test invalid image type
    image = 123
    with pytest.raises(TypeError):
        mt_image.validate_image(image)


def test_validate_device():
    # Test device as string
    device = "cpu"
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    result = mt_image.validate_device(device, info)
    assert isinstance(result, torch.device), "Device should be a torch device"
    assert str(result) == device, "Device should match the input string"

    # Test device as torch.device
    device = torch.device("cpu")
    result = mt_image.validate_device(device, info)
    assert isinstance(result, torch.device), "Device should remain as torch device"
    assert result == device, "Device should match the input torch device"

    # Test device as None
    device = None
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    result = mt_image.validate_device(device, info)
    assert isinstance(result, torch.device), "Device should be a torch device"
    assert result == info.data["image"].device, "Device should match the device of the input image"

    # Test invalid device type
    device = 123
    info = ValidationInfoMock(data={"image": torch.randn(5, 5)})
    with pytest.raises(TypeError):
        mt_image.validate_device(device, info)

    # Test no image in the info
    device = None
    info = ValidationInfoMock(data={})
    with pytest.raises(ValueError):
        mt_image.validate_device(device, info)


def test_validate_wavelengths():
    # Test valid wavelengths as torch tensor
    wavelengths = torch.tensor([400, 500, 600])
    result = mt_image.validate_wavelengths(wavelengths)
    assert isinstance(result, torch.Tensor), "Wavelengths should remain as torch tensor"
    assert torch.all(torch.eq(result, wavelengths)), "Wavelengths should be preserved"

    # Test valid wavelengths as numpy array
    wavelengths = np.array([400, 500, 600])
    result = mt_image.validate_wavelengths(wavelengths)
    assert isinstance(result, torch.Tensor), "Wavelengths should be converted to torch tensor"
    assert torch.all(torch.eq(result, torch.tensor(wavelengths))), "Wavelengths should be preserved"

    # Test valid wavelengths as list of integers
    wavelengths = [400, 500, 600]
    result = mt_image.validate_wavelengths(wavelengths)
    assert isinstance(result, torch.Tensor), "Wavelengths should be converted to torch tensor"
    assert torch.all(torch.eq(result, torch.tensor(wavelengths))), "Wavelengths should be preserved"

    # Test valid wavelengths as tuple of floats
    wavelengths = (400.0, 500.0, 600.0)
    result = mt_image.validate_wavelengths(wavelengths)
    assert isinstance(result, torch.Tensor), "Wavelengths should be converted to torch tensor"
    assert torch.all(torch.eq(result, torch.tensor(wavelengths))), "Wavelengths should be preserved"

    # Test invalid wavelengths type
    wavelengths = "invalid"
    with pytest.raises(TypeError):
        mt_image.validate_wavelengths(wavelengths)

    # Test invalid wavelengths type
    wavelengths = 123
    with pytest.raises(TypeError):
        mt_image.validate_wavelengths(wavelengths)


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
    with pytest.raises(ValueError):
        mt_image.validate_shapes(wavelengths, image, band_axis)

    wavelengths = torch.rand(10)
    image = torch.randn(5, 5, 5)
    band_axis = 2
    with pytest.raises(ValueError):
        mt_image.validate_shapes(wavelengths, image, band_axis)


def test_validate_binary_mask():
    # Test valid binary mask as numpy array
    mask = np.random.randint(0, 2, size=(3, 5, 5))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor), "Binary mask should be converted to torch tensor"
    assert result.shape == torch.Size([3, 5, 5]), "Binary mask shape should be preserved"

    # Test valid binary mask as torch tensor
    mask = torch.randint(0, 2, size=(3, 5, 5))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor), "Binary mask should remain as torch tensor"
    assert result.shape == torch.Size([3, 5, 5]), "Binary mask shape should be preserved"

    # Test valid binary mask as string 'artificial'
    mask = "artificial"
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    result = mt_image.validate_binary_mask(mask, info)
    assert isinstance(result, torch.Tensor), "Binary mask should be converted to torch tensor"
    assert result.shape == torch.Size([3, 5, 5]), "Binary mask shape should be preserved"
    assert torch.equal(result, torch.ones_like(info.data["image"])), "The simplest mask with no data should be created"
    # TODO: Fersoil: Check if the mask is binary

    # Test invalid binary mask type
    mask = 123
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(ValueError):
        mt_image.validate_binary_mask(mask, info)

    # Test invalid binary mask shape
    mask = np.random.randint(0, 2, size=(3, 5, 10))
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(ValueError):
        mt_image.validate_binary_mask(mask, info)

    # Test invalid binary mask shape
    mask = "Tymon where are you?"
    info = ValidationInfoMock(
        data={"image": torch.randn(3, 5, 5), "orientation": ("C", "H", "W"), "device": torch.device("cpu")}
    )
    with pytest.raises(ValueError):
        mt_image.validate_binary_mask(mask, info)

    # Test no image in the info
    mask = torch.ones(3, 5, 5)
    info = ValidationInfoMock(data={})
    with pytest.raises(ValueError):
        mt_image.validate_binary_mask(mask, info)


######################################################################
########################## IMAGE DATACLASS ###########################
######################################################################


def test_image():
    # Test valid image with default parameters
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    valid_image = mt_image.Image(image=sample, wavelengths=wavelengths)
    assert valid_image.image == sample, "Sample should be preserved"
    assert valid_image.wavelengths == torch.tensor(wavelengths), "Wavelengths should be preserved"
    assert valid_image.orientation == ("C", "H", "W"), "Orientation should be set to default"
    assert (
        valid_image.binary_mask.shape == sample.shape
    ), "Binary mask should be created with the same shape as the input image"
    assert all(
        [
            valid_image.device == sample.device,
            valid_image.binary_mask.device == sample.device,
            valid_image.image.device == sample.device,
        ]
    ), "Device should be set to the device of the input image"

    # Test valid image with custom parameters
    orientation = ("C", "H", "W")
    binary_mask = "artificial"
    device = torch.device("cpu")
    valid_image = mt_image.Image(
        image=sample, wavelengths=wavelengths, orientation=orientation, binary_mask=binary_mask, device=device
    )
    assert valid_image.image == sample, "Sample should be preserved"
    assert valid_image.wavelengths == torch.tensor(wavelengths), "Wavelengths should be preserved"
    assert valid_image.orientation == orientation, "Orientation should be set to the custom value"
    assert (
        valid_image.binary_mask.shape == sample.shape
    ), "Binary mask should be created with the same shape as the input image"
    assert all(
        [valid_image.device == device, valid_image.binary_mask.device == device, valid_image.image.device == device]
    ), "Device should be set to the custom device"

    # Test invalid image type
    sample = "invalid"
    wavelengths = [0]
    with pytest.raises(TypeError):
        mt_image.Image(image=sample, wavelengths=wavelengths)

    # Test invalid wavelengths type
    sample = torch.tensor([[[0]]])
    wavelengths = "invalid"
    with pytest.raises(TypeError):
        mt_image.Image(image=sample, wavelengths=wavelengths)

    # Test invalid orientation type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    orientation = "invalid"
    with pytest.raises(ValueError):
        mt_image.Image(image=sample, wavelengths=wavelengths, orientation=orientation)

    # Test invalid binary mask type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    binary_mask = 123
    with pytest.raises(ValueError):
        mt_image.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)

    # Test invalid device type
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    device = "invalid"
    with pytest.raises(ValueError):
        mt_image.Image(image=sample, wavelengths=wavelengths, device=device)


def test_get_band_axis():
    # Test case with orientation ("C", "H", "W")
    orientation = ("C", "H", "W")
    expected_result = 0

    result = mt_image.get_band_axis(orientation)

    assert result == expected_result, "Incorrect band axis index returned"

    # Test case with orientation ("H", "C", "W")
    orientation = ("H", "C", "W")
    expected_result = 1

    result = mt_image.get_band_axis(orientation)

    assert result == expected_result, "Incorrect band axis index returned"

    # Test case with orientation ("W", "H", "C")
    orientation = ("W", "H", "C")
    expected_result = 2

    result = mt_image.get_band_axis(orientation)

    assert result == expected_result, "Incorrect band axis index returned"


def test_get_squeezed_binary_mask():
    # Test with binary mask as None
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    binary_mask = None
    result = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask).get_squeezed_binary_mask
    expected_result = torch.ones(5, 5, dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result)), "Incorrect squeezed binary mask"

    # Test with binary mask as numpy array
    binary_mask = np.random.randint(0, 2, size=(3, 5, 5))
    result = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask).get_squeezed_binary_mask
    expected_result = torch.as_tensor(binary_mask[0, :, :], dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result)), "Incorrect squeezed binary mask"

    # Test with binary mask as torch tensor
    binary_mask = torch.randint(0, 2, size=(3, 5, 5))
    result = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask).get_squeezed_binary_mask
    expected_result = torch.as_tensor(binary_mask[0, :, :], dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result)), "Incorrect squeezed binary mask"

    # Test with binary mask as string 'artificial'
    binary_mask = "artificial"
    result = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask).get_squeezed_binary_mask
    expected_result = torch.ones(5, 5, dtype=torch.bool)
    assert torch.all(torch.eq(result, expected_result))


def test_validate_image_data():
    # Test valid image data
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    data = mt_image.Image(
        image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask
    )
    result = data.validate_image_data()
    assert result == data, "Valid image data should not raise an exception"

    # Test invalid image data with mismatched wavelengths and image shape
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500])
    orientation = ("C", "H", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    with pytest.raises(ValidationError):
        data = mt_image.Image(
            image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask
        )

    # Test invalid image data with mismatched band axis
    image = torch.randn(3, 5, 5)
    wavelengths = torch.tensor([400, 500, 600])
    orientation = ("H", "C", "W")
    device = torch.device("cpu")
    binary_mask = torch.ones(3, 5, 5)
    with pytest.raises(ValidationError):
        data = mt_image.Image(
            image=image, wavelengths=wavelengths, orientation=orientation, device=device, binary_mask=binary_mask
        )


def test_image_to():
    # Test moving image and binary mask to CPU device
    image = torch.randn(1, 5, 5)
    wavelengths = [0]
    device = torch.device("cpu")
    valid_image = mt_image.Image(image=image, wavelengths=wavelengths)
    result = valid_image.to(device)
    assert result.image.device == device, "Image should be moved to the specified device"
    assert result.binary_mask.device == device, "Binary mask should be moved to the specified device"
    assert result.device == device, "Device attribute should be updated"

    # Test moving image and binary mask to CUDA device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        valid_image = mt_image.Image(image=image, wavelengths=wavelengths)
        result = valid_image.to(device)
        assert result.image.device == device, "Image should be moved to the specified device"
        assert result.binary_mask.device == device, "Binary mask should be moved to the specified device"
        assert result.device == device, "Device attribute should be updated"

    # Test moving image and binary mask to a string device
    device = "cpu"
    valid_image = mt_image.Image(image=image, wavelengths=wavelengths)
    result = valid_image.to(device)
    assert result.image.device == torch.device(device), "Image should be moved to the specified device"
    assert result.binary_mask.device == torch.device(device), "Binary mask should be moved to the specified device"
    assert result.device == torch.device(device), "Device attribute should be updated"


# def test_get_central_band(): # TODO: #Fersoil
#     # Create a sample Image object
#     image = torch.randn(10, 5, 5)
#     wavelengths = torch.arange(10)
#     binary_mask = torch.ones(10, 5, 5, dtype=torch.bool)
#     image_obj = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask)

#     # Test selecting central band with mask and normalization
#     selected_wavelengths = torch.tensor([3, 4, 5])
#     expected_result = (image[3:6, :, :] - image[3:6, :, :].min()) / (image[3:6, :, :].max() - image[3:6, :, :].min())
#     expected_result *= binary_mask[3:6, :, :]
#     result = image_obj._get_central_band(selected_wavelengths, mask=True, cutoff_min=False, normalize=True)
#     assert torch.allclose(result, expected_result), "Incorrect central band with mask and normalization"

#     # Test selecting central band without mask and cutoff min
#     selected_wavelengths = torch.tensor([7, 8, 9])
#     expected_result = image[7:10, :, :]
#     result = image_obj._get_central_band(selected_wavelengths, mask=False, cutoff_min=True, normalize=False)
#     assert torch.allclose(result, expected_result), "Incorrect central band without mask and cutoff min"

#     # Test selecting central band with mask and cutoff min
#     selected_wavelengths = torch.tensor([1, 2, 3])
#     expected_result = (image[1:4, :, :] - image[1:4, :, :].min()) / (image[1:4, :, :].max() - image[1:4, :, :].min())
#     expected_result *= binary_mask[1:4, :, :]
#     result = image_obj._get_central_band(selected_wavelengths, mask=True, cutoff_min=True, normalize=True)
#     assert torch.allclose(result, expected_result), "Incorrect central band with mask and cutoff min"


def test_select_single_band_from_name():
    # Test selecting a valid band using the center method
    image = torch.randn(len(wavelengths_main), 5, 5)
    wavelengths = wavelengths_main
    binary_mask = torch.ones(len(wavelengths_main), 5, 5, dtype=torch.bool)
    band_name = "R"
    method = "center"
    mask = True
    cutoff_min = False
    normalize = True

    hsi_image = mt_image.Image(image=image, wavelengths=wavelengths, binary_mask=binary_mask)
    result = hsi_image.select_single_band_from_name(
        band_name=band_name, method=method, mask=mask, cutoff_min=cutoff_min, normalize=normalize
    )

    assert result.shape == (5, 5), "Selected band shape should match the image shape"
    assert isinstance(result, torch.Tensor), "Selected band should be a torch tensor"

    # Test selecting an invalid band
    band_name = "InvalidBand"
    with pytest.raises(AssertionError):
        hsi_image.select_single_band_from_name(
            band_name=band_name, method=method, mask=mask, cutoff_min=cutoff_min, normalize=normalize
        )

    # Test selecting a valid band using an unsupported method
    band_name = "R"
    method = "unsupported"
    with pytest.raises(NotImplementedError):
        hsi_image.select_single_band_from_name(
            band_name=band_name, method=method, mask=mask, cutoff_min=cutoff_min, normalize=normalize
        )


def test_get_rgb_image():
    # Test with default settings
    image = torch.randn(len(wavelengths_main), 10, 10)
    wavelengths = wavelengths_main
    orientation = ("C", "H", "W")
    binary_mask = torch.ones(len(wavelengths_main), 10, 10, dtype=torch.bool)
    device = torch.device("cpu")
    test_image = mt_image.Image(
        image=image, wavelengths=wavelengths, orientation=orientation, binary_mask=binary_mask, device=device
    )

    result = test_image.get_rgb_image()

    assert result.shape == torch.Size([3, 10, 10]), "RGB image shape should be [3, 10, 10]"

    # Test with specific output band axis
    result = test_image.get_rgb_image(output_rgb_band_axis=2)

    assert result.shape == torch.Size([10, 10, 3]), "RGB image shape should be [10, 10, 3]"

    # Test without applying a mask
    result = test_image.get_rgb_image(mask=False)

    assert result.shape == torch.Size([3, 10, 10]), "RGB image shape should be [3, 10, 10]"
