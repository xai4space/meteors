import torch
import pytest
import meteors as mt


def test_image():
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    mt.Image(image=sample, wavelengths=wavelengths, binary_mask="artificial")


def test_wavelengths():
    sample = torch.tensor([[[0]]])
    with pytest.raises(
        ValueError,
        match="Improper length of wavelengths - it should correspond to the number of channels",
    ):
        wavelengths = [0, 1]
        mt.Image(image=sample, wavelengths=wavelengths)


def test_artificial_mask():
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    image = mt.Image(image=sample, wavelengths=wavelengths, binary_mask="artificial")
    assert torch.equal(image.binary_mask, torch.tensor([[[0]]])), "The simplest mask with no data should be created"


def test_incorrect_shape_mask():
    sample = torch.tensor([[[0]]])
    wavelengths = [0]
    with pytest.raises(ValueError):
        binary_mask = torch.tensor([[[0, 0]]])
        mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)

    with pytest.raises(
        ValueError,
        match="Mask should be a tensor, numpy ndarray or a string 'artificial' which will create an automatic mask",
    ):
        binary_mask = "very bad mask"
        mt.Image(image=sample, wavelengths=wavelengths, binary_mask=binary_mask)


def test_rgb_image():
    # Placeholder for the RGB image test
    pass
