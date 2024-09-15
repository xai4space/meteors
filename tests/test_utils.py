import pytest

import torch

from meteors.utils import utils
from meteors import HSI


def test_torch_dtype_to_python_dtype():
    # Test case 1: torch.float
    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.complex128):
        result = utils.torch_dtype_to_python_dtype(dtype)
        assert isinstance(result, type) and result == float  # noqa

    # Test case 2: torch.int
    for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        result = utils.torch_dtype_to_python_dtype(dtype)
        assert isinstance(result, type) and result == int  # noqa

    # Test case 3: torch.bool
    dtype = torch.bool
    result = utils.torch_dtype_to_python_dtype(dtype)
    assert isinstance(result, type) and result == bool  # noqa

    # Test case 4: Invalid dtype
    dtype = str
    with pytest.raises(TypeError):
        utils.torch_dtype_to_python_dtype(dtype)


def test_change_dtype_of_list():
    # Test case 1: Change dtype of elements in a list of integers
    original_list = [1, 2, 3, 4, 5]
    dtype = float
    expected_result = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = utils.change_dtype_of_list(original_list, dtype)
    assert result == expected_result

    # Test case 2: Change dtype of elements in a list of floats
    original_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    dtype = int
    expected_result = [1, 2, 3, 4, 5]
    result = utils.change_dtype_of_list(original_list, dtype)
    assert result == expected_result

    # Test case 3: Change dtype of elements in a list of strings
    original_list = ["1", "2", "3", "4", "5"]
    dtype = int
    expected_result = [1, 2, 3, 4, 5]
    result = utils.change_dtype_of_list(original_list, dtype)
    assert result == expected_result

    # Test case 4: Change dtype of a single element
    original_list = 10
    dtype = str
    expected_result = ["10"]
    result = utils.change_dtype_of_list(original_list, dtype)
    assert result == expected_result

    # Test case 5: Change dtype of an empty list
    original_list = []
    dtype = bool
    expected_result = []
    result = utils.change_dtype_of_list(original_list, dtype)
    assert result == expected_result


def test_expand_spectral_mask():
    # Create a sample hsi
    wavelengths = torch.tensor([400, 450, 500, 550, 600])
    hsi = HSI(image=torch.ones((len(wavelengths), 3, 3)), wavelengths=wavelengths, orientation=("C", "H", "W"))

    # Create a sample band mask
    band_mask_single_dim = torch.tensor([1, 2, 3, 4, 5])

    # Test case 1: Repeat dimensions is False
    repeat_dimensions = False
    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim, repeat_dimensions)

    expected_shape = (5, 1, 1)
    assert expanded_band_mask.shape == expected_shape

    expected_values = torch.tensor(
        [
            [[1]],
            [[2]],
            [[3]],
            [[4]],
            [[5]],
        ]
    )
    assert torch.equal(expanded_band_mask, expected_values)

    # Test case 2: Repeat dimensions is True and band axis is 0
    repeat_dimensions = True
    hsi = HSI(image=torch.ones((len(wavelengths), 3, 3)), wavelengths=wavelengths, orientation=("C", "H", "W"))
    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim, repeat_dimensions)

    expected_shape = (5, 3, 3)
    assert expanded_band_mask.shape == expected_shape

    expected_values = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
        ]
    )
    assert torch.equal(expanded_band_mask, expected_values)

    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim.unsqueeze(1), repeat_dimensions)
    assert expanded_band_mask.shape == expected_shape
    assert torch.equal(expanded_band_mask, expected_values)

    # Test case 3: Repeat dimensions is True and band axis is 1
    repeat_dimensions = True
    hsi = HSI(image=torch.ones((3, len(wavelengths), 3)), wavelengths=wavelengths, orientation=("H", "C", "W"))
    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim, repeat_dimensions)

    expected_shape = (3, 5, 3)
    assert expanded_band_mask.shape == expected_shape

    expected_values = torch.tensor(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
        ]
    )
    assert torch.equal(expanded_band_mask, expected_values)

    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim.unsqueeze(0), repeat_dimensions)
    assert expanded_band_mask.shape == expected_shape
    assert torch.equal(expanded_band_mask, expected_values)

    # Test case 4: Repeat dimensions is True and band axis is 2
    repeat_dimensions = True
    hsi = HSI(image=torch.ones((3, 3, len(wavelengths))), wavelengths=wavelengths, orientation=("H", "W", "C"))
    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim, repeat_dimensions)

    expected_shape = (3, 3, 5)
    assert expanded_band_mask.shape == expected_shape

    expected_values = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        ]
    )
    assert torch.equal(expanded_band_mask, expected_values)

    expanded_band_mask = utils.expand_spectral_mask(hsi, band_mask_single_dim.unsqueeze(0), repeat_dimensions)


def test_adjust_shape():
    # Test case 1: Same shape
    target = torch.tensor([[1, 2], [3, 4]])
    source = torch.tensor([[5, 6], [7, 8]])
    expected_result = torch.tensor([[5, 6], [7, 8]])
    result = utils.adjust_shape(target, source)
    assert torch.allclose(result, expected_result)

    # Test case 2: Source tensor has fewer dimensions
    target = torch.tensor([[1, 2], [3, 4]])
    source = torch.tensor([5, 6])
    expected_result = torch.tensor([[5, 6], [5, 6]])
    result = utils.adjust_shape(target, source)
    assert torch.allclose(result, expected_result)

    # Test case 3: Source tensor has more dimensions
    target = torch.tensor([[1, 2], [3, 4]])
    source = torch.tensor([[[5, 6]]])
    expected_result = torch.tensor([[5, 6], [5, 6]])
    result = utils.adjust_shape(target, source)
    assert torch.allclose(result, expected_result)

    # Test case 4: Source tensor has different shape and needs broadcasting
    target = torch.tensor([[1, 2], [3, 4]])
    source = torch.tensor([5, 6])
    expected_result = torch.tensor([[5, 6], [5, 6]])
    result = utils.adjust_shape(target, source)
    assert torch.allclose(result, expected_result)


def test_aggregate_by_mask():
    # Test case 1: Aggregating data by mask with mean
    data = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
    mask = torch.tensor([[[0, 0, 1], [1, 1, 2]], [[2, 2, 0], [0, 1, 2]]])
    expected_result = torch.tensor(
        [[[5.5, 5.5, 5.75], [5.75, 5.75, 8.25]], [[8.25, 8.25, 5.5], [5.5, 5.75, 8.25]]], dtype=torch.float32
    )
    result = utils.aggregate_by_mask(data, mask, torch.mean)
    assert torch.allclose(result, expected_result)

    # Test case 2: Aggregating data by mask with max
    expected_result = torch.tensor([[[10, 10, 11], [11, 11, 12]], [[12, 12, 10], [10, 11, 12]]], dtype=torch.float32)
    result = utils.aggregate_by_mask(data, mask, torch.max)
    assert torch.allclose(result, expected_result)

    # Test case 3: Aggregating data by mask with min
    expected_result = torch.tensor([[[1, 1, 3], [3, 3, 6]], [[6, 6, 1], [1, 3, 6]]], dtype=torch.float32)
    result = utils.aggregate_by_mask(data, mask, torch.min)
    assert torch.allclose(result, expected_result)

    # Test case 4: Mismatched shapes
    data = torch.randn(3, 3, 3)
    mask = torch.zeros((3, 3, 2), dtype=torch.long)
    with pytest.raises(ValueError):
        utils.aggregate_by_mask(data, mask, torch.mean)

    # Test case 5: Single value per mask
    data = torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)
    mask = torch.tensor([[[0], [1]], [[2], [3]]])
    result = utils.aggregate_by_mask(data, mask, torch.mean)
    expected_result = torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.float32)
    assert torch.allclose(result, expected_result)

    # Test case 6: Empty mask
    data = torch.tensor([])
    mask = torch.tensor([])
    result = utils.aggregate_by_mask(data, mask, torch.mean)
    expected_result = torch.tensor([])
    assert torch.allclose(result, expected_result)
