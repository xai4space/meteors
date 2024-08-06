import pytest

import torch

from meteors.utils import utils


def test_torch_dtype_to_python_dtype():
    # Test case 1: torch.float
    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.complex128):
        result = utils.torch_dtype_to_python_dtype(dtype)
        assert result == float

    # Test case 2: torch.int
    for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        result = utils.torch_dtype_to_python_dtype(dtype)
        assert result == int

    # Test case 3: torch.bool
    dtype = torch.bool
    result = utils.torch_dtype_to_python_dtype(dtype)
    assert result == bool

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
