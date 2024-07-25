from __future__ import annotations

from typing_extensions import Type, Any, TypeVar, Callable, Iterable

import torch

T = TypeVar("T")


def torch_dtype_to_python_dtype(dtype: torch.dtype) -> Type:
    """Converts a PyTorch dtype to a Python data type.

    Args:
        dtype (torch.dtype): The PyTorch dtype to be converted.

    Returns:
        Type: The corresponding Python data type.

    Raises:
        TypeError: If the PyTorch dtype cannot be converted to a Python data type.
    """
    if dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.complex128):
        return float
    elif dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return int
    elif dtype == torch.bool:
        return bool
    else:
        raise TypeError(f"Can't convert PyTorch dtype {dtype} to Python data type")


def change_dtype_of_list(original_list: list[Any] | Any, dtype: Callable[[Any], T]) -> list[T]:
    """Change the data type of elements in a list.

    Args:
        original_list (list[Any]): The original list.
        dtype (T): The desired data type.

    Returns:
        list[T]: The modified list with elements of the desired data type.
    """
    if isinstance(original_list, Iterable) and not isinstance(original_list, str):
        return [dtype(element) for element in original_list]
    else:
        return [dtype(original_list)]
