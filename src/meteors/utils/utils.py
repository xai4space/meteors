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


def adjust_shape(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Adjust the shape of a source tensor to match the shape of target tensor.

    Args:
        target (torch.Tensor): The target tensor.
        source (torch.Tensor): The source tensor.

    Returns:
        torch.Tensor: The source tensor with the shape adjusted to match the target tensor.
    """
    if source.shape == target.shape:
        return source

    # If the source tensor has fewer dimensions than the target tensor, add dimensions to the source tensor
    while source.dim() < target.dim():
        source = source.unsqueeze(0)

    # If the source tensor has more dimensions than the target tensor, remove dimensions from the source tensor
    while source.dim() > target.dim():
        source = source.squeeze(0)

    # If the source tensor has a different shape than the target tensor, broadcast the source tensor to match the target tensor
    if source.shape != target.shape:
        source = source.expand_as(target)

    return source


def agg_segmentation_postprocessing(
    soft_labels: bool = False, classes_numb: int = 0, class_axis: int = 1
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Postprocessing function for aggregating segmentation outputs.

    This function takes the segmentation model output and sums the pixel scores for
    all pixels predicted as each class, returning a tensor with a single value for
    each class. This makes it easier to attribute with respect to a single output
    scalar, as opposed to an individual pixel output attribution.

    Args:
        soft_labels (bool, optional): Whether the model output is soft labels (probabilities for each class)
            or hard labels (one-hot encoded). Defaults to False
        classes_numb (bool, optional): The number of classes in the model output.
            If 0, use unique values in the output. Defaults to 0.
        class_axis (int, optional): The axis of the model output tensor that contains the class predictions
            if the model output is soft labels. Defaults to 1.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            The postprocessing function that accepts model outputs with lime mask of masked region
            and returns aggregated outputs.
    """

    def postprocessing_function(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nonlocal classes_numb
        # Adjust the shape of the mask to match the shape of the output
        mask = adjust_shape(output, mask)

        # If the output is soft labels, get the class with the highest probability
        if soft_labels:
            output_labels = output.argmax(dim=class_axis)
        else:
            output_labels = output

        # IMPORTANT if mask does not have the same shape as output, you need to adjust the mask to the output shape

        # Mask the output to only consider the pixels is in the mask
        masked_output = torch.where(mask, output_labels, torch.ones_like(output_labels) * -1)

        # Get flattened masked output
        batch_size = masked_output.size(0)
        flattened_masked_output = masked_output.reshape(batch_size, -1)

        # Get the unique class labels and their counts
        unique_classes, counts = [], []
        for batch in flattened_masked_output:
            u, c = torch.unique(batch, return_counts=True)
            unique_classes.append(u)
            counts.append(c)

        if classes_numb == 0:
            # Find the maximum number of unique classes across all batches
            classes_numb = max(int(u.max().item()) for u in unique_classes) + 1

        # Create a tensor to store the final counts for each class
        final_counts = torch.zeros(batch_size, classes_numb)
        for i in range(batch_size):
            for u, c in zip(unique_classes[i], counts[i]):
                if int(u.item()) != -1:
                    final_counts[i, int(u.item())] = c

        return final_counts

    return postprocessing_function
