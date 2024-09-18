from __future__ import annotations
from typing_extensions import Type, Any, TypeVar, Callable, Iterable

import torch

from meteors import HSI

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


def expand_spectral_mask(hsi: HSI, spectral_mask_single_dim: torch.Tensor, repeat_dimensions: bool) -> torch.Tensor:
    """Expands the spectral mask to match the dimensions of the input hsi.

    Args:
        hsi (HSI): The input hsi.
        spectral_mask_single_dim (torch.Tensor): The mask tensor with a single dimension.
        repeat_dimensions (bool): Whether to repeat the dimensions of the mask to match the hsi.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    spectral_mask = spectral_mask_single_dim
    if hsi.spectral_axis == 0:
        if spectral_mask_single_dim.ndim == 1:
            spectral_mask = spectral_mask_single_dim.unsqueeze(-1).unsqueeze(-1)
        elif spectral_mask_single_dim.ndim == 2:
            spectral_mask = spectral_mask_single_dim.unsqueeze(-1)
    elif hsi.spectral_axis == 1:
        if spectral_mask_single_dim.ndim == 1:
            spectral_mask = spectral_mask_single_dim.unsqueeze(0).unsqueeze(-1)
        elif spectral_mask_single_dim.ndim == 2:
            spectral_mask = spectral_mask_single_dim.unsqueeze(-1)
    elif hsi.spectral_axis == 2:
        if spectral_mask_single_dim.ndim == 1:
            spectral_mask = spectral_mask_single_dim.unsqueeze(0).unsqueeze(0)
        elif spectral_mask_single_dim.ndim == 2:
            spectral_mask = spectral_mask_single_dim.unsqueeze(0)
    if repeat_dimensions:
        spectral_mask = spectral_mask.expand_as(hsi.image)

    return spectral_mask


def adjust_shape(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Adjust the shape of a source tensor to match the shape of target tensor. The function squeezes or unsqueezes the
    source tensor if the dimensions of source and target don't match. Afterwards, it tries to broadcasts the source
    tensor to match the target dimensions.

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
    soft_labels: bool = False,
    classes_numb: int = 0,
    class_axis: int = 1,
    use_mask: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:  # pragma: no cover
    """Generator for postprocessing function for aggregating segmentation outputs.

    This generator creates a postprocessing function that takes the model output and input mask
    (these parameters are mandatory for the function), and then creates a 2d tensor
    with aggregated output <batch_size, classes_numb>. This is an example of a post-processing function
    for segmentation model output that aggregates pixel results. We encouraged to write more tailored functions.

    Args:
        soft_labels (bool, optional): Whether the model output is soft labels (probabilities for each class)
            or hard labels (one-hot encoded). Defaults to False
        classes_numb (bool, optional): The number of classes in the model output.
            If 0, use unique values in the output. Defaults to 0.
        class_axis (int, optional): The axis of the model output tensor that contains the class predictions
            if the model output is soft labels. Defaults to 1.
        use_mask (bool, optional): Whether to use the mask to multiply the output before aggregation.
            Defaults to True.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            The postprocessing function that accepts model outputs with lime mask of masked region
            and returns aggregated outputs.
    """

    def postprocessing_function(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Postprocessing function for aggregating segmentation outputs.

        This function takes the segmentation model output and sums the pixel scores for
        all pixels predicted as each class, returning a tensor with a single value for
        each class. This makes it easier to attribute with respect to a single output
        scalar, as opposed to an individual pixel output attribution.

        Args:
            output (torch.Tensor): The model output tensor.
            mask (torch.Tensor): The mask tensor used to mask the input.

        Returns:
            torch.Tensor: The aggregated output tensor.
        """
        nonlocal classes_numb
        # Adjust the shape of the mask to match the shape of the output
        mask = adjust_shape(output, mask)

        # If the output is soft labels, get the class with the highest probability
        if soft_labels:
            output_labels = output.argmax(dim=class_axis)
        else:
            output_labels = output

        # IMPORTANT if mask does not have the same shape as output, you need to adjust the mask to the output shape

        if use_mask:
            # Mask the output to only consider the pixels is in the mask
            masked_output = torch.where(mask, output_labels, torch.ones_like(output_labels) * -1)
        else:
            masked_output = output_labels

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


def aggregate_by_mask(
    data: torch.Tensor, mask: torch.Tensor, agg_func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Aggregate data by mask.

    This function aggregates the data tensor by the mask tensor (with IDs) using the aggregation function provided.

    Args:
        data (torch.Tensor): The data tensor to be aggregated.
        mask (torch.Tensor): The mask tensor used to aggregate the data.
        agg_func (Callable[[torch.Tensor], torch.Tensor]): The aggregation function to be applied to the data tensor.

    Raises:
        ValueError: If the data and mask tensors have different shapes.

    Returns:
        torch.Tensor: The aggregated data tensor.
    """
    if data.shape != mask.shape:
        raise ValueError("The data and mask tensors must have the same shapes")

    # Get unique values in the mask
    unique_ids = torch.unique(mask)

    # Initialize the result tensor
    result = torch.zeros_like(data, dtype=data.dtype)

    # Aggregate for each unique id
    for id in unique_ids:
        # Create a boolean mask for the current id
        id_mask = mask == id

        # Use the mask to select values from data
        selected_values = data[id_mask]

        # Get the aggregated result for the selected values
        agg_result = agg_func(selected_values)

        # Assign the aggregated result to the result tensor
        result[id_mask] = agg_result

    return result
