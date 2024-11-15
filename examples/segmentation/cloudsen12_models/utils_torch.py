import numpy as np
import torch


def find_padding(v: int, divisor: int = 8):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def padded_predict(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    divisor: int = 32,
    hard_labels=False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """Predict on a tensor adding padding if necessary.

    Args:
        tensor: torch.Tensor (C, H, W) of input values
        model: torch.nn.Module
        divisor: int
        hard_labels: bool
        device: torch.device

    Returns:
        2D or 3D torch.Tensor with the prediction
    """
    assert len(tensor.shape) == 3, f"Expected 3D tensor, found {len(tensor.shape)}D tensor"

    # Calculate padding needed to make the tensor dimensions divisible by the divisor
    pad_r = find_padding(tensor.shape[-2], divisor)
    pad_c = find_padding(tensor.shape[-1], divisor)

    # Pad the tensor using PyTorch's `F.pad` (reflect padding)
    tensor_padded = torch.nn.functional.pad(tensor, (pad_c[0], pad_c[1], pad_r[0], pad_r[1]), mode="reflect")

    # Slices to remove padding after prediction
    slice_rows = slice(pad_r[0], None if pad_r[1] == 0 else -pad_r[1])
    slice_cols = slice(pad_c[0], None if pad_c[1] == 0 else -pad_c[1])

    # Move tensor to the specified device
    tensor_padded = tensor_padded.to(device).unsqueeze(0)  # Add batch dimension

    # Ensure model is on the correct device
    model = model.to(device)

    with torch.no_grad():
        pred_padded = model(tensor_padded)[0]
        if len(pred_padded.shape) == 3:
            pred = pred_padded[:, slice_rows, slice_cols]
        elif len(pred_padded.shape) == 2:
            pred = pred_padded[slice_rows, slice_cols]
        else:
            raise NotImplementedError(f"Don't know how to slice the tensor of shape {pred_padded.shape}")

    pred_padded = model(tensor_padded)[0]
    if len(pred_padded.shape) == 3:
        pred = pred_padded[:, slice_rows, slice_cols]
    elif len(pred_padded.shape) == 2:
        pred = pred_padded[slice_rows, slice_cols]
    else:
        raise NotImplementedError(f"Don't know how to slice the tensor of shape {pred_padded.shape}")

    return pred
