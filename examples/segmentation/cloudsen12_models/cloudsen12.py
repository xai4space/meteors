import torch
import torch.nn
import segmentation_models_pytorch as smp
import numpy as np
from typing import Optional, Sequence
import matplotlib.axes
import matplotlib.image
import os
from .utils_torch import padded_predict
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt


INTERPRETATION_CLOUDSEN12 = ["clear", "Thick cloud", "Thin cloud", "Cloud shadow"]
COLORS_CLOUDSEN12 = (
    np.array(
        [
            [20, 20, 255],  # clear
            [220, 220, 220],  # Thick cloud
            [180, 180, 180],  # Thin cloud
            [60, 60, 60],
        ],  # cloud shadow
        dtype=np.float32,
    )
    / 255
)

# sourced from georeader.readers.S2_SAFE_reader
BANDS_S2_L1C = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

BANDS_S2_L2A = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


# Models stored in this folder https://drive.google.com/drive/folders/1gpqEOWZRHlxLSJMMc3TXW7BRDuGBkDRa?usp=drive_link
MODELS_CLOUDSEN12 = {
    "cloudsen12": {"model_file": "cloudsen12.pt", "bands": BANDS_S2_L1C, "type": "weights"},
    "UNetMobV2_V1": {"model_file": "UNetMobV2_V1.pt", "bands": BANDS_S2_L1C, "type": "weights"},
    "UNetMobV2_V2": {"model_file": "UNetMobV2_V2.pt", "bands": BANDS_S2_L1C, "type": "weights"},
    "cloudsen12l2a": {"model_file": "cloudsen12l2a.pt", "bands": BANDS_S2_L2A, "type": "jit"},
    "dtacs4bands": {"model_file": "dtacs4bands.pt", "bands": ["B08", "B04", "B03", "B02"], "type": "jit"},
    "landsat30": {
        "model_file": "landsat30.pt",
        "bands": ["B01", "B02", "B03", "B04", "B08", "B10", "B11", "B12"],
        "type": "jit",
    },
}

# Landsat-8/9 naming convention: https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes


def download_weights(weights_file: str):
    """Download weights from HuggingFace.

    Args:
        weights_file (str): path to weights file

    Raises:
        ImportError: if gdown is not installed

    Example:
        >>> download_weights("weights/s2models/cloudsen12.ckpt", MODELS_CLOUDSEN12["all"]["url"])
    """
    from huggingface_hub import hf_hub_download

    if not os.path.exists(weights_file):
        model_file = os.path.basename(weights_file)
        dirdownload = os.path.dirname(weights_file)
        os.makedirs(dirdownload, exist_ok=True)
        model_file_path = hf_hub_download(
            repo_id="isp-uv-es/cloudsen12_models",
            filename=model_file,
            local_dir=dirdownload,
            local_dir_use_symlinks=False,
        )
        if model_file_path != weights_file:
            raise ValueError(f"Error downloading {weights_file} file is downloaded in {model_file_path}")
        if not os.path.exists(weights_file):
            raise ValueError(f"Error downloading {weights_file}")

    return weights_file


class CDModel(torch.nn.Module):
    f"""
    Cloud detection model trained on the cloudSEN12 dataset.
    The model is a UNet with a MobileNetV2 encoder.
    The model was trained on Sentinel-2 L1C images with 13 channels (13, H, W) in TOA units.
    The model outputs a segmentation mask with 4 classes:
        0: clear
        1: Thick cloud
        2: thin cloud
        3: cloud shadow

    Example:
        model = CDModel(device=torch.device("cpu"))
        weights_file = "weights/s2models/cloudsen12.ckpt"
        if not os.path.exists(weights_file):
            download_weights(weights_file, MODELS_CLOUDSEN12['all']['url'])

        weights = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(weights["state_dict"])

        img = ... # Sentinel-2 L1C image with 13 channels (13, H, W) in TOA units

        output = model.predict(img)
        # output is a (H, W) uint8 array with values in {INTERPRETATION_CLOUDSEN12}

        plot_cloudSEN12mask(output)

    """

    def __init__(self, bands: Sequence[str], device=torch.device("cpu"), model: Optional[torch.nn.Module] = None):
        super().__init__()
        if model is None:
            self.model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=len(bands), classes=4)
        else:
            self.model = model

        self.bands = bands
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def forward(self, tensor: torch.Tensor, hard_labels=False) -> torch.Tensor:
        pred_cont = self.model(tensor)
        if hard_labels:
            pred_discrete = torch.argmax(pred_cont, dim=1).type(torch.uint8)
            return pred_discrete
        else:
            return torch.softmax(pred_cont, dim=1)

    def predict(self, tensor: ArrayLike, hard_labels=False) -> ArrayLike:
        """

        Args:
            tensor: np.array (len(self.bands), H, W) of TOA reflectances (values between 0 and 1)
            hard_labels: if True, return the hard labels - the class with the highest probability

        Returns:
            uint8 np.array (H, W) with interpretation {0: clear, 1: Thick cloud, 2: thin cloud, 3: cloud shadow} in case hard_labels is True,
            otherwise a float32 np.array (4, H, W) with the probabilities of each class
        """
        if hasattr(tensor, "values") and hasattr(tensor, "transform"):
            raise NotImplementedError(
                "GeoTensor not supported, please use the original cloudsen12 code to perform inference"
            )
        transform = None

        if isinstance(tensor, np.ndarray):
            tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.to(dtype=torch.float32)

        if len(tensor.shape) == 4:
            # extract examples from batches
            pred = []
            for i in range(tensor.shape[0]):
                pred.append(self.predict(tensor[i]))
            pred = torch.stack(pred, axis=0)
            return pred

        assert tensor.shape[0] == len(self.bands), f"Expected {len(self.bands)} channels found {tensor.shape[0]}"

        pred = padded_predict(tensor, self, 32, self.device)

        if hard_labels:
            pred = torch.argmax(pred, dim=0).type(torch.uint8)

        if transform is not None:
            raise NotImplementedError(
                "Transform not supported. Please use the original cloudsen12 code to perform inference"
            )

        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.int8).unsqueeze(0)
        return pred


def load_model_by_name(
    name: str, weights_folder: str = "cloudsen12_models", device: torch.device = torch.device("cpu")
) -> CDModel:
    f"""
    Load a model by name

    Args:
        name (str): name of the model. One of {MODELS_CLOUDSEN12.keys()}
        device (torch.device, optional): device to load the model. Defaults to torch.device("cpu").

    Raises:
        ValueError: if the name is not in MODELS_CLOUDSEN12

    Returns:
        CDModel: cloud detection model
    """
    if name not in MODELS_CLOUDSEN12:
        raise ValueError(f"Model name {name} not in {MODELS_CLOUDSEN12.keys()}")

    model_file = MODELS_CLOUDSEN12[name]["model_file"]
    if not isinstance(model_file, str) and isinstance(model_file, Sequence):
        weights_file = os.path.join(weights_folder, *model_file)
    else:
        weights_file = os.path.join(weights_folder, model_file)

    if MODELS_CLOUDSEN12[name]["type"] == "weights":
        model = CDModel(device=device, bands=MODELS_CLOUDSEN12[name]["bands"])
        if not os.path.exists(weights_file):
            download_weights(weights_file)
        weights = torch.load(weights_file, map_location=device)
        if "state_dict" in weights:
            model.load_state_dict(weights["state_dict"])
        else:
            model.model.load_state_dict(weights)

    else:
        if not os.path.exists(weights_file):
            download_weights(weights_file)

        model = CDModel(device=device, bands=MODELS_CLOUDSEN12[name]["bands"], model=torch.jit.load(weights_file))

    return model


def plot_cloudSEN12mask(
    mask: ArrayLike,
    legend: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """

    Args:
        mask: (H, W)
        legend: plot the legend
        ax: matplotlib.Axes to plot

    Returns:
        matplotlib.axes.Axes
    """

    if not isinstance(mask, np.ndarray) and not isinstance(mask, torch.Tensor):
        raise NotImplementedError(
            "plotting for GeoData not supported, please use the original cloudsen12 code or refer to the georeader.plot.plot_segmentation_mask function"
        )
    if not isinstance(mask, np.ndarray):
        mask = mask.detach().numpy()

    ax = ax or plt.gca()
    ax.axis("off")

    ax.imshow(COLORS_CLOUDSEN12[mask.squeeze()])

    patches = [
        matplotlib.patches.Patch(color=color, label=label)
        for color, label in zip(COLORS_CLOUDSEN12, INTERPRETATION_CLOUDSEN12)
    ]

    if legend:
        ax.legend(handles=patches)

    return ax
