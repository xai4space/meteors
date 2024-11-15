import os
import urllib
import hashlib
import warnings
import torch
from tqdm import tqdm

from clip_model import build_model


BASE_MODEL_URL = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"


def download(url: str, root: str, error_checksum: bool = True) -> str:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    if expected_sha256 == "main":
        expected_sha256 = "0cc03ba20aff35a41312de47da843a0d8541acba3c2101d9691f3ab999128d34"  # CLIP sha256
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        # real_sha256 = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        # print("INFO: Real SHA256: {}".format(real_sha256))
        # print("INFO: Expected SHA256: {}".format(expected_sha256))
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:  # type: ignore
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        if error_checksum:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
        else:
            warnings.warn("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def load_base_clip(download_root: str, class_num: int = 1000):
    model_path = download(BASE_MODEL_URL, download_root)
    model = torch.jit.load(model_path, map_location="cpu").eval()
    model = build_model(model.state_dict(), downstream_task=4, class_num=class_num)
    model.float()
    return model
