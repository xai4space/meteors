from __future__ import annotations

from typing import List, Sequence, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import pywt
import os


import urllib
import hashlib
import warnings
from tqdm import tqdm

from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
)


# filter out scikit learn warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


TRAIN_SIZE = 1732
TEST_SIZE = 1154
TRAIN_DIR = "train_data_simulated/"
TEST_DIR = "test_data_simulated/"


CLASSES = ["P", "K", "Mg", "pH"]

FEATURES = [
    "Real part of FFT over avg. reflectance",
    "Imag part of FFT over avg. reflectance",
    "Approximation coefficients of wavelet transform sym3",
    "Approximation coefficients of wavelet transform dmey",
    "1st derivative of avg. reflectance",
    "2nd derivative of avg. reflectance",
    "3rd derivative of avg. reflectance",
    "Average reflectance over channels",
]

with open("data/wavelenghts.txt", "r") as f:
    BANDS_HYPERVIEW = f.readline()
BANDS_HYPERVIEW = [float(wave.strip()) for wave in BANDS_HYPERVIEW.split(",")]

FEATURE_NAMES_HYPERVIEW = [f"{trans_name} | {bands_name}" for trans_name in FEATURES for bands_name in BANDS_HYPERVIEW]



class SpectralCurveFiltering:
    def __init__(self, merge_function=np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return self.merge_function(sample, axis=(1, 2))


class BaselineRegressor:
    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full((len(X_test), self.classes_count), self.mean)



def preprocess(samples_lst: List[str], features: List[str]) -> Tuple:
    def _shape_pad(data: np.ndarray) -> np.ndarray:
        """
        This sub-function makes padding to have square fields sizes.
        Not mandatory but eliminates the risk of calculation error
        in singular value decomposition.
        Padding by warping also improves the performance slightly.
        """
        max_edge = np.max(data.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(
            data,
            ((0, 0), (0, (shape[0] - data.shape[1])), (0, (shape[1] - data.shape[2]))),
            "wrap",
        )
        return padded

    filtering = SpectralCurveFiltering()
    w1 = pywt.Wavelet("sym3")
    w2 = pywt.Wavelet("dmey")

    all_feature_names = []

    for sample_index, sample_path in tqdm(enumerate(samples_lst), total=len(samples_lst)):
        with np.load(sample_path) as npz:
            data = np.ma.MaskedArray(**npz)
            data = _shape_pad(data)
            # Get the spatial features:
            s = np.linalg.svd(data, full_matrices=False, compute_uv=False)
            s0 = s[:, 0]
            s1 = s[:, 1]
            s2 = s[:, 2]
            s3 = s[:, 3]
            s4 = s[:, 4]
            dXds1 = s0 / (s1 + np.finfo(float).eps)
            ffts = np.fft.fft(s0)
            reals = np.real(ffts)
            imags = np.imag(ffts)

            # Get the specific spectral features:
            data = filtering(data)

            cA0, cD0 = pywt.dwt(data, wavelet=w2, mode="constant")
            cAx, cDx = pywt.dwt(cA0[12:92], wavelet=w2, mode="constant")
            cAy, cDy = pywt.dwt(cAx[15:55], wavelet=w2, mode="constant")
            cAz, cDz = pywt.dwt(cAy[15:35], wavelet=w2, mode="constant")
            cAw2 = np.concatenate((cA0[12:92], cAx[15:55], cAy[15:35], cAz[15:25]), -1)

            cA0, cD0 = pywt.dwt(data, wavelet=w1, mode="constant")
            cAx, cDx = pywt.dwt(cA0[1:-1], wavelet=w1, mode="constant")
            cAy, cDy = pywt.dwt(cAx[1:-1], wavelet=w1, mode="constant")
            cAz, cDz = pywt.dwt(cAy[1:-1], wavelet=w1, mode="constant")
            cAw1 = np.concatenate((cA0, cAx, cAy, cAz), -1)

            dXdl = np.gradient(data, axis=0)
            d2Xdl2 = np.gradient(dXdl, axis=0)
            d3Xdl3 = np.gradient(d2Xdl2, axis=0)

            fft = np.fft.fft(data)
            real = np.real(fft)
            imag = np.imag(fft)

            features_to_select = {
                "spatial": (dXds1, s0, s1, s2, s3, s4, reals, imags),
                "fft": (real, imag),
                "gradient": (dXdl, d2Xdl2, d3Xdl3),
                "mean": (data,),
                "dwt": (cAw1, cAw2),
            }

            # The best Feature combination for Random Forest based regression:
            sample_features: list = []
            sample_feature_names = []
            for feature_name in features:
                sample_features.extend(features_to_select[feature_name])
                sample_feature_names.extend([feature_name] * len(np.concatenate(features_to_select[feature_name])))

            sample_features = np.concatenate(sample_features, -1)
            samples_lst[sample_index] = sample_features
            all_feature_names.append(sample_feature_names)

    data = np.vstack(samples_lst)
    return pd.DataFrame(data, columns=FEATURE_NAMES_HYPERVIEW)


def download(url: str, root: str, error_checksum: bool = True) -> str:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-1]
    if expected_sha256 == "RF_model_150_bands.joblib":
        expected_sha256 = "f9167332bd4d87b6f8b863e53f9fc38e19d70c175ea8d4fb14df8ec676484684"  # RF_model_150_bands.joblib sha256
    if expected_sha256 == "RF_model_spatial-fft-dwt-gradient-mean_150_bands.joblib":
        expected_sha256 = "e078a7ef342fd313981c4b6a281e3497e32d6207f5c69cdbdd90814d6be5384b" # RF_model_spatial-fft-dwt-gradient-mean_150_bands.joblib
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError("{} exists and is not a regular file".format(download_target))

    if os.path.isfile(download_target):
        real_sha256 = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
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