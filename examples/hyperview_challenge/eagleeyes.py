from __future__ import annotations

from typing import List, Sequence, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import pywt
import os

from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
)


TRAIN_SIZE = 1732
TEST_SIZE = 1154
TRAIN_DIR = "train_data_simulated/"
TEST_DIR = "test_data_simulated/"


CLASSES = ["P", "K", "Mg", "pH"]

class_metrics = {
    "avg_acc": (balanced_accuracy_score, {}),
    "acc": (accuracy_score, {}),
    "mcc": (matthews_corrcoef, {}),
    "f1": (f1_score, {"average": "macro"}),
}

ph_classes_names = [
    "acidic",
    "strongly acidic",
    "slightly acidic",
    "neutral",
    "alkaline",
]
classes_names = ["very low", "low", "medium", "high", "very high"]
ph_thresholds = [4.6, 5.6, 6.6, 7.3]

# According to the pH class:
phosphorus_thresholds = [
    [50, 110, 186, 262],  # 0 Acidic
    [49, 103, 158, 215],  # 1 Strongly acidic
    [47, 99, 152, 207],  # ...
    [27, 54, 75, 99],
    [27, 54, 75, 99],
]
# According to the soil class:
potassium_thresholds = [
    [32, 75, 119, 162],  # 0 Very light
    [52, 99, 145, 191],  # 1 Light
    [98, 139, 200, 241],  # 2 Medium
    [126, 174, 270, 318],  # 3 Heavy
]
# According to the soil class:
magnesium_thresholds = [
    [7, 21, 51, 80],
    [31, 43, 67, 93],
    [48, 77, 106, 135],
    [69, 93, 142, 191],
]


def element_classification(result: float, thresholds: Sequence[float | int]) -> int:
    id_ = 0
    for i, t in enumerate(thresholds):
        if result > t:
            id_ = i + 1
        else:
            break
    return id_


def ph_classification(result: float) -> int:
    return element_classification(result, ph_thresholds)


def phosphorus_classification(result: float, ph_class: int) -> int:
    return element_classification(result, phosphorus_thresholds[int(ph_class)])


def potassium_classification(result: float, soil_class: int) -> int:
    return element_classification(result, potassium_thresholds[int(soil_class)])


def magnesium_classification(result: float, soil_class: int) -> int:
    return element_classification(result, magnesium_thresholds[int(soil_class)])


def get_classes(y: pd.DataFrame, soil_class: int = 3) -> pd.DataFrame:
    y_classes: dict[str, list] = {k: [] for k in CLASSES}
    for _, row in y.iterrows():
        y_classes["pH"].append(ph_classification(row["pH"]))
        y_classes["P"].append(phosphorus_classification(row["P"], y_classes["pH"][-1]))
        y_classes["K"].append(potassium_classification(row["K"], soil_class))
        y_classes["Mg"].append(magnesium_classification(row["Mg"], soil_class))
    return pd.DataFrame.from_dict(y_classes)


def load_data() -> Tuple[List, pd.DataFrame, List, pd.DataFrame]:
    X_train = [os.path.join(TRAIN_DIR, f"{i}.npz") for i in range(TRAIN_SIZE)]
    X_test = [os.path.join(TEST_DIR, f"{i}.npz") for i in range(TEST_SIZE)]
    y_train = pd.read_csv("train_gt.csv")
    y_test = pd.read_csv("test_gt.csv")
    return X_train, y_train, X_test, y_test


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

    return np.vstack(samples_lst), all_feature_names
