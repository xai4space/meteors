import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap
import numpy as np
import torch

import meteors as mt
import meteors.shap.explanation as mt_shap_explanation

import pandas as pd

import pytest

from pydantic import ValidationError


class ValidationInfoMock:
    def __init__(self, data):
        self.data = data


def test_ensure_data_type_and_reshape():
    data = np.random.rand(10, 10)
    assert mt_shap_explanation.ensure_data_type_and_reshape(data) is data

    pd_data = pd.DataFrame(data)
    new_data = mt_shap_explanation.ensure_data_type_and_reshape(pd_data)

    assert isinstance(new_data, np.ndarray)
    assert new_data.shape == data.shape

    data = torch.tensor(data)
    new_data = mt_shap_explanation.ensure_data_type_and_reshape(data)

    assert isinstance(new_data, np.ndarray)
    assert new_data.shape == data.shape

    incorrect_data = "invalid"
    with pytest.raises(TypeError):
        mt_shap_explanation.ensure_data_type_and_reshape(incorrect_data)

    non_castable_data = [1, "bad", [1, 2, 3]]
    with pytest.raises(TypeError, match="Expected NumPy array | Torch Tensor | Pandas DataFrame as data"):
        mt_shap_explanation.ensure_data_type_and_reshape(non_castable_data)

    # check if the data is reshaped
    data = np.random.rand(10)
    new_data = mt_shap_explanation.ensure_data_type_and_reshape(data)

    assert new_data.shape == (1, 10)

    data = pd.Series(np.random.rand(10))
    new_data = mt_shap_explanation.ensure_data_type_and_reshape(data)

    assert new_data.shape == (1, 10)
    assert isinstance(new_data, np.ndarray)


def test_shap_explanation():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainer = shap.KernelExplainer(knn.predict_proba, X_train)

    raw_explanation = explainer(X_test)

    explanation = mt.shap.SHAPExplanation(data=X_test, explanations=raw_explanation, explanation_method="kernel")

    assert explanation.data.shape == X_test.shape
    assert explanation.explanations.shape == raw_explanation.shape
    assert explanation.explanation_method == "kernel"

    # custom property works
    assert all(explanation.feature_names == X_test.columns)

    # two dimensional output from a model
    Y_train_wide = np.vstack([Y_train, 1 - Y_train]).T

    lr = sklearn.linear_model.LinearRegression()
    lr.fit(X_train, Y_train_wide)

    explainer = shap.LinearExplainer(lr, X_train)

    raw_explanation = explainer(X_test)

    explanation = mt.shap.SHAPExplanation(data=X_test, explanations=raw_explanation, explanation_method="linear")

    assert explanation.data.shape == X_test.shape
    assert explanation.explanations.shape == raw_explanation.shape
    assert explanation.explanation_method == "linear"

    # incorrect shapes
    X_train_smaller = X_train[:10]

    with pytest.raises(ValidationError):
        explanation = mt.shap.SHAPExplanation(
            data=X_train_smaller, explanations=raw_explanation, explanation_method="linear"
        )

    # expect loguru warning for incorrect explanation method
    explanation = mt.shap.SHAPExplanation(data=X_test, explanations=raw_explanation, explanation_method="invalid")

    # local explanations
    local_explanation = mt.shap.SHAPExplanation(
        data=X_test.iloc[0], explanations=raw_explanation[0], explanation_method="linear"
    )
    assert local_explanation.is_local_explanation


def test_feature_names():
    # case 1: no feature names

    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 10))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert shap_explanation.feature_names is None

    # case 2: feature names
    feature_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
    explanations = shap.Explanation(np.random.rand(10, 10), feature_names=feature_names)

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert shap_explanation.feature_names == feature_names


def test_is_local_explanation():
    # case 1: no multitarget explanation
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 10))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert not shap_explanation.is_local_explanation

    # case 2: multitarget explanation
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 10, 2))
    explanations = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert not explanations.is_local_explanation

    # case 3: local single-target explanation
    data = np.random.rand(10)
    explanations = shap.Explanation(np.random.rand(10))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert shap_explanation.is_local_explanation

    # case 4: local multitarget explanation
    data = np.random.rand(10)
    explanations = shap.Explanation(np.random.rand(10, 2))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    assert shap_explanation.is_local_explanation


def test___validate_shapes():
    # not testing the function itself, but the cases that it covers

    # case 1: no multitarget explanation
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 10))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    # case 2: multitarget explanation
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 10, 2))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    # case 3: local single-target explanation
    data = np.random.rand(10)
    explanations = shap.Explanation(np.random.rand(10))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    # the shape should be adjusted
    assert shap_explanation.explanations.shape == (1, 10)
    assert shap_explanation.data.shape == (1, 10)

    # case 4: local multitarget explanation

    data = np.random.rand(10)
    explanations = shap.Explanation(np.random.rand(10, 2))

    shap_explanation = mt_shap_explanation.SHAPExplanation(
        data=data, explanations=explanations, explanation_method="kernel"
    )

    # the shape should be adjusted
    assert shap_explanation.explanations.shape == (1, 10, 2)
    assert shap_explanation.data.shape == (1, 10)

    # case 5: invalid shape
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 9))

    with pytest.raises(ValidationError):
        shap_explanation = mt_shap_explanation.SHAPExplanation(
            data=data, explanations=explanations, explanation_method="kernel"
        )

    # case 6: invalid shape - multitarget
    data = np.random.rand(10, 10)
    explanations = shap.Explanation(np.random.rand(10, 9, 3))

    with pytest.raises(ValidationError):
        shap_explanation = mt_shap_explanation.SHAPExplanation(
            data=data, explanations=explanations, explanation_method="kernel"
        )
