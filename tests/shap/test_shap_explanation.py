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


def test_ensure_explainer_type():
    assert mt_shap_explanation.ensure_explainer_type("exact") == "exact"
    assert mt_shap_explanation.ensure_explainer_type("invalid") == "invalid"


def test_ensure_data_type():
    data = np.random.rand(10, 10)
    assert mt_shap_explanation.ensure_data_type(data) is data

    pd_data = pd.DataFrame(data)
    new_data = mt_shap_explanation.ensure_data_type(pd_data)

    assert isinstance(new_data, np.ndarray)
    assert new_data.shape == data.shape

    data = torch.tensor(data)
    new_data = mt_shap_explanation.ensure_data_type(data)

    assert isinstance(new_data, np.ndarray)
    assert new_data.shape == data.shape

    incorrect_data = "invalid"
    with pytest.raises(TypeError):
        mt_shap_explanation.ensure_data_type(incorrect_data)


# def test_process_and_validate_explanations():
#     info = ValidationInfoMock({"data": np.random.rand(10, 10)})

#     # test case 1 - correct shape and type
#     explanations = shap.Explanation(np.random.rand(10, 10))
#     mt_shap_explanation.process_and_validate_explanations(explanations, info)

#     # test case 2 - correct shape, but multidimensional output
#     explanations = shap.Explanation(np.random.rand(10, 10, 2))
#     mt_shap_explanation.process_and_validate_explanations(explanations, info)

#     # test case 3 - incorrect type
#     with pytest.raises(TypeError):
#         # numpy input
#         mt_shap_explanation.process_and_validate_explanations(np.random.rand(10, 10), info)

#     # test case 4 - incorrect shape
#     with pytest.raises(mt_shap_explanation.ShapeMismatchError):
#         mt_shap_explanation.process_and_validate_explanations(shap.Explanation(np.random.rand(10, 9)), info)


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
