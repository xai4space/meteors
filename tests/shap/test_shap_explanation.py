import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap
import numpy as np

import meteors as mt


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
