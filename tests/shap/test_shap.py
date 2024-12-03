import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split

import pytest
import shap

import meteors as mt
import torch
import numpy as np


def test_explainer_types():
    X_train, _, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    true_explanation_shape = (*X_train.shape, 3)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(knn.predict_proba, "classification")

    # invalid type
    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="invalid")

    # kernel explainer type
    kernel_explainer = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    kernel_explanation = kernel_explainer.explain(X_train)
    assert isinstance(kernel_explanation, mt.shap.SHAPExplanation)
    assert kernel_explanation.explanation_method == "kernel"
    assert kernel_explanation.data.shape == X_train.shape
    assert kernel_explanation.explanations.shape == true_explanation_shape

    linear = sklearn.linear_model.LogisticRegression()
    linear.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(linear, "classification")

    # linear explainer type
    linear_explainer = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")

    linear_explanation = linear_explainer.explain(X_train)
    assert isinstance(linear_explanation, mt.shap.SHAPExplanation)
    assert linear_explanation.explanation_method == "linear"
    assert linear_explanation.data.shape == X_train.shape
    assert linear_explanation.explanations.shape == true_explanation_shape

    # exact explainer type
    exact_explainer = mt.shap.HyperSHAP(explainable_model, X_train)

    exact_explanation = exact_explainer.explain(X_train)
    assert isinstance(exact_explanation, mt.shap.SHAPExplanation)
    assert exact_explanation.explanation_method == "exact"
    assert exact_explanation.data.shape == X_train.shape
    assert exact_explanation.explanations.shape == true_explanation_shape

    tree = sklearn.tree.DecisionTreeClassifier()
    tree.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(tree, "classification")

    # tree explainer type
    tree_explainer = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Tree")

    tree_explanation = tree_explainer.explain(X_train)
    assert isinstance(tree_explanation, mt.shap.SHAPExplanation)
    assert tree_explanation.explanation_method == "tree"
    assert tree_explanation.data.shape == X_train.shape
    assert tree_explanation.explanations.shape == true_explanation_shape

    # explainer type mismatch
    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")


def test_hyper_shap_init():
    # proper initialization
    X_train, _, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    linear = sklearn.linear_model.LogisticRegression()
    linear.fit(X_train, Y_train)
    explainable_model = mt.models.ExplainableModel(linear, "classification")
    mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")

    # invalid model
    invalid_model = "invalid"
    with pytest.raises(TypeError):
        mt.shap.HyperSHAP(invalid_model, X_train, explainer_type="Linear")

    # model as callable, but not ExplainableModel
    with pytest.raises(TypeError):
        mt.shap.HyperSHAP(linear, X_train, explainer_type="Linear")

    # not initialized explainer
    explainable_model = mt.models.ExplainableModel(sklearn.linear_model.LogisticRegression(), "classification")

    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Invalid")

    # incorrect data shape
    explainable_model = mt.models.ExplainableModel(linear, "classification")
    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, np.random.rand(10, 10), explainer_type="Linear", init=False)


def test_shap_explain():
    X_train, X_test, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    linear = sklearn.linear_model.LogisticRegression()
    linear.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(linear, "classification")

    explainer = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")

    # correct explanations
    explanations = explainer.explain(X_test)
    assert isinstance(explanations, mt.shap.SHAPExplanation)

    # incorrect data type
    with pytest.raises(TypeError):
        explainer.explain("invalid")

    # torch tensor input
    X_test_tensor = torch.tensor(X_test.to_numpy())
    explanations = explainer.explain(X_test_tensor)
