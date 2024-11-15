import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split

import pytest
import shap

import meteors as mt


def simple_explanation_test():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(knn.predict_proba, "classification")

    hyper_shap = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    explanation = hyper_shap.explain(X_test)

    assert isinstance(explanation, mt.shap.SHAPExplanation)

    with pytest.raises(TypeError):
        mt.shap.HyperSHAP("invalid", X_train, explainer_type="Kernel")


def test_explainer_types():
    X_train, _, Y_train, _ = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(knn.predict_proba, "classification")

    # invalid type
    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="invalid")

    # kernel explainer type
    mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    linear = sklearn.linear_model.LogisticRegression()
    linear.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(linear, "classification")

    # linear explainer type
    mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")
    # exact explainer type
    mt.shap.HyperSHAP(explainable_model, X_train)

    tree = sklearn.tree.DecisionTreeClassifier()
    tree.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(tree, "classification")

    # tree explainer type
    mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Tree")

    # explainer type mismatch
    with pytest.raises(ValueError):
        mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Linear")


def test_shap_explain():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

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
