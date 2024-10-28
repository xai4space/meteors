import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap
import pytest

import matplotlib.pyplot as plt

from meteors.shap import HyperSHAP
import meteors.visualize.shap as shap_visualize
from meteors.models import ExplainableModel


from meteors.visualize.shap.plots import (
    validate_explanations_and_explainer_type,
    validate_observation_index,
    validate_target,
)

import meteors as mt


@pytest.fixture
def model_and_data():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    model = ExplainableModel(forward_func=knn.predict_proba, problem_type="regression")

    return model, (X_train, X_test, Y_train, Y_test)


@pytest.fixture
def model_data_explainer():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(knn.predict_proba, "classification")

    explainer = HyperSHAP(explainable_model, X_train, explainer_type="Kernel")
    explanation = explainer.explain(X_test)

    return explainable_model, (X_train, X_test, Y_train, Y_test), explainer, explanation


def test_validate_explanations_and_explainer_type(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    validate_explanations_and_explainer_type(explainer, explanation)

    with pytest.raises(TypeError):
        validate_explanations_and_explainer_type(explainer, 10)

    with pytest.raises(TypeError):
        validate_explanations_and_explainer_type(10, explanation)


def test_validate_observation_index(model_data_explainer):
    _, (_, X_test, _, _), explainer, explanation = model_data_explainer

    # test case 1 - correct index
    validate_observation_index(0, explanation)

    # test case 2 - incorrect integer index
    with pytest.raises(ValueError):
        validate_observation_index(len(X_test) + 1, explanation)

    # test case 3 - None index when multiple observations allowed
    index = validate_observation_index(None, explanation, require_single_observation=False)
    assert index is None

    # test case 4 - None index when explanation is not for single observation and require_single_observation is True
    with pytest.raises(ValueError):
        validate_observation_index(None, explanation, require_single_observation=True)

    # test case 5 - incorrect type
    with pytest.raises(TypeError):
        validate_observation_index("10", explanation)

    # test case 6 - single observation with correct index
    explanation = explainer.explain(X_test.iloc[0])
    index = validate_observation_index(0, explanation, require_single_observation=True)
    assert index == 0
    index = validate_observation_index(None, explanation, require_single_observation=True)
    assert index == 0

    # test case 7 - single observation with incorrect index
    with pytest.raises(ValueError):
        validate_observation_index(1, explanation)


def test_validate_target(model_data_explainer):
    _, (_, X_test, _, _), explainer, explanation = model_data_explainer

    # test case 1 - correct index
    validate_target(0, explanation)

    # test case 2 - incorrect integer index
    with pytest.raises(ValueError):
        validate_target(explanation.explanations.shape[2] + 1, explanation)

    # test case 3 - None index
    validate_target(None, explanation)

    # test case 4 - incorrect type
    with pytest.raises(TypeError):
        validate_target("10", explanation)

    # test case 5 - target for single observation
    explanation = explainer.explain(X_test.iloc[0])
    validate_target(2, explanation)

    # test case 6 - single target value required
    validate_target(0, explanation, require_single_target=True)

    # test case 7 - single target value required but multiple targets present
    with pytest.raises(ValueError):
        validate_target(None, explanation, require_single_target=True)

    # test case 8 - single target required with only one target available
    explanation.explanations = explanation.explanations[..., 0]

    target = validate_target(None, explanation, require_single_target=True)
    assert target == 0


def test_force(model_and_data):
    explainable_model, (X_train, X_test, Y_train, Y_test) = model_and_data

    explainer = HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    explanation = explainer.explain(X_test)

    fig = shap_visualize.force(explainer, explanation, observation_index=0, target=1)

    with pytest.raises(ValueError):
        fig = shap_visualize.force(explainer, explanation, observation_index=0, target=10)

    with pytest.raises(ValueError):
        fig = shap_visualize.force(explainer, explanation, observation_index=100, target=10)

    plt.close(fig)


def test_beeswarm():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(knn.predict_proba, "classification")

    explainer = HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    explanation = explainer.explain(X_test)

    shap_visualize.beeswarm(explainer, explanation, target=1)
