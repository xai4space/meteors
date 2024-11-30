import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap
import pytest
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from meteors.shap import HyperSHAP
import meteors.visualize.shap_visualize as shap_visualize
from meteors.models import ExplainableModel


from meteors.visualize.shap_visualize import (
    validate_explanations_and_explainer_type,
    validate_observation_index,
    validate_target,
    validate_mapping_dict,
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
    index = validate_observation_index(None, explanation, require_local_explanation=False)
    assert index is None

    # test case 4 - None index when explanation is not for single observation and require_single_observation is True
    with pytest.raises(ValueError):
        validate_observation_index(
            None,
            explanation,
            require_local_explanation=True,
        )

    # test case 5 - incorrect type
    with pytest.raises(TypeError):
        validate_observation_index("10", explanation)

    # test case 6 - single observation with correct index
    explanation = explainer.explain(X_test.iloc[0])
    index = validate_observation_index(0, explanation, require_local_explanation=True)
    assert index == 0
    index = validate_observation_index(None, explanation, require_local_explanation=True)
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


def test_validate_mapping_dict():
    explanation_values = np.random.rand(10, 5)  # 10 samples, 5 features

    # Case 1: Valid wavelength mapping
    mapping = {400.5: [0, 1, 3, 4], 500.2: 2}
    expected = {400.5: [0, 1, 3, 4], 500.2: [2]}
    result = validate_mapping_dict(mapping, explanation_values)
    assert result == expected, "Failed to validate a correct wavelength mapping"

    # Case 2: Valid transformation mapping
    mapping = {"log_transform": [0, 1, 3, 4], "sqrt_transform": 2}
    expected = {"log_transform": [0, 1, 3, 4], "sqrt_transform": [2]}
    result = validate_mapping_dict(mapping, explanation_values, wavelengths=False)
    assert result == expected, "Failed to validate a correct transformation mapping"

    # Case 3: Invalid mapping type
    with pytest.raises(TypeError, match="Expected dict as mapping"):
        validate_mapping_dict("invalid_mapping", explanation_values)

    # Case 4: Invalid explanation values type
    with pytest.raises(TypeError, match="Expected np.ndarray as explantion_values"):
        validate_mapping_dict({}, "invalid_explanation_values")

    # Case 5: Invalid key type in wavelengths
    mapping = {"invalid_key": [0], 500.2: [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="Expected numeric as key in wavelengths"):
        validate_mapping_dict(mapping, explanation_values)

    # Case 6: Invalid key type in transformation mapping
    mapping = {123: [0, 1, 2, 3, 4]}
    with pytest.raises(TypeError, match="Expected str as key in mapping"):
        validate_mapping_dict(mapping, explanation_values, wavelengths=False)

    # Case 7: Invalid value type
    mapping = {400.5: "invalid_value", 500.2: [1, 2, 3, 4]}
    with pytest.raises(TypeError, match="Expected list or a single integer as value in mapping"):
        validate_mapping_dict(mapping, explanation_values)

    # Case 8: Feature index out of bounds
    mapping = {400.5: [0, 6]}
    with pytest.raises(ValueError, match="Feature index out of bounds"):
        validate_mapping_dict(mapping, explanation_values)

    # Case 9: Duplicate feature index
    mapping = {400.5: [0, 1], 500.2: [1, 2], 600.3: [3, 4]}
    with pytest.raises(ValueError, match="Feature index .* already used in another entry"):
        validate_mapping_dict(mapping, explanation_values)

    # Case 10: Missing features in mapping
    mapping = {400.5: [0, 1]}
    with pytest.raises(ValueError, match="Not all features are used in the mapping"):
        validate_mapping_dict(mapping, explanation_values)


def test_force(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    fig = shap_visualize.force(explainer, explanation, observation_index=0, target=1)

    with pytest.raises(ValueError):
        fig = shap_visualize.force(explainer, explanation, observation_index=0, target=10)

    with pytest.raises(ValueError):
        fig = shap_visualize.force(explainer, explanation, observation_index=100, target=0)

    plt.close(fig)


def test_beeswarm(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.beeswarm(explainer, explanation, target=1, use_pyplot=False)


def test_waterfall(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.waterfall(explainer, explanation, target=1, observation_index=0)


def test_heatmap(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.heatmap(
        explainer,
        explanation,
        target=0,
    )


def test_bar(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.bar(explainer, explanation, target=0, use_pyplot=False)

    shap_visualize.bar(explainer, explanation, target=1, observation_index=1, use_pyplot=False)


def test_dependence(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.dependence_plot(1, explainer, explanation, target=0, use_pyplot=False)

    fig, ax = plt.subplots()

    shap_visualize.dependence_plot(1, explainer, explanation, target=1, ax=ax, use_pyplot=False)


def test_wavelengths_bar(model_data_explainer):
    # Turn off interactive mode for testing
    plt.ioff()

    _, _, explainer, explanation = model_data_explainer

    # Case 1: Valid input with wavelengths mapping only
    wavelengths_mapping = {400: [0], 410: [2, 3, 1]}
    ax = shap_visualize.wavelengths_bar(
        explainer, explanation, target=0, wavelengths_mapping=wavelengths_mapping, use_pyplot=False
    )
    assert isinstance(ax, plt.Axes), "Failed to return an Axes object with valid input."
    plt.close(ax.figure)
    del ax

    # Case 2: Valid input with transformations mapping
    transformations_mapping = {"log_transform": [0, 2], "sqrt_transform": [1, 3]}
    ax = shap_visualize.wavelengths_bar(
        explainer,
        explanation,
        target=0,
        wavelengths_mapping=wavelengths_mapping,
        transformations_mapping=transformations_mapping,
        use_pyplot=False,
    )
    assert isinstance(ax, plt.Axes), "Failed to return an Axes object with valid transformations mapping."
    plt.close(ax.figure)
    del ax

    # Case 3: Valid input with switched flags
    ax = shap_visualize.wavelengths_bar(
        explainer,
        explanation,
        target=0,
        wavelengths_mapping=wavelengths_mapping,
        transformations_mapping=transformations_mapping,
        average_the_same_bands=True,
        separate_features=True,
        use_pyplot=False,
    )
    assert isinstance(ax, plt.Axes), "Failed to return an Axes object with valid transformations mapping."
    plt.close(ax.figure)
    del ax

    # Case 4: Invalid wavelengths_mapping type
    with pytest.raises(TypeError, match="Expected dict as mapping"):
        shap_visualize.wavelengths_bar(explainer, explanation, target=0, wavelengths_mapping="invalid_mapping")

    # Case 5: Missing wavelengths_mapping
    with pytest.raises(ValueError, match="Wavelengths mapping must be provided"):
        shap_visualize.wavelengths_bar(
            explainer,
            explanation,
            target=0,
        )

    # Case 6: Invalid transformations_mapping type
    with pytest.raises(TypeError, match="Expected dict as mapping"):
        shap_visualize.wavelengths_bar(
            explainer,
            explanation,
            target=0,
            wavelengths_mapping=wavelengths_mapping,
            transformations_mapping="invalid_mapping",
        )

    # Case 7: Feature index out of bounds in wavelengths_mapping
    wavelengths_mapping_invalid = {400: [0, 6]}
    with pytest.raises(ValueError, match="Feature index out of bounds"):
        shap_visualize.wavelengths_bar(
            explainer, explanation, target=0, wavelengths_mapping=wavelengths_mapping_invalid
        )

    # Case 8: Feature index out of bounds in transformations_mapping
    transformations_mapping_invalid = {"log_transform": [0, 6]}
    with pytest.raises(ValueError, match="Feature index out of bounds"):
        shap_visualize.wavelengths_bar(
            explainer,
            explanation,
            target=0,
            wavelengths_mapping=wavelengths_mapping,
            transformations_mapping=transformations_mapping_invalid,
        )

    # Case 9: Valid custom colormap
    custom_cmap = ListedColormap(["red", "green", "blue"])
    ax = shap_visualize.wavelengths_bar(
        explainer,
        explanation,
        target=0,
        wavelengths_mapping=wavelengths_mapping,
        transformations_mapping=transformations_mapping,
        cmap=custom_cmap,
        use_pyplot=False,
    )
    assert isinstance(ax, plt.Axes), "Failed to return an Axes object with valid transformations mapping."
    plt.close(ax.figure)
    del ax

    # Case 10: Incorrect colormap
    with pytest.raises(ValueError):
        one_color_cmap = ListedColormap(["red"])
        shap_visualize.wavelengths_bar(
            explainer,
            explanation,
            target=0,
            wavelengths_mapping=wavelengths_mapping,
            transformations_mapping=transformations_mapping,
            cmap=one_color_cmap,
            use_pyplot=False,
        )

    # Case 11: Separate features with no transformations mapping
    ax = shap_visualize.wavelengths_bar(
        explainer,
        explanation,
        target=1,
        wavelengths_mapping=wavelengths_mapping,
        separate_features=True,
        use_pyplot=False,
    )
    assert isinstance(ax, plt.Axes), "Failed to return Axes with separate_features enabled."
    plt.close(ax.figure)
    del ax

    # Case 12: Use pyplot to display the plot
    # This test ensures no errors occur during plotting but does not verify visual correctness
    shap_visualize.wavelengths_bar(
        explainer, explanation, target=0, wavelengths_mapping=wavelengths_mapping, use_pyplot=False
    )
    plt.close(plt.gca().figure)
