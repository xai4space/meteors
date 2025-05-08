import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap
import pytest
import numpy as np
import pandas as pd

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


@pytest.fixture
def synthetic_data_explainer():
    n_samples = 300
    # Generate feature 1: strong predictor

    feature1_class0 = np.random.normal(loc=-2.0, scale=1.0, size=n_samples // 2)
    feature1_class1 = np.random.normal(loc=2.0, scale=1.0, size=n_samples // 2)

    # Generate feature 2: moderate predictor
    feature2_class0 = np.random.normal(loc=-1.0, scale=1.5, size=n_samples // 2)
    feature2_class1 = np.random.normal(loc=1.0, scale=1.5, size=n_samples // 2)

    # Generate feature 3: weak predictor (mostly noise)
    feature3_class0 = np.random.normal(loc=-0.5, scale=2.0, size=n_samples // 2)
    feature3_class1 = np.random.normal(loc=0.5, scale=2.0, size=n_samples // 2)

    # Combine features
    X = np.vstack(
        [
            np.column_stack([feature1_class0, feature2_class0, feature3_class0]),
            np.column_stack([feature1_class1, feature2_class1, feature3_class1]),
        ]
    )

    # Create target variable (0 for first half, 1 for second half)
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = sklearn.ensemble.RandomForestRegressor(random_state=0)
    rf_model.fit(X_train, y_train)

    explainable_model = mt.models.ExplainableModel(rf_model, "classification")
    explainer = HyperSHAP(explainable_model, X_train, explainer_type="tree")

    explanation = explainer.explain(X_test)
    return explainable_model, (X_train, X_test, y_train, y_test), explainer, explanation


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
    assert target is None

    # test case 9 - single target required with only one target available
    target = validate_target(0, explanation)
    assert target is None


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
    mapping = {(0, 1): [0], 500.2: [1, 2, 3, 4]}
    with pytest.raises(TypeError, match="Expected numeric as key in wavelengths"):
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

    # Case 11: Incorrect feature index type
    mapping = {400.5: [0, 1], 600.3: ["invalid", 4]}
    with pytest.raises(TypeError, match="Expected int as feature index in mapping"):
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

    with pytest.raises(ValueError):
        shap_visualize.beeswarm(explainer, explanation, target=1, observation_index=0, use_pyplot=False)


def test_waterfall(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.waterfall(explainer, explanation, target=1, observation_index=0)

    plt.close(plt.gcf())

    # Case 3: Test invalid observation index
    with pytest.raises(ValueError):
        shap_visualize.waterfall(explainer, explanation, target=1, observation_index=None)


def test_heatmap(model_data_explainer, synthetic_data_explainer):
    _, (X_train, X_test, _, _), explainer, explanation = model_data_explainer

    shap_visualize.heatmap(
        explainer,
        explanation,
        target=0,
    )

    plt.close(plt.gcf())

    # test local explanation
    explanation = explainer.explain(X_test[0:1])
    with pytest.raises(ValueError):
        shap_visualize.heatmap(explainer, explanation, target=0)

    # test explanation with no feature names

    _, (X_train, X_test, _, _), explainer, explanation = synthetic_data_explainer

    shap_visualize.heatmap(
        explainer,
        explanation,
        target=0,
    )

    plt.close(plt.gcf())


def test_bar(model_data_explainer):
    _, _, explainer, explanation = model_data_explainer

    shap_visualize.bar(explainer, explanation, target=0, use_pyplot=False)

    shap_visualize.bar(explainer, explanation, target=1, observation_index=1, use_pyplot=False)


def test_dependence(model_data_explainer, monkeypatch):
    # Unpack the fixture
    _, (X_train, X_test, _, _), explainer, explanation = model_data_explainer

    # Prepare display features for testing
    display_features = pd.DataFrame(X_train, columns=["sepal length", "sepal width", "petal length", "petal width"])

    # TEST CASE 1: Basic functionality with feature index
    # Verify plot generation with default parameters using feature index
    ax1 = shap_visualize.dependence_plot(feature=0, explainer=explainer, target=0, explanation=explanation)
    assert ax1 is not None, "Failed to generate plot with feature index"
    plt.close()  # Close the plot to prevent display

    # TEST CASE 2: Feature specified by name
    # Check plot generation when feature is specified by name
    ax2 = shap_visualize.dependence_plot(
        feature="sepal length (cm)", explainer=explainer, target=0, explanation=explanation
    )
    assert ax2 is not None, "Failed to generate plot with feature name"
    plt.close()

    # TEST CASE 3: Interaction index scenarios
    # Test with explicit interaction index
    ax3 = shap_visualize.dependence_plot(
        feature=0, explainer=explainer, explanation=explanation, target=0, interaction_index=1
    )
    assert ax3 is not None, "Failed to generate plot with explicit interaction index"
    plt.close()

    # TEST CASE 4: Interaction index as 'auto'
    ax4 = shap_visualize.dependence_plot(
        feature=0, explainer=explainer, explanation=explanation, target=0, interaction_index="auto"
    )
    assert ax4 is not None, "Failed to generate plot with auto interaction index"
    plt.close()

    # TEST CASE 5: Custom display features
    ax5 = shap_visualize.dependence_plot(
        feature=0, explainer=explainer, explanation=explanation, target=0, display_features=display_features
    )
    assert ax5 is not None, "Failed to generate plot with display features"
    plt.close()

    # TEST CASE 6: Comprehensive styling
    ax6 = shap_visualize.dependence_plot(
        feature=0,
        explainer=explainer,
        explanation=explanation,
        target=0,
        color="#FF0000",  # Red color
        axis_color="#00FF00",  # Green axis
        cmap="viridis",  # Specific colormap
        dot_size=20,  # Custom dot size
        x_jitter=0.2,  # Add jitter
        alpha=0.5,  # Transparency
        title="Test Dependence Plot",
    )
    assert ax6 is not None, "Failed to generate plot with comprehensive styling"
    plt.close()

    # TEST CASE 7: Percentile-based axis bounds
    ax7 = shap_visualize.dependence_plot(
        feature=0,
        explainer=explainer,
        explanation=explanation,
        target=0,
        xmin="percentile(10)",
        xmax="percentile(90)",
        ymin="percentile(5)",
        ymax="percentile(95)",
    )
    assert ax7 is not None, "Failed to generate plot with percentile-based bounds"
    plt.close()

    # TEST CASE 8: Use pyplot flag
    # Temporarily mock pyplot to verify behavior
    def mock_show():
        pass

    monkeypatch.setattr(plt, "show", mock_show)
    result = shap_visualize.dependence_plot(
        feature=0, explainer=explainer, explanation=explanation, target=0, use_pyplot=True
    )
    assert result is None, "use_pyplot=True should return None"

    # ERROR HANDLING TEST CASES

    # TEST CASE 9: Invalid feature type
    with pytest.raises(TypeError, match="Expected int or str as feature"):
        shap_visualize.dependence_plot(
            feature=1.5,  # Invalid type
            explainer=explainer,
            explanation=explanation,
            target=0,
        )

    # TEST CASE 10: Invalid interaction index type
    with pytest.raises(TypeError, match="Expected str or int as interaction_index"):
        shap_visualize.dependence_plot(
            feature=0,
            explainer=explainer,
            explanation=explanation,
            interaction_index=[1, 2],  # Invalid type
            target=0,
        )

    # TEST CASE 11: Invalid display features type
    with pytest.raises(TypeError, match="Expected pd.DataFrame or np.ndarray as display_features"):
        shap_visualize.dependence_plot(
            feature=0,
            explainer=explainer,
            explanation=explanation,
            display_features={"invalid": "type"},
            target=0,
        )

    # Simulate a local explanation to test local explanation error
    # TEST CASE 12: Local explanation error
    local_explanation = explainer.explain(X_test[0:1])
    with pytest.raises(ValueError, match="The dependence plot requires a global explanation"):
        shap_visualize.dependence_plot(feature=0, explainer=explainer, explanation=local_explanation, target=0)

    # TEST CASE 13: Missing target
    with pytest.raises(ValueError):
        shap_visualize.dependence_plot(feature=0, explainer=explainer, explanation=local_explanation)


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

    # Case 13: Empty transformations mapping
    with pytest.raises(ValueError):
        shap_visualize.wavelengths_bar(
            explainer,
            explanation,
            target=0,
            wavelengths_mapping=wavelengths_mapping,
            transformations_mapping={},
            use_pyplot=False,
        )

    # Case 14: One target, one class to predict
    X_train = np.random.rand(10, 5)
    Y_train = X_train[:, 0] + X_train[:, 1] + np.random.rand(10) * 0.1

    X_test = np.random.rand(10, 5)

    linear = sklearn.linear_model.LinearRegression()
    linear.fit(X_train, Y_train)

    explainable_model = mt.models.ExplainableModel(linear, "regression")
    explainer = HyperSHAP(explainable_model, X_train, explainer_type="Linear")
    explanation = explainer.explain(X_test)

    wavelengths_mapping = {400: [0, 1, 2], 500: [3, 4]}

    shap_visualize.wavelengths_bar(
        explainer, explanation, target=None, wavelengths_mapping=wavelengths_mapping, use_pyplot=False
    )

    plt.close(plt.gca().figure)

    # different target specification
    shap_visualize.wavelengths_bar(
        explainer, explanation, target=0, wavelengths_mapping=wavelengths_mapping, use_pyplot=False
    )
    plt.close(plt.gca().figure)
