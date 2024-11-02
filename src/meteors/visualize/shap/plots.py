from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from meteors.shap import HyperSHAP, SHAPExplanation

import shap
import numpy as np
import pandas as pd


def validate_observation_index(
    observation_index: int | None, explanation: SHAPExplanation, require_local_explanation: bool = False, plot_type=None
) -> int | None:
    """function validates the observation index for the explanation.
    It checks if the observation index is an integer or None and within the bounds of the explanation.
    In case the single observation is required (local explanation), the function checks if the selected observation can be determined (the input explanation is also local).

    In case the single observation is not required (global explanation), the function allows for None as the observation index, which means that the explanation is for all observations.

    Args:
        observation_index (int | None): a selected observation index. Defaults to None.
        explanation (SHAPExplanation): An explanation object, from which the observations will be selected.
        require_local_explanation (bool, optional): A flag that determines whether the plot function accepts global explanations. Defaults to False.
        plot_type (str, optional): The type of the plot. Defaults to None. Used to print the customized ValueError message.

    Raises:
        TypeError: In case the observation index is not an integer or None.
        ValueError: Whether the integer observation index is out of bounds.
        ValueError: if the observation index is not specified, but explanation contains multiple observations and the plot supports only local explanations.

    Returns:
        int | None: parsed observation index.
    """
    if not isinstance(observation_index, int) and observation_index is not None:
        raise TypeError(f"Expected int or None as observation index, but got {type(observation_index)}")
    if isinstance(observation_index, int):
        if explanation.is_local_explanation and observation_index != 0:
            raise ValueError(
                "Passed explanation is for a single observation only - index must be 0 or None for single observation explanations"
            )
        if not explanation.is_local_explanation and (
            observation_index < 0 or observation_index >= explanation.data.shape[0]
        ):
            raise ValueError(
                f"Observation index out of bounds: {observation_index}. The explanation contains {explanation.explanations.values.shape[0]} observations"
            )

    if require_local_explanation:
        if observation_index is None and not explanation.is_local_explanation:
            raise ValueError(
                f"The plot of type {plot_type} only supports local explanations. \nPassed explanation contains multiple observations and no observation index specified."
            )
        if observation_index is None:
            observation_index = 0

    return observation_index


def validate_target(
    target: int | None, explanation: SHAPExplanation, require_single_target: bool = False, plot_type: str | None = None
) -> int | None:
    """function validates the target value for the explanation.
    It checks if the target is an integer or None and within the bounds of the explanation.
    In case the plot requires a single target value, the function checks if the target is specified and if the explanation contains multiple targets.

    Args:
        target (int | None): a target value index. Defaults to None.
        explanation (SHAPExplanation): explanation object, from which the target value will be selected.
        require_single_target (bool, optional): If the plot requires single target output from the model. Defaults to False.
        plot_type (str, optional): A type of the plot. Defaults to None.

    Raises:
        TypeError: If the target is not an integer or None.
        ValueError: If the target index is out of bounds.
        ValueError: If target value is not passed, the model outputs multiple values and the plot requires a single target value.

    Returns:
        int | None: the preprocessed target index
    """
    if target is not None and not isinstance(target, int):
        raise TypeError(f"Expected int or None as target, but got {type(target)}")

    if target is not None and (target < 0 or target >= explanation.target_dims):
        raise ValueError(f"Target index out of bounds: {target}")

    if require_single_target:
        if target is None and explanation.target_dims > 1:
            raise ValueError(
                f"The plot of type {plot_type} requires a single target value. \nPassed explanation contains multiple targets and no target index specified."
            )
        if target is None:
            target = 0

    return target


def validate_explanations_and_explainer_type(explainer: HyperSHAP, explanation: SHAPExplanation) -> None:
    """a simple function that validates the explainer and explanation types.

    Args:
        explainer (HyperSHAP): an explainer object (HyperSHAP).
        explanation (SHAPExplanation): an explanation object (SHAPExplanation).

    Raises:
        TypeError: If the explainer is not HyperSHAP type.
        TypeError: If the explanation is not SHAPExplanation type.
    """
    if not isinstance(explainer, HyperSHAP):
        raise TypeError(f"Expected HyperSHAP as explainer, but got {type(explainer)}")
    if not isinstance(explanation, SHAPExplanation):
        raise TypeError(f"Expected HyperSHAP as explanation, but got {type(explanation)}")


def force(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
    **kwargs,
) -> Figure | None:
    """Visualize the given SHAP explanation using the force plot. The function utilizes the `shap.plots.force` function, which reference might be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html
    Force plot works for local explanations.

    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        observation_index (int | None, optional): An index of observation that should be locally explained. In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.


    Returns:
       matplotlib.figure.Figure | None:
            If use_pyplot is False, returns the figure object.
            If use_pyplot is True, returns None .
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)
    observation_index = validate_observation_index(observation_index, explanation, require_local_explanation=True)

    explanations_values = explanation.explanations.values
    feature_values = explanation.data
    if not explanation.is_local_explanation:
        explanations_values = explanations_values[observation_index]
        feature_values = feature_values[observation_index]
    if target is not None:
        explanations_values = explanations_values[..., target]

    fig = shap.plots.force(
        base_value=explainer._explainer.expected_value[observation_index],
        shap_values=explanations_values,
        features=feature_values,
        feature_names=explanation.feature_names,
        matplotlib=True,
        show=use_pyplot,
        **kwargs,
    )
    return fig


def beeswarm(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
    ax: Axes | None = None,
) -> Axes | None:
    """Create a beeswarm plot for the given explanation. A reference for the plot might be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html


    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        observation_index (int | None, optional): An index of observation that should be locally explained. This plot supports also global explanations, so this value is not necessary and can be set to None.
        In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be displayed on the passed axes.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)
    observation_index = validate_observation_index(None, explanation, require_local_explanation=False)

    explanations = explanation.explanations
    if target is not None:
        explanations = explanations[..., target]
    if observation_index is not None:
        explanations = explanations[observation_index]

    # fig = shap.plots.beeswarm(explanations, ax=ax, show=use_pyplot)
    # Current release of SHAP does not support passing ax parameter to the beeswarm plot, even though it is present in the documentation.
    ax = shap.plots.beeswarm(explanations, show=use_pyplot)
    return ax


def dependence_plot(
    feature: int | str,
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    interaction_index: str = "auto",
    display_features: np.ndarray | pd.DataFrame | None = None,
    color: str = "#1E88E5",
    axis_color: str = "#333333",
    cmap: str | Colormap = None,
    dot_size: int = 16,
    x_jitter: float = 0,
    alpha: float = 1,
    title: str | None = None,
    xmin: float | str | None = None,
    xmax: float | str | None = None,
    ymin: float | str | None = None,
    ymax: float | str | None = None,
    ax: Axes | None = None,
    use_pyplot: bool = True,
) -> Axes | None:
    """Create a SHAP dependence plot for the given feature, colored by an interaction feature.

    This plot is useful to understand how two features interact with each other influencing the model's behaviour.


    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extension of the classical partial dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.

    Args:
        feature (int | str): A selected feature to be plotted. If this is an int it is the index of the feature to plot in the data provided. In case the string is provided, it could be either the name of the feature to plot, or it can have the form `rank(int)`, where `int` is the rank of the feature in the feature importance ranking.
        explainer (HyperSHAP): a HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): an explanation object coming from the meteors package. This object contains the explanations for the model. It should contain the global explanation for the plot to make sense.
        target (int, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        interaction_index (str, optional): The feature name or index of the feature to color by. Defaults to "auto", which means that the feature with the highest interaction value will be selected. If None, the plot will not be colored by any other feature importance. In case a feature index is provided, the interactions of the selected feature with the main feature will be calculated.
        display_features (np.ndarray, pd.DataFrame, optional): An object used to decode the feature names in a more human-readable form. Defaults to None.
        color (str, optional): The color of the dots on the plot used in case `interaction_index` is None. Defaults to "#1E88E5".
        axis_color (str, optional): The color of the axis and labels. It makes the plot a little more readable. Defaults to "#333333".
        cmap (str | matplotlib.colors.Colormap, optional): The name of the colormap or the matplotlib colormap to use. Defaults to None, which means that the default colormap will be used - matplotlib.colors.red_blue
        dot_size (int, optional): The size of the dots on the plot. Defaults to 16.
        x_jitter (float, optional): Adds random jitter to feature values. Should be in a (0, 1) range, where 0 means no jitter. Could improve plot's readability in case the selected feature is discrete. Defaults to 0.
        alpha (float, optional): The transparency of the dots on the plot. Defaults to 1.
        title (str, optional): The title of the plot. Defaults to None.
        xmin (float | str, optional): Represents the lower bound of the plot's x-axis. It can also be a string of the format `percentile(float)` that percentile of the feature's value used on the x-axis. Defaults to None, which means that the minimum value of the feature will be used.
        xmax (float | str, optional): Represents the upper bound of the plot's x-axis. It can also be a string of the format `percentile(float)` that percentile of the feature's value used on the x-axis. Defaults to None, which means that the maximum value of the feature will be used.
        ymin (float | str, optional): Represents the lower bound of the plot's y-axis. It can also be a string of the format `percentile(float)` that percentile of the feature's value used on the y-axis. Defaults to None, which means that the minimum value of the feature will be used.
        ymax (float | str, optional): Represents the upper bound of the plot's y-axis. It can also be a string of the format `percentile(float)` that percentile of the feature's value used on the y-axis. Defaults to None, which means that the maximum value of the feature will be used.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be displayed on the passed axes. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to True.

    Raises:
        ValueError: Raised in case the explanation is not global or is multitarget, and no target to explain is specified.
        TypeError: Raised in case any of the passed arguments is of the wrong type.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)

    explanation_values = explanation.explanations.values
    if target is not None:
        explanation_values = explanation_values[..., target]

    if explanation_values.ndim == 2 and target is None and not explanation.is_local_explanation:
        raise ValueError("The dependence plot requires a single target value.")

    if explanation.is_local_explanation:
        raise ValueError("The dependence plot requires a global explanation.")

    if not isinstance(feature, int) and not isinstance(feature, str):
        raise TypeError(f"Expected int or str as feature, but got {type(feature)}")

    if (
        interaction_index is not None
        and not isinstance(interaction_index, str)
        and not isinstance(interaction_index, int)
    ):
        raise TypeError(f"Expected str or int as interaction_index, but got {type(interaction_index)}")

    if (
        display_features is not None
        and not isinstance(display_features, pd.DataFrame)
        and not isinstance(display_features, np.ndarray)
    ):
        raise TypeError(f"Expected pd.DataFrame or np.ndarray as display_features, but got {type(display_features)}")

    ax = shap.dependence_plot(
        ind=feature,
        shap_values=explanation_values,
        features=explanation.data,
        feature_names=explanation.feature_names,
        display_features=display_features,
        interaction_index=interaction_index,
        color=color,
        axis_color=axis_color,
        cmap=cmap,
        dot_size=dot_size,
        x_jitter=x_jitter,
        alpha=alpha,
        title=title,
        xmin=xmin,
        xmax=xmax,
        ax=ax,
        show=use_pyplot,
        ymin=ymin,
        ymax=ymax,
    )
    return ax


def waterfall(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
) -> Axes | None:
    """Create a waterfall chart for the passed SHAP explanation.
    The reference on this plot may be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html

    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        observation_index (int | None, optional): An index of observation that should be locally explained. In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    observation_index = validate_observation_index(observation_index, explanation, require_local_explanation=True)
    target = validate_target(target, explanation, require_single_target=True)

    explanations_values = explanation.explanations[observation_index]
    if target is not None:
        explanations_values = explanations_values[..., target]

    ax = shap.plots.waterfall(
        explanations_values,
        show=use_pyplot,
    )
    return ax


def bar(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
    ax: Axes | None = None,
    **kwargs,
):  # TODO: add aggregation
    """Creates a bar plot for the given SHAP explanation. The function utilizes the `shap.plots.bar` function, which reference might be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.bar.html

    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        observation_index (int | None, optional): An index of observation that should be locally explained. This plot supports also global explanations, so this value is not necessary and can be set to None.
        In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be displayed on the passed axes.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)

    explanation_values = explanation.explanations
    if target is not None:
        explanation_values = explanation_values[..., target]
    if observation_index is not None:
        explanation_values = explanation_values[observation_index]

    ax = shap.plots.bar(explanation_values, show=use_pyplot, **kwargs)
    return ax


def heatmap(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Create a heatmap plot of a set of SHAP values.
    The function utilizes the `shap.plots.heatmap` function,
    which reference might be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html

    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        observation_index (int | None, optional): An index of observation that should be locally explained. This plot supports also global explanations, so this value is not necessary and can be set to None.
        In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be displayed on the passed axes.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)

    explanation_values = explanation.explanations
    if target is not None:
        explanation_values = explanation_values[..., target]
    if observation_index is not None:
        explanation_values = explanation_values[observation_index]

    ax = shap.plots.heatmap(explanation_values, show=use_pyplot, ax=ax, **kwargs)
    return ax


def partial_dependence_plot(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    raise NotImplementedError("The function is not implemented yet.")
