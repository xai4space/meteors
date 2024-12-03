from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from meteors.shap import HyperSHAP, SHAPExplanation

import shap
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from loguru import logger


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

    if target is not None and (target < 0 or target >= explanation.num_target_outputs):
        raise ValueError(f"Target index out of bounds: {target}")

    if target == 0 and explanation.num_target_outputs == 1:
        if len(explanation.explanations.shape) == 2:
            logger.debug(
                "Detected explanation for a single target based on the explanation shape validation. Coercing the target to None."
            )
            target = None

    if require_single_target:
        if target is None and explanation.num_target_outputs > 1:
            raise ValueError(
                f"The plot of type {plot_type} requires a single target value. \nPassed explanation contains multiple targets and no target index specified."
            )

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


def validate_mapping_dict(
    mapping: dict[float | int | str, list[int] | int], explanation_values: np.ndarray, wavelengths: bool = True
) -> dict[float | str, list[int]]:
    """validates the mapping dictionary. The function checks if the fields of the dictionary are in the correct format, coercing it to the correct types if possible.
    Later, it checks if all the features are used in the mapping, and returns the parsed dictionary.


    Args:
        mapping (dict[float | int | str, list[int] | int]): Wavelengths mapping and transformation mapping should be in a format: {band_wavelength | transformation_function_name: [feature_index, ...] | single_feature_index, ...}.
        Transformation function name should be a string.
        In case the wavelength is a string, there will be an attempt to convert it to a float.
        explanation_values (np.ndarray): explanation values for the model from the shap.Explanations object.
        wavelengths (bool, optional): If True, the function will proceed with the check for wavelengths mapping. Otherwise it will perform check for feature aggregation. Defaults to True.

    Raises:
        TypeError: Raised if mappings or explanation values are not in the correct format.

    Returns:
        mapping: dict[float, list[int]]: validated mapping dictionary.
    """

    # check if the mapping are in the correct format
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected dict as mapping, but got {type(mapping)}")
    if not isinstance(explanation_values, np.ndarray):
        raise TypeError(f"Expected np.ndarray as explantion_values, but got {type(explanation_values)}")

    used_indices = set()
    ncols = explanation_values.shape[1]

    parsed_mapping = {}

    for key, value in mapping.items():
        if wavelengths:
            if isinstance(key, str):
                try:
                    key = float(key)
                except ValueError:
                    raise ValueError(f"Expected numeric as key in wavelengths, but got {key}")
            if not isinstance(key, float) and not isinstance(key, int):
                raise TypeError(f"Expected numeric as key in wavelengths, but got {type(key)}")
        else:
            if not isinstance(key, str):
                raise TypeError(f"Expected str as key in mapping, but got {type(key)}")
        if not isinstance(value, list) and not isinstance(value, int):
            raise TypeError(f"Expected list or a single integer as value in mapping, but got {type(value)}")

        if not isinstance(value, list):
            value = [value]

        for feature_index in value:
            if not isinstance(feature_index, int):
                raise TypeError(f"Expected int as feature index in mapping, but got {type(feature_index)}")
            if feature_index < 0 or feature_index >= explanation_values.shape[1]:
                raise ValueError(f"Feature index out of bounds: {feature_index}")
            if feature_index in used_indices:
                raise ValueError(f"Feature index {feature_index} already used in another entry of the aggregation.")
            used_indices.add(feature_index)

        parsed_mapping[key] = value

    if len(used_indices) != ncols:
        raise ValueError(
            f"Not all features are used in the mapping. There are {ncols} features, but only {len(used_indices)} are used in the mapping."
        )

    return parsed_mapping


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
    target = validate_target(target, explanation, require_single_target=True, plot_type="force")
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
    target = validate_target(target, explanation, require_single_target=True, plot_type="beeswarm")
    observation_index = validate_observation_index(observation_index, explanation, require_local_explanation=False)

    explanations = explanation.explanations
    if target is not None:
        explanations = explanations[..., target]
    if observation_index is not None or explanation.is_local_explanation:
        raise ValueError("The beeswarm plot does not support local explanations.")

    # fig = shap.plots.beeswarm(explanations, ax=ax, show=use_pyplot)
    # Current release of SHAP does not support passing ax parameter to the beeswarm plot, even though it is present in the documentation.
    return shap.plots.beeswarm(explanations, show=use_pyplot)


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
    use_pyplot: bool = False,
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
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.

    Raises:
        ValueError: Raised in case the explanation is not global or is multitarget, and no target to explain is specified.
        TypeError: Raised in case any of the passed arguments is of the wrong type.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True, plot_type="dependence_plot")

    explanation_values = explanation.explanations.values
    if target is not None:
        explanation_values = explanation_values[..., target]

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

    shap.dependence_plot(
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
    if use_pyplot:
        return None
    if ax is None:
        ax = plt.gca()
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
    target = validate_target(target, explanation, require_single_target=True, plot_type="waterfall")

    explanations_values = explanation.explanations[observation_index]
    if target is not None:
        explanations_values = explanations_values[..., target]

    ax = shap.plots.waterfall(
        explanations_values,
        show=use_pyplot,
    )
    return ax


def wavelengths_bar(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    wavelengths_mapping: dict | None = None,
    transformations_mapping: dict | None = None,
    separate_features: bool = True,
    average_the_same_bands: bool = False,
    cmap: str | Colormap = "tab20",
    use_pyplot: bool = False,
    ax: Axes | None = None,
):
    """
    Creates the aggregated bar plot of SHAP values, grouped by the wavelengths and additionally by the transformation functions
    The function aggregates the values of the features by the wavelengths and creates a bar plot for the aggregated values.

    Args:
        explainer (HyperSHAP):
            A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation):
            A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional):
            Specifies which output to explain if the model has multiple outputs. Defaults to None.
        wavelengths_mapping (dict[str, list[int]] | None, optional):
            Maps band wavelengths (in nanometers) to feature indices in the format:
            `{"band_wavelength": [feature_index, ...], ...}`. Must be provided. Defaults to None.
        transformations_mapping (dict[str, list[int]] | None, optional):
            Maps transformation names to feature indices in the format:
            `{"transformation_name": [feature_index, ...], ...}`. Can be used to more deeply analyze the model's behaviour by inspecting which transformations are the most important. Defaults to None.
        separate_features (bool, optional):
            If True, includes dashed lines in the plot to separate contributions of different features within the same band. Defaults to True.
        average_the_same_bands (bool, optional):
            If True, averages the SHAP values of the features within the same band or the same transformation/band group if the transformation mapping is provided. Defaults to False.
        cmap (str | Colormap, optional):
            The name of the colormap or the matplotlib colormap to use. Defaults to "tab20". Should be a valid matplotlib colormap name or a matplotlib colormap object, that allows for iteration over the colors. Number of colors should be no less than the number of transformations. Defaults to "tab20".
        use_pyplot (bool, optional):
            If True, displays the plot immediately using `matplotlib.pyplot.show`. If the plot is shown, the function does not return the Axes object. Defaults to False.
        ax (matplotlib.axes.Axes, optional):
            If provided, the plot is drawn on the specified axes. If None, a new axes is created. Defaults to None.

    Raises:
        ValueError: If `wavelengths_mapping` is not provided.
        ValueError: If any of the mappings contain invalid data.
        TypeError: If invalid types are passed for `wavelengths_mapping` or `transformations_mapping`.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True, plot_type="wavelengths_bar")

    explanation_values = explanation.explanations.values
    if target is not None:
        explanation_values = explanation_values[..., target]

    if wavelengths_mapping is None:
        raise ValueError("Wavelengths mapping must be provided.")

    parsed_wavelengths_mapping = validate_mapping_dict(wavelengths_mapping, explanation_values)

    # list of wavelengths present in the data
    wavelengths_list = list(parsed_wavelengths_mapping.keys())

    # features aggregated by the original feature index - simple global explanations
    mean_abs_shap_values = np.mean(np.abs(explanation_values), axis=0)

    # whether to assign transformation for each feature separately
    plot_transformations = True
    if transformations_mapping is None:
        plot_transformations = False
        parsed_transformations_mapping = {"None": list(range(explanation_values.shape[1]))}
    else:
        parsed_transformations_mapping = validate_mapping_dict(
            transformations_mapping, explanation_values, wavelengths=False
        )  # type: ignore

    # dictionary that will be filled with the contributions of each wavelength transformed with given transformation
    # each element should be a numpy array of shape (number_of_features, number_of_wavelengths).
    # in this way we can plot the contributions of each transformation separately as a stacked bar plot
    per_transformation_contributions = {}
    transformations_list = list(parsed_transformations_mapping.keys())

    for transformation, transformation_feature_indices in parsed_transformations_mapping.items():
        maximum_intersection = 0
        intersections_dict = {}
        for wavelength, wavelength_feature_indices in parsed_wavelengths_mapping.items():
            intersection_of_features = set(wavelength_feature_indices).intersection(transformation_feature_indices)
            intersections_dict[wavelength] = list(intersection_of_features)
            if len(intersection_of_features) > maximum_intersection:
                maximum_intersection = len(intersection_of_features)

        # Initialize contributions array based on the condition
        rows = 1 if average_the_same_bands else maximum_intersection
        per_transformation_contributions[transformation] = np.zeros((rows, len(wavelengths_list)))

        # Fill the contributions array
        for wavelength_idx, wavelength in enumerate(wavelengths_list):
            if average_the_same_bands:
                # Average SHAP values for features in the same band
                if len(intersections_dict[wavelength]) > 0:
                    per_transformation_contributions[transformation][0][wavelength_idx] = np.mean(
                        [mean_abs_shap_values[feature_idx] for feature_idx in intersections_dict[wavelength]]
                    )
            else:
                # Assign SHAP values directly to the respective features
                for i, feature_idx in enumerate(intersections_dict[wavelength]):
                    per_transformation_contributions[transformation][i][wavelength_idx] = mean_abs_shap_values[
                        feature_idx
                    ]

    # get the colormap
    cmap = cmap if isinstance(cmap, Colormap) else plt.get_cmap(cmap)
    colors = [cmap(i) for i in range(len(transformations_list))]

    # init the axes if not provided
    ax = ax or plt.gca()

    if len(transformations_list) > cmap.N:
        raise ValueError(
            f"Number of transformations ({len(transformations_list)}) is greater than the number of colors in the colormap ({cmap.N}). Please provide a colormap with more colors."
        )

    # the bottoms of the current bars
    bottom = np.zeros(len(wavelengths_list))

    ### plot the bars

    contributions_nested_list = list(per_transformation_contributions.values())

    for transformation_idx in range(len(per_transformation_contributions)):  # iterate over transformations
        transformation = transformations_list[transformation_idx]
        in_group_features_number = len(per_transformation_contributions[transformation])
        for in_group_feature_idx in range(
            in_group_features_number
        ):  # iterate over features in the transformation/wavelength group
            contributions = contributions_nested_list[transformation_idx][in_group_feature_idx]
            bars = ax.bar(
                wavelengths_list,
                contributions,
                bottom=bottom,
                label=transformation if in_group_feature_idx == 0 else "",  # Avoid duplicate legend entries
                color=colors[transformation_idx],
            )
            bottom += contributions

            # draw the values at the top of the bars, excluding the last one
            if (
                not (
                    transformation_idx == len(per_transformation_contributions) - 1
                    and in_group_feature_idx == in_group_features_number - 1
                )
                and separate_features
            ):
                for bar in bars:
                    if bar.get_height() > 0.0001:  # do not draw the values for small bars
                        ax.hlines(
                            bar.get_height() + bar.get_y(),
                            bar.get_x(),
                            bar.get_x() + bar.get_width(),
                            linewidth=0.5,
                            color=ax.get_facecolor(),
                        )

    ax.set_ylim(0, np.max(bottom) * 1.1)
    ax.set_xlabel("Wavelengths (nm)")
    ax.set_xticks(wavelengths_list)
    ax.set_ylabel("mean(|SHAP value|)")

    title = "SHAP explanations"
    if target is not None:
        title += f" for target {target}"
    if average_the_same_bands:
        title += " (averaged across the same bands)"

    ax.set_title(title)
    if plot_transformations:
        ax.legend(title="Transformations", loc="upper right")
    if use_pyplot:
        plt.show()
        return None

    return ax


def bar(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    target: int | None = None,
    observation_index: int | None = None,
    use_pyplot: bool = False,
    ax: Axes | None = None,
    **kwargs,
):
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
    target = validate_target(target, explanation, require_single_target=True, plot_type="bar")

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
    use_pyplot: bool = False,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Create a heatmap plot of a set of SHAP values.
    The function utilizes the `shap.plots.heatmap` function,
    which reference might be found here: https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html
    This plot supports only global explanations.

    Args:
        explainer (HyperSHAP): A HyperSHAP explainer object used to generate the explanation.
        explanation (SHAPExplanation): A SHAPExplanation coming from the meteors package. This object contains the explanations for the model. It can contain both the local and global explanations.
        target (int | None, optional): In case the explained model outputs multiple values, this field specifies which of the outputs we want to explain. Defaults to None.
        In case the passed explanation object is already local and contains data about only a single observation, this value could be also set to None. Defaults to None.
        use_pyplot (bool, optional): If True, uses pyplot to display the image and shows it to the user. If False, returns the figure and axes objects. Defaults to False.
        ax (matplotlib.axes.Axes, optional): If provided, the plot will be displayed on the passed axes.

    Returns:
        Axes | None:
            If use_pyplot is False, returns the axes object.
            If use_pyplot is True, returns None.
    """
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True, plot_type="heatmap")

    explanation_values = explanation.explanations
    if target is not None:
        explanation_values = explanation_values[..., target]
    if explanation.is_local_explanation:
        raise ValueError("The heatmap plot does not support local explanations.")

    ax = shap.plots.heatmap(explanation_values, show=use_pyplot, ax=ax, **kwargs)
    return ax
