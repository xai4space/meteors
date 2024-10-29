from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from meteors.shap import HyperSHAP, SHAPExplanation

import shap


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


def dependence_plot(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)

    raise NotImplementedError("The function is not implemented yet.")


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
