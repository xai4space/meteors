from __future__ import annotations

from meteors.shap import HyperSHAP, SHAPExplanation

import shap


def validate_observation_index(
    observation_index: int | None, explanation: SHAPExplanation, require_single_observation: bool = False
):
    if not isinstance(observation_index, int):
        if observation_index is not None:
            raise TypeError(f"Expected int as observation index, but got {type(observation_index)}")
    if explanation.explanation_for_single_observation and (observation_index != 0 and observation_index is not None):
        raise ValueError("Observation index must be 0 or None for single observation explanations")
    if observation_index is not None and (
        observation_index < 0 or observation_index >= explanation.explanations.values.shape[0]
    ):
        raise ValueError(f"Observation index out of bounds: {observation_index}")

    if require_single_observation:
        if observation_index is None and not explanation.explanation_for_single_observation:
            raise ValueError("Observation index must be specified when the explanation contains multiple observations")
        if observation_index is None:
            observation_index = 0

    return observation_index


def validate_target(target: int | None, explanation: SHAPExplanation, require_single_target: bool = False):
    if target is not None and not isinstance(target, int):
        raise TypeError(f"Expected int or None as target, but got {type(target)}")

    if target is not None and (target < 0 or target >= explanation.target_dims):
        raise ValueError(f"Target index out of bounds: {target}")

    if require_single_target:
        if target is None and explanation.target_dims > 1:
            raise ValueError(
                "Target index must be specified when the explanation contains multiple targets.\nThis plot only supports single target explanations."
            )
        if target is None:
            target = 0

    return target


def validate_explanations_and_explainer_type(explainer: HyperSHAP, explanation: SHAPExplanation):
    if not isinstance(explainer, HyperSHAP):
        raise TypeError(f"Expected HyperSHAP as explainer, but got {type(explainer)}")
    if not isinstance(explanation, SHAPExplanation):
        raise TypeError(f"Expected HyperSHAP as explanation, but got {type(explanation)}")


def force(
    explainer: HyperSHAP,
    explanation: SHAPExplanation,
    observation_index: int | None = None,
    target: int | None = None,
    **kwargs,
):
    validate_explanations_and_explainer_type(explainer, explanation)
    target = validate_target(target, explanation, require_single_target=True)
    observation_index = validate_observation_index(observation_index, explanation, require_single_observation=True)

    explanations_values = explanation.explanations.values[observation_index]
    if target is not None:
        explanations_values = explanations_values[..., target]

    fig = shap.plots.force(
        explainer._explainer.expected_value[observation_index],
        explanations_values,
        matplotlib=True,
        show=False,
        **kwargs,
    )
    return fig


def beeswarm(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None, **kwargs):
    if not isinstance(explainer, HyperSHAP):
        raise TypeError(f"Expected HyperSHAP as explainer, but got {type(explainer)}")
    if not isinstance(explanation, SHAPExplanation):
        raise TypeError(f"Expected HyperSHAP as explanation, but got {type(explanation)}")

    fig = shap.plots.beeswarm(explanation.explanations[..., target], show=False, **kwargs)
    return fig


def dependence_plot(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    pass


def waterfall(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    pass


def bar(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    pass


def heatmap(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    pass


def partial_dependence_plot(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    pass
