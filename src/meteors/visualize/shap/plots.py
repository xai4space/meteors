from __future__ import annotations

from meteors.shap import HyperSHAP, SHAPExplanation

import shap


def force(explainer: HyperSHAP, explanation: SHAPExplanation, observation_index: int, target: int | None = None):
    if not isinstance(explainer, HyperSHAP):
        raise TypeError(f"Expected HyperSHAP as explainer, but got {type(explainer)}")
    if not isinstance(explanation, SHAPExplanation):
        raise TypeError(f"Expected HyperSHAP as explanation, but got {type(explanation)}")
    if not isinstance(observation_index, int):
        raise TypeError(f"Expected int as observation index, but got {type(observation_index)}")
    if observation_index < 0 or observation_index >= explanation.explanations.values.shape[0]:
        raise ValueError(f"Observation index out of bounds: {observation_index}")

    fig = shap.plots.force(
        explainer._explainer.expected_value[observation_index],
        explanation.explanations.values[observation_index, ..., target],
        matplotlib=True,
        show=False,
    )
    return fig


def beeswarm(explainer: HyperSHAP, explanation: SHAPExplanation, target: int | None = None):
    if not isinstance(explainer, HyperSHAP):
        raise TypeError(f"Expected HyperSHAP as explainer, but got {type(explainer)}")
    if not isinstance(explanation, SHAPExplanation):
        raise TypeError(f"Expected HyperSHAP as explanation, but got {type(explanation)}")

    fig = shap.plots.beeswarm(explanation.explanations[..., target], show=False)
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
