from __future__ import annotations

import shap
import numpy as np
import pandas as pd
import torch

from .explanation import SHAPExplanation

from meteors.models import ExplainableModel

from typing import Literal

AVAILABLE_SHAP_EXPLAINERS = ["exact", "tree", "kernel", "linear"]


class HyperSHAP:
    def __init__(
        self,
        callable: ExplainableModel,
        masker,
        explainer_type: Literal["exact", "tree", "kernel", "linear"] = "exact",
        **kwargs,
    ) -> None:
        if not isinstance(callable, ExplainableModel):
            raise TypeError(f"Expected ExplainableModel as callable, but got {type(callable)}")

        self.explainable_model = callable

        self.explainer_type = explainer_type.lower() if explainer_type is not None else None  # type: ignore

        if self.explainer_type not in AVAILABLE_SHAP_EXPLAINERS:
            raise ValueError(
                f"Invalid explainer type: {explainer_type}. Available options: {AVAILABLE_SHAP_EXPLAINERS}"
            )

        try:
            if self.explainer_type == "exact":
                self._explainer = shap.Explainer(self.explainable_model.forward_func, masker, **kwargs)
            elif self.explainer_type == "tree":
                self._explainer = shap.TreeExplainer(self.explainable_model.forward_func, masker, **kwargs)
            elif self.explainer_type == "kernel":
                self._explainer = shap.KernelExplainer(self.explainable_model.forward_func, masker, **kwargs)
            elif self.explainer_type == "linear":
                self._explainer = shap.LinearExplainer(self.explainable_model.forward_func, masker, **kwargs)
        except ValueError as e:
            raise ValueError(f"Could not initialize the explainer: {e}")

    def explain(self, data: np.ndarray | pd.DataFrame | torch.tensor) -> SHAPExplanation:
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        elif isinstance(data, torch.Tensor):
            data = data.numpy()

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, pd.DataFrame or torch.Tensor as data, but got {type(data)}")

        try:
            data.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse the data to the numeric data type: {e}")

        if not self._explainer:
            raise ValueError("The explainer has not been initialized")

        try:
            shap_values = self._explainer(data)
        except ValueError as e:
            raise ValueError(f"Could not generate SHAP explanations: {e}")

        explanations = SHAPExplanation(data=data, explanations=shap_values, explanation_method=self.explainer_type)

        return explanations
