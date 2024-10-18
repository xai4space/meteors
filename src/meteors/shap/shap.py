from __future__ import annotations

import shap
import numpy as np
import pandas as pd
import torch

from meteors.utils.models import ExplainableModel
from .explanation import SHAPExplanation

from typing import Literal


class HyperSHAP:
    def __init__(
        self, callable: ExplainableModel, masker, explainer_type: Literal["Exact", "Tree", "Kernel"] | None = None
    ) -> None:
        if not isinstance(callable, ExplainableModel):
            raise TypeError(f"Expected ExplainableModel as callable, but got {type(callable)}")

        self.explainable_model = callable

        try:
            if explainer_type is None or explainer_type == "Exact":
                self._explainer = shap.Explainer(self.explainable_model.forward_func, masker)
            elif explainer_type == "Tree":
                self._explainer = shap.TreeExplainer(self.explainable_model.forward_func, masker)
            elif explainer_type == "Kernel":
                self._explainer = shap.KernelExplainer(self.explainable_model.forward_func, masker)
        except ValueError as e:
            raise ValueError(f"Could not initialize the explainer: {e}")

    def explain(self, data: np.ndarray | pd.DataFrame | torch.tensor) -> SHAPExplanation:
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif isinstance(data, torch.Tensor):
            data = data.numpy()

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected np.ndarray as data, but got {type(data)}")

        if not self._explainer:
            raise ValueError("The explainer has not been initialized")

        shap_values = self._explainer(data)

        explanations = SHAPExplanation(data=data, explanations=shap_values, explanation_method="base")

        return explanations
