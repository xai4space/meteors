from __future__ import annotations

import shap
import numpy as np
import pandas as pd
import torch

from .explainer import SHAPExplanation, ensure_data_type_and_reshape
from meteors.models import ExplainableModel

from typing import Literal

AVAILABLE_SHAP_EXPLAINERS = ["exact", "tree", "kernel", "linear"]


class HyperSHAP:
    """
    HyperSHAP is a class that provides [SHAP (SHapley Additive exPlanations)](https://arxiv.org/abs/1705.07874)
    explanations for a given model, based on [shap package](https://github.com/shap/shap?tab=readme-ov-file)

    Attributes:
        explainable_model (ExplainableModel): The model to be explained.
        explainer_type (str): The type of SHAP explainer to use. Options are "exact", "tree", "kernel", "linear".
        _explainer: The SHAP explainer instance.
    """

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
        """
        Generate SHAP explanations for the given data.

        Args:
            data (np.ndarray | pd.DataFrame | torch.Tensor): The input data for which SHAP explanations
            are to be generated. It can be a NumPy array, a pandas DataFrame, or a PyTorch tensor.

        Raises:
            TypeError: If the input data cannot be converted to a numeric NumPy array.
            ValueError: If the explainer has not been initialized or if SHAP explanations could not be generated.

        Returns:
            SHAPExplanation: An object containing the SHAP explanations for the input data.
        """
        data = ensure_data_type_and_reshape(data)

        if not self._explainer:
            raise ValueError("The explainer has not been initialized")

        try:
            shap_values = self._explainer(data)
        except ValueError as e:
            raise ValueError(f"Could not generate SHAP explanations: {e}")

        explanations = SHAPExplanation(data=data, explanations=shap_values, explanation_method=self.explainer_type)

        return explanations
