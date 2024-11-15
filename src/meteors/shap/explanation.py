from __future__ import annotations

from typing_extensions import Annotated
from pydantic.functional_validators import PlainValidator
from pydantic import ConfigDict, Field, ValidationInfo, BaseModel

import numpy as np
import pandas as pd
import torch

from loguru import logger

from meteors.exceptions import ShapeMismatchError

import shap

AVAILABLE_SHAP_EXPLAINERS = ["exact", "tree", "kernel", "linear"]


###################################################################
########################### VALIDATIONS ###########################
###################################################################


def ensure_explainer_type(explainer_type: str | None) -> str | None:
    explainer_type = explainer_type.lower() if explainer_type is not None else None
    if explainer_type not in AVAILABLE_SHAP_EXPLAINERS and explainer_type is not None:
        logger.warning(f"Invalid explainer type: {explainer_type}. ")
        return explainer_type
    return explainer_type


def ensure_data_type(data: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Expected np.ndarray, pd.DataFrame, or torch.Tensor as data, but got {type(data)}")
    return data


def process_and_validate_explanations(explanations: shap.Explanation, info: ValidationInfo) -> shap.Explanation:
    if not isinstance(explanations, shap.Explanation):
        raise TypeError(f"Expected shap.Explanation as explanations, but got {type(explanations)}")

    if "data" not in info.data:
        raise RuntimeError("ValidationInfo object is broken - pydantic internal error")

    data_shape = info.data["data"].shape
    # check if the shape of the explanations is correct
    if explanations.shape != data_shape:
        # only the last dimension differs
        if explanations.shape[:-1] == data_shape:
            logger.debug("Detected explanation for multiple targets based on the explanation shape validation.")
        else:
            raise ShapeMismatchError(
                f"Shape of the explanations does not match the shape of the input data. Expected {data_shape}, but got {explanations.shape}"
            )

    return explanations


###################################################################
########################### EXPLANATION ###########################
###################################################################


class SHAPExplanation(BaseModel):
    """Represents an object that contains SHAP explanations for a model.

    Args:
        data (np.ndarray): a numpy array containing the input data.
        explanations (np.ndarray): a numpy array containing the SHAP explanations.
        If the model outputs a single value, the shape of the array should be equal to the shape of the input data.
        In case the model outputs multiple values, the last dimension of the array should be equal to the number of outputs and the rest of the dimensions should be equal to the input data.
        explanation_method (str): the method used to generate the explanation.
        feature_names (list[str]): a list of feature names.
        aggregations (list[list[str]] | list[str]): a list of feature aggregations.
    """

    data: Annotated[
        np.ndarray | torch.Tensor | pd.DataFrame,
        PlainValidator(ensure_data_type),
        Field(
            description="A numpy array containing the input data.",
        ),
    ]
    explanations: Annotated[
        shap.Explanation,
        PlainValidator(process_and_validate_explanations),
        Field(
            description="A numpy array containing the SHAP explanations.",
        ),
    ]
    explanation_method: Annotated[
        str | None,
        PlainValidator(ensure_explainer_type),
        Field(
            description="The method used to generate the explanation.",
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def target_dims(self) -> int:
        """number of target dimensions in the explanations. It is equal to the number of outputs of the model.
        Returns:
            int: number of target dimensions in the explanations.
        """
        if self.explanations.shape == self.data.shape:
            return 1
        return self.explanations.values.shape[-1]

    @property
    def is_local_explanation(self) -> bool:
        """check if the explanation is for a single observation - a local explanation.
        Returns:
            bool: True if the explanation is for a single observation, False otherwise.
        """
        if len(self.data.shape) == 1:
            return True
        if len(self.data.shape) == 2 and self.data.shape[0] == 1:
            return True
        return False

    @property
    def feature_names(self) -> list[str] | pd.Index | None:
        """list of feature names.
        Returns:
            list[str] | pd.Index | None: list of feature names.
        """
        if self.explanations is not None:
            return self.explanations.feature_names
        return None
