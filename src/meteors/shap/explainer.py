from __future__ import annotations

from typing_extensions import Annotated, Self
from pydantic import ConfigDict, Field, BaseModel, BeforeValidator, model_validator

import numpy as np
import pandas as pd
import torch

from loguru import logger

from meteors.exceptions import ShapeMismatchError

import shap as base_shap


###################################################################
########################### VALIDATIONS ###########################
###################################################################


def ensure_data_type_and_reshape(data: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray:
    """
    Ensures that the input data is converted to a NumPy ndarray.
    If the input is one dimensional, it reshapes it to a two-dimensional array.

    Args:
        data (np.ndarray | torch.Tensor | pd.DataFrame): The input data which can be a NumPy ndarray,
            a PyTorch tensor, or a Pandas DataFrame/Series. The data type should be numeric

    Returns:
        np.ndarray: The input data converted to a NumPy ndarray.

    """
    converted_data = None
    if isinstance(data, np.ndarray):
        converted_data = data
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        converted_data = data.to_numpy()
    elif isinstance(data, torch.Tensor):
        converted_data = data.numpy()
    else:
        try:
            converted_data = np.array(data)
        except Exception as e:
            raise TypeError(
                f"Expected NumPy array | Torch Tensor | Pandas DataFrame as data, but got {type(data)} and failed to convert to NumPy array"
            ) from e
    if not np.issubdtype(converted_data.dtype, np.number):
        raise TypeError(f"Expected numeric data, but got {converted_data.dtype}")

    # reshaping the data if it is one-dimensional
    converted_data = np.reshape(converted_data, (1, -1)) if len(converted_data.shape) == 1 else converted_data
    return converted_data


def add_dimension_to_local_explanation(explanation: base_shap.Explanation) -> base_shap.Explanation:
    """
    Adds a new axis to the SHAP explanation for a local explanation
    """
    explanation.values = np.expand_dims(explanation.values, axis=0)
    explanation.base_values = np.expand_dims(explanation.base_values, axis=0)
    explanation.data = np.expand_dims(explanation.data, axis=0)
    return explanation


###################################################################
########################### EXPLANATION ###########################
###################################################################


class SHAPExplanation(BaseModel):
    """Represents an object that contains SHAP explanations for a model.

    Args:
        data (np.ndarray): a numpy array containing the input data.
        explanations (np.ndarray): a numpy array containing the SHAP explanations.
        If the model outputs a single value, the shape of the array should be equal to the shape of the input data.
        In case the model outputs multiple values, the last dimension of the array should be equal to the number of
        outputs and the rest of the dimensions should be equal to the input data.
        explanation_method (str): the method used to generate the explanation.
        feature_names (list[str]): a list of feature names.
        aggregations (list[list[str]] | list[str]): a list of feature aggregations.
    """

    data: Annotated[
        np.ndarray,
        BeforeValidator(ensure_data_type_and_reshape),
        Field(
            description="A numpy array containing the input data.",
        ),
    ]
    explanations: Annotated[
        base_shap.Explanation,
        Field(
            description="A numpy array containing the SHAP explanations.",
        ),
    ]
    explanation_method: Annotated[
        str | None,
        Field(
            description="The method used to generate the explanation.",
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _validate_shapes(self):
        """
        Validates that the shape of the explanations matches the shape of the input data.

        This method checks if the shape of `self.explanations` matches the shape of `self.data`.
        If the shapes do not match, it checks if the explanation is multitarget - the last dimension of the
        explanation will not match the data shape. In case the explanation is local - the explanation is one
        axis shorter, it unsquezees it - adds a new axis in the beginning of the explanation.
        Otherwise, it raises a `ShapeMismatchError`
        Raises:
            ShapeMismatchError: If the shape of the explanations does not match the shape of the input data,
                the mismatch is not only in the last dimension and cannot unsqueeze the data,
                in case the explanation is local.
        """
        data_shape = self.data.shape
        explanation_shape = self.explanations.shape
        if explanation_shape == data_shape:
            return  # no need to validate

        if len(explanation_shape) == len(data_shape) + 1:
            # the explanation has one more dimension than the data - the multitarget explanation, no local explanation
            if explanation_shape[:-1] == data_shape:
                logger.debug("Detected explanation for multiple targets based on the explanation shape validation.")
                return
            else:
                raise ShapeMismatchError(
                    "Shape of the explanations does not match the shape of the input data. "
                    f"Expected {data_shape}, but got {self.explanations.shape}"
                )
        elif len(explanation_shape) == len(data_shape):
            if explanation_shape[-2] == data_shape[-1] and data_shape[0] == 1:
                logger.debug("Detected explanation for multiple targets based on the explanation shape validation.")
                logger.debug("This is a local explanation for multiple targets. Adding a new axis to the explanation.")
                self.explanations = add_dimension_to_local_explanation(self.explanations)
                if self.explanations.shape[:-1] == data_shape:
                    return
        elif len(explanation_shape) == len(data_shape) - 1:
            logger.debug("Detected local explanation for a single target. Removing the last axis from the explanation.")
            self.explanations = add_dimension_to_local_explanation(self.explanations)
            if self.explanations.shape == data_shape:
                return

        raise ShapeMismatchError(
            "Shape of the explanations does not match the shape of the input data. "
            f"Expected {data_shape}, but got {explanation_shape}"
        )

    @model_validator(mode="after")
    def validate_explanations(self) -> Self:
        """
        Validates the the explanations.

        Returns:
            Self: The instance of the class for method chaining.
        """
        self._validate_shapes()
        return self

    @property
    def num_target_outputs(self) -> int:
        """
        Number of target dimensions in the explanations. It is equal to the number of outputs of the model. For instance,
        in scope of the HYPERVIEW Challenge, the EAGLEEYES model has 4 outputs - it predicts 4 different soil parameters.
        For explanations of this model, the `num_target_outputs` property will equal to 4.
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
        if len(self.data.shape) == 2 and self.data.shape[0] == 1:
            return True
        return False

    @property
    def feature_names(self) -> list[str] | pd.Index | None:
        """List of feature names.

        Returns:
            list[str] | pd.Index | None: list of feature names.
        """
        return self.explanations.feature_names
