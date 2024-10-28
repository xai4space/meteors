from __future__ import annotations

from typing_extensions import Annotated, Any
from pydantic.functional_validators import PlainValidator
from pydantic import ConfigDict, Field, ValidationInfo, BaseModel, model_validator

import numpy as np
import pandas as pd
import torch

from loguru import logger

from meteors.exceptions import ShapeMismatchError

import shap

AVAILABLE_SHAP_EXPLAINERS = ["exact", "tree", "kernel", "linear", None]

###################################################################
########################### VALIDATIONS ###########################
###################################################################


def ensure_explainer_type(explainer_type: str | None) -> str | None:
    if explainer_type not in AVAILABLE_SHAP_EXPLAINERS:
        logger.warning(f"Invalid explainer type: {explainer_type}. Defaulting to None.")
        return explainer_type
    return explainer_type


def ensure_data_type(data: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
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


def get_feature_names(values: dict[str, Any]) -> dict[str, Any]:
    """function captures the feature names from the data if they are not provided

    Args:
        values (dict[str, Any]): input pydantic model values

    Raises:
        RuntimeError: An error is raised if the data field is missing in the SHAPExplanation object.

    Returns:
        dict[str, Any]: pydantic model values, possibly with the feature names added
    """

    if "data" not in values:
        raise RuntimeError("Missing `data` field in the SHAPExplanation object.")

    data = values["data"]
    if isinstance(data, pd.DataFrame) and "feature_names" not in values:
        column_names = data.columns
        values["feature_names"] = column_names
    return values


def validate_feature_names(feature_names: list[str] | None, info: ValidationInfo) -> list[str] | None:
    if feature_names is not None and not all(isinstance(name, str) for name in feature_names):
        raise TypeError("Feature names should be a list of strings.")

    if "data" not in info.data:
        raise RuntimeError("ValidationInfo object is broken - pydantic internal error")

    data = info.data["data"]

    shape_data = data.shape
    if feature_names is not None and len(feature_names) != shape_data[1]:
        raise ShapeMismatchError(
            f"Number of feature names does not match the number of features in the data. Expected {shape_data[1]}, but got {len(feature_names)}"
        )

    return feature_names


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
    feature_names: Annotated[
        list[str] | None,
        PlainValidator(validate_feature_names),
        Field(
            description="A list of feature names.",
        ),
    ] = None
    aggregations: Annotated[
        list[list[str]] | list[str] | None,
        Field(
            description="A list of feature aggregations.",
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _get_feature_names = model_validator(mode="before")(get_feature_names)
