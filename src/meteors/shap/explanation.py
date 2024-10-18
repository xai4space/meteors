from __future__ import annotations

from pydantic import BaseModel

from typing_extensions import Annotated

from pydantic import ConfigDict, Field

import numpy as np


import shap


###################################################################
########################### EXPLANATION ###########################
###################################################################


class SHAPExplanation(BaseModel):
    """Represents an object that contains SHAP explanations for a model.

    Args:
        data (np.ndarray): a numpy array containing the input data.
        explanations (np.ndarray): a numpy array containing the SHAP explanations.
        explanation_method (str): the method used to generate the explanation.
        feature_names (list[str]): a list of feature names.
        aggregations (list[list[str]] | list[str]): a list of feature aggregations.
    """

    data: Annotated[
        np.ndarray,
        Field(
            description="A numpy array containing the input data.",
        ),
    ]
    explanations: Annotated[
        shap.Explanation,
        Field(
            description="A numpy array containing the SHAP explanations.",
        ),
    ]
    explanation_method: Annotated[
        str,
        Field(
            description="The method used to generate the explanation.",
        ),
    ]
    feature_names: Annotated[
        list[str] | None,
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
