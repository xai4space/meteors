from __future__ import annotations

from typing import Callable
from abc import ABC, abstractmethod
import warnings

import torch


class ExplainableModel:
    """A class representing an explainable model.

    Args:
        forward_func (Callable): The forward function of the model.
        problem_type (str): The type of the problem the model is designed to solve.

    Attributes:
        forward_func (Callable): The forward function of the model.
        problem_type (str): The type of the problem the model is designed to solve.

    Raises:
        TypeError: If the problem_type is not a string.
        ValueError: If the problem_type is not one of 'classification', 'regression', or 'segmentation'.

    Methods:
        __call__(self, x: torch.Tensor) -> torch.Tensor: Calls the forward function of the model.
        to(self, device: torch.device | str) -> ExplainableModel: Moves the model to the specified device.
    """

    VALID_PROBLEM_TYPES = {"classification", "regression", "segmentation"}

    def __init__(self, forward_func: Callable[[torch.Tensor], torch.Tensor], problem_type: str) -> None:
        self.forward_func = forward_func
        self.problem_type = self.validate_problem_type(problem_type)

    @staticmethod
    def validate_problem_type(value: str) -> str:
        """Validates the problem type.

        Args:
            value (str): The problem type.

        Returns:
            str: The validated problem type.

        Raises:
            TypeError: If the problem type is not a string.
            ValueError: If the problem type is not one of 'classification', 'regression', or 'segmentation'.
        """
        if not isinstance(value, str):
            raise TypeError("Problem type must be a string")

        value = value.lower()

        if value not in ExplainableModel.VALID_PROBLEM_TYPES:
            raise ValueError(f"Invalid problem type. Expected one of {ExplainableModel.VALID_PROBLEM_TYPES}")

        return value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.forward_func(x)

    def to(self, device: torch.device | str) -> ExplainableModel:
        """Moves the model to the specified device.

        Args:
            device (torch.device | str): The device to move the model to.

        Returns:
            ExplainableModel: The model itself.
        """
        if hasattr(self.forward_func, "to") and callable(getattr(self.forward_func, "to")):
            self.forward_func.to(device)
        else:
            warnings.warn("Model does not have a `to` method, it might not be a PyTorch model.")
        return self


class InterpretableModel(ABC):
    """Abstract base class for an interpretable model.

    This class defines the interface for an interpretable model, which is a model that provides
    interpretability in addition to its predictive capabilities.

    Attributes:
        None

    Methods:
        fit(train_data: torch.utils.data.DataLoader, **kwargs) -> None:
            Fits the model to the training data.

        get_representation() -> torch.Tensor:
            Returns the learned representation of the model.

        __call__(x: torch.Tensor) -> torch.Tensor:
            Makes predictions for the input tensor x.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the model to the training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_representation(self) -> torch.Tensor:
        """Returns the representation of the object as a torch.Tensor.

        Returns:
            torch.Tensor: The representation of the object.
        """
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        pass
