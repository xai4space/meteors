from __future__ import annotations

from typing_extensions import Self, Type, Callable
from abc import ABC
from loguru import logger
from functools import cached_property

from captum.attr import Attribution


import torch

from meteors.models import ExplainableModel
from meteors import HSI
from meteors.attr import HSIAttributes
from meteors.exceptions import ShapeMismatchError


def validate_and_transform_baseline(baseline: int | float | torch.Tensor | None, hsi: HSI) -> torch.Tensor:
    """Function validates the baseline and transforms it to the same device as the hsi tensor.

    Args:
        baseline (int | float | torch.Tensor): a baseline value that is used for the attribution method with the shape
            of the hsi image tensor. If scalar passed, it will be broadcasted to the shape of the hsi tensor. If tensor
            passed, it should have the same shape as the hsi tensor. If None passed, the baseline will be set to 0 and
            broadcasted to the shape of the hsi tensor. Defaults to None.
        hsi (HSI): a HSI object for which the baseline is being validated

    Raises:
        ShapeMismatchError: If the shape of the baseline tensor does not match the shape of the hsi tensor.

    Returns:
        torch.Tensor: a baseline tensor with the same shape as the hsi tensor that is on the same device
            as the hsi tensor.
    """

    if baseline is None:
        baseline = 0
    if isinstance(baseline, (int, float)):
        baseline = torch.zeros_like(hsi.image) + baseline
    elif isinstance(baseline, torch.Tensor):
        if baseline.shape != hsi.image.shape:
            raise ShapeMismatchError(
                f"Passed baseline and HSI have incorrect shapes: {baseline.shape} and {hsi.image.shape}"
            )
    if not isinstance(baseline, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor | int | float as baseline, but got {type(baseline)}")

    baseline = baseline.to(hsi.image.device)  # cast the baseline to the same device as the hsi tensor
    return baseline


###################################################################
############################ EXPLAINER ############################
###################################################################


class Explainer(ABC):
    """Explainer class for explaining models.

    Attributes:
        explainable_model (ExplainableModel | Explainer): The explainable model to be explained.
        forward_func (Callable): The forward function of the explainable model.
        chained_explainer (Explainer | None): The chained explainer. Defaults to None.
    """

    attribute: Callable[
        [list[HSI] | HSI, list[int] | int | None],
        HSIAttributes | list[HSIAttributes],
    ]

    def __init__(self, callable: ExplainableModel | Explainer) -> None:
        if not isinstance(callable, ExplainableModel) and not isinstance(callable, Explainer):
            raise TypeError(f"Expected ExplainableModel or Explainer as callable, but got {type(callable)}")
        self.chained_explainer = None
        self._attribution_method: Attribution | None = (
            None  # the inner attribution method coming from the captum library
        )

        self._is_final_explainer = (
            True  # flag to check whether there is a situation: Explainer(Explainer(Explainer(ExplainableModel)))
        )

        if isinstance(callable, Explainer):
            self._is_final_explainer = False
            if isinstance(callable.explainable_model, Explainer) or not callable._is_final_explainer:
                raise ValueError("Cannot chain Explainer with another Explainer. The maximum depth of chaining is 1")
            self.chained_explainer = callable
            self.explainable_model: ExplainableModel = callable.explainable_model
            logger.debug(
                f"Initializing {self.__class__.__name__} explainer on model {callable.explainable_model} chained with {callable.__class__.__name__}"
            )
        else:
            self.explainable_model = callable
            logger.debug(f"Initializing {self.__class__.__name__} explainer on model {callable}")

        self.forward_func = self.explainable_model.forward_func

    def has_convergence_delta(self) -> bool:
        """Check if the explainer has a convergence delta.

        Returns:
            bool: True if the explainer has a convergence delta, False otherwise.
        """
        if self._attribution_method is not None and hasattr(self._attribution_method, "has_convergence_delta"):
            return self._attribution_method.has_convergence_delta()
        return False

    @classmethod
    def get_name(cls: Type["Explainer"]) -> str:
        """Get the name of the explainer.

        Returns:
            str: The name of the explainer.
        """
        return "".join([char if char.islower() or idx == 0 else " " + char for idx, char in enumerate(cls.__name__)])

    def compute_convergence_delta(self):
        """Compute the convergence delta of the explainer.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Convergence delta computation not implemented in the explainer base class")

    @property
    def multiplies_by_inputs(self) -> bool:
        if self._attribution_method is not None and hasattr(self._attribution_method, "multiplies_by_inputs"):
            return self._attribution_method.multiplies_by_inputs
        return False

    @cached_property
    def device(self) -> torch.device:
        """Get the device on which the explainable model is located.

        Returns:
            torch.device: The device on which the explainable model is located.
        """
        try:
            device = next(self.explainable_model.forward_func.parameters()).device  # type: ignore
        except Exception:
            logger.debug("Could not extract device from the explainable model, setting device to cpu")
            logger.warning("Not a torch model, setting device to cpu")
            device = torch.device("cpu")
        return device

    def to(self, device: str | torch.device) -> Self:
        """Move the explainable model to the specified device.

        Args:
            device (str or torch.device): The device to move the explainable model to.

        Returns:
            Self: The updated Explainer instance.
        """
        self.explainable_model = self.explainable_model.to(device)
        return self
