from __future__ import annotations


from typing_extensions import Self, Callable, Type
from abc import ABC
from loguru import logger
from functools import cached_property

from captum.attr import Attribution


import torch

from meteors.utils.models import ExplainableModel


def validate_attribution_method_initialization(attribution_method: Explainer):
    if attribution_method is None:
        raise ValueError("Attribution method is not initialized")
    if not isinstance(attribution_method.explainable_model, ExplainableModel):
        raise AttributeError(
            f"The attribution method {attribution_method.__class__.__name__} is not initialized properly"
        )
    if attribution_method.explainable_model is None:
        raise ValueError(f"The attribution method {attribution_method.__class__.__name__} is not properly initialized")


###################################################################
############################ EXPLAINER ############################
###################################################################


class Explainer(ABC):
    """Explainer class for explaining models.

    Args:
        callable (ExplainableModel | Explainer): The explainable model to be explained, or another explainer - used for chaining explainers such as NoiseTunnel.
    """

    def __init__(self, callable: ExplainableModel | Explainer):
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

    attribute: Callable
    """Default method to attribute the input to the model.

    Args:
        image (Image): The input image to be explained.
        target (int | None, optional): The target class index to be explained. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        ImageAttributes: The image attributes.
    """

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
        return "".join(
            [char if char.islower() or idx == 0 else " " + char for idx, char in enumerate("Hyper" + cls.__name__)]
        )

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
