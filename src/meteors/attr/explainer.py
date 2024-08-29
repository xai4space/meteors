from __future__ import annotations


from typing_extensions import Self
from abc import ABC
from loguru import logger
from functools import cached_property


import torch


from meteors.utils.models import ExplainableModel


###################################################################
############################ EXPLAINER ############################
###################################################################


class Explainer(ABC):
    """Explainer class for explaining models.

    Args:
        explainable_model (ExplainableModel): The explainable model to be explained.
    """

    def __init__(self, explainable_model: ExplainableModel):
        if not isinstance(explainable_model, ExplainableModel):
            raise TypeError(f"Expected ExplainableModel, but got {type(explainable_model)}")

        logger.debug(f"Initializing {self.__class__.__name__} explainer on model {explainable_model}")

        self.explainable_model = explainable_model

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
