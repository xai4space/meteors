from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from captum._utils.models.linear_model.train import sklearn_train_linear_model

from .abstract import InterpretableModel

############################################################################################################
### Code below is copied from captum library. It has slight modifications made jointly by me or Vladimir ###
############################################################################################################


class LinearModel(nn.Module, InterpretableModel):
    SUPPORTED_NORMS: list[str | None] = [None, "batch_norm", "layer_norm"]

    def __init__(self, train_fn: Callable, **kwargs) -> None:
        r"""Constructs a linear model with a training function and additional construction arguments that will be sent to
        `self._construct_model_params` after a `self.fit` is called. Please note that this assumes the `self.train_fn`
        will call `self._construct_model_params`.

        Please note that this is an experimental feature.

        Args:
            train_fn (Callable):
                The function to train with. See
                `captum._models.linear_model.train.sgd_train_linear_model`
                and
                `captum._models.linear_model.train.sklearn_train_linear_model`
                for examples.
            kwargs:
                Any additional keyword arguments to send to
                `self._construct_model_params` once a `self.fit` is called.
        """
        super().__init__()

        self.norm: nn.Module | None = None
        self.linear: nn.Linear | None = None
        self.train_fn = train_fn
        self.construct_kwargs = kwargs

    def _construct_model_params(
        self,
        in_features: int | None = None,
        out_features: int | None = None,
        norm_type: str | None = None,
        affine_norm: bool = False,
        bias: bool = True,
        weight_values: torch.Tensor | None = None,
        bias_value: torch.Tensor | None = None,
        classes: torch.Tensor | None = None,
    ) -> None:
        r"""Lazily initializes a linear model. This will be called for you in a train method.

        Args:
            in_features (int):
                The number of input features.
            output_features (int):
                The number of output features.
            norm_type (str, optional):
                The type of normalization that can occur. Please assign this
                to one of `PyTorchLinearModel.SUPPORTED_NORMS`.
            affine_norm (bool):
                Whether or not to learn an affine transformation of the
                normalization parameters used.
            bias (bool):
                Whether to add a bias term. Not needed if normalized input.
            weight_values (Tensor, optional):
                The values to initialize the linear model with. This must be a
                1D or 2D tensor, and of the form `(num_outputs, num_features)` or
                `(num_features,)`. Additionally, if this is provided you need not
                to provide `in_features` or `out_features`.
            bias_value (Tensor, optional):
                The bias value to initialize the model with.
            classes (Tensor, optional):
                The list of prediction classes supported by the model in case it
                performs classification. In case of regression, it is set to None.
                Default: None
        """
        if norm_type not in LinearModel.SUPPORTED_NORMS:
            raise ValueError(f"{norm_type} not supported. Please use {LinearModel.SUPPORTED_NORMS}")

        if weight_values is not None:
            in_features = weight_values.shape[-1]
            out_features = 1 if len(weight_values.shape) == 1 else weight_values.shape[0]

        if in_features is None or out_features is None:
            raise ValueError("Please provide `in_features` and `out_features` or `weight_values`")

        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features, eps=1e-8, affine=affine_norm)
        elif norm_type == "layer_norm":
            self.norm = nn.LayerNorm(in_features, eps=1e-8, elementwise_affine=affine_norm)
        else:
            self.norm = None

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if weight_values is not None:
            self.linear.weight.data = weight_values

        if bias_value is not None:
            if not bias:
                raise ValueError("`bias_value` is not None and bias is False")

            self.linear.bias.data = bias_value

        if classes is not None:
            self.linear.classes = classes

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        r"""Calls `self.train_fn`."""
        return self.train_fn(
            self,
            dataloader=train_data,
            construct_kwargs=self.construct_kwargs,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        assert self.linear is not None
        if self.norm:
            x = self.norm(x)
        return self.linear(x)

    def get_representation(self) -> torch.Tensor:
        r"""Returns a tensor which describes the hyper-plane input space.

        This does
        not include the bias. For bias/intercept, please use `self.bias`.
        """
        assert self.linear is not None
        return self.linear.weight.detach()

    def bias(self) -> torch.Tensor | None:
        r"""Returns the bias of the linear model.

        If no bias is present, it returns None.
        """
        if self.linear is None or not hasattr(self.linear, "bias") or self.linear.bias is None:
            return None
        return self.linear.bias.detach()

    def classes(self) -> torch.Tensor | None:
        r"""Returns the list of prediction classes supported by the model in case it performs classification.

        In case of regression, it returns None.
        """
        if self.linear is None or not hasattr(self.linear, "classes") or self.linear.classes is None:
            return None
        return self.linear.classes.detach()


class SkLearnLinearModel(LinearModel):
    def __init__(self, sklearn_module: str, **kwargs) -> None:
        r"""Factory class to construct a `LinearModel` with sklearn training method.

        Please note that this assumes:

        0. You have sklearn and numpy installed
        1. The dataset can fit into memory

        SkLearn support does introduce some slight overhead as we convert the
        tensors to numpy and then convert the resulting trained model to a
        `LinearModel` object. However, this conversion should be negligible.

        Args:
            sklearn_module
                The module under sklearn to construct and use for training, e.g.
                use "svm.LinearSVC" for an SVM or "linear_model.Lasso" for Lasso.

                There are factory classes defined for you for common use cases,
                such as `SkLearnLasso`.
            kwargs
                The kwargs to pass to the construction of the sklearn model
        """
        super().__init__(train_fn=sklearn_train_linear_model, **kwargs)

        self.sklearn_module = sklearn_module

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        r"""Fits the model to the given training data using the sklearn module.

        Args:
            train_data
                The training data to use for fitting the model.
            kwargs
                Additional arguments to pass to the `.fit` method of the sklearn module.
        """
        return super().fit(train_data=train_data, sklearn_trainer=self.sklearn_module, **kwargs)


class SkLearnLasso(SkLearnLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""Factory class.

        Trains a `LinearModel` model with
        `sklearn.linear_model.Lasso`. You will need sklearn version >= 0.23 to
        support sample weights.
        """
        super().__init__(sklearn_module="linear_model.Lasso", **kwargs)

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the `SkLearnLasso` model to the provided training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method.

        Returns:
            None
        """
        return super().fit(train_data=train_data, **kwargs)


class SkLearnRidge(SkLearnLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""Factory class.

        Trains a `LinearModel` model with
        `sklearn.linear_model.Ridge`. You will need sklearn version >= 0.23 to
        support sample weights.
        """
        super().__init__(sklearn_module="linear_model.Ridge", **kwargs)

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the `SkLearnRidge` model to the provided training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method.

        Returns:
            None
        """
        return super().fit(train_data=train_data, **kwargs)


class SkLearnLinearRegression(SkLearnLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""Factory class.

        Trains a `LinearModel` model with
        `sklearn.linear_model.LinearRegression`. You will need sklearn version >= 0.23 to
        support sample weights.
        """
        super().__init__(sklearn_module="linear_model.LinearRegression", **kwargs)

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the `SkLearnLinearRegression` model to the provided training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method.

        Returns:
            None
        """
        return super().fit(train_data=train_data, **kwargs)


class SkLearnLogisticRegression(SkLearnLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""Factory class.

        Trains a `LinearModel` model with
        `sklearn.linear_model.LogisticRegression`. You will need sklearn version >= 0.23 to
        support sample weights.
        """
        super().__init__(sklearn_module="linear_model.LogisticRegression", **kwargs)

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the `SkLearnLogisticRegression` model to the provided training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method.

        Returns:
            None
        """
        return super().fit(train_data=train_data, **kwargs)


class SkLearnSGDClassifier(SkLearnLinearModel):
    def __init__(self, **kwargs) -> None:
        r"""Factory class.

        Trains a `LinearModel` model with
        `sklearn.linear_model.SGDClassifier`. You will need sklearn version >= 0.23 to
        support sample weights.
        """
        super().__init__(sklearn_module="linear_model.SGDClassifier", **kwargs)

    def fit(self, train_data: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fits the `SkLearnSGDClassifier` model to the provided training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method.

        Returns:
            None
        """
        return super().fit(train_data=train_data, **kwargs)
