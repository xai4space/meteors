import pytest

import torch


from meteors.utils.models import ExplainableModel, SkLearnLasso
from meteors.utils.models.models import SkLearnLinearModel, LinearModel


def test_explainable_model():
    explainable_model = ExplainableModel(problem_type="regression", forward_func=lambda x: x.mean(dim=(1, 2, 3)))
    assert explainable_model is not None

    explainable_model.to("cpu")


def test_problem_type_validation():
    with pytest.raises(ValueError):
        ExplainableModel(problem_type="vacuum_cleaning", forward_func=lambda x: x.mean(dim=(1, 2, 3)))

    with pytest.raises(TypeError):
        ExplainableModel(problem_type=2115, forward_func=lambda x: x.mean(dim=(1, 2, 3)))


def test_linear_model_initialization():
    def dummy_train_fn(model, dataloader, construct_kwargs, **kwargs):
        model._construct_model_params(**construct_kwargs)

    model = LinearModel(train_fn=dummy_train_fn, in_features=10, out_features=2)
    assert model.linear is None
    assert model.norm is None

    X = torch.rand(10, 10)
    y = torch.bernoulli(X)
    data_loader = torch.utils.data.DataLoader(dataset=[X, y], batch_size=10)

    model.fit(data_loader)

    assert model.linear.in_features == 10
    assert model.linear.out_features == 2


def test_linear_model_forward():
    def dummy_train_fn(model, dataloader, construct_kwargs, **kwargs):
        model._construct_model_params(**construct_kwargs)

    model = LinearModel(train_fn=dummy_train_fn, in_features=10, out_features=2)

    X = torch.rand(10, 10)
    y = torch.bernoulli(X)
    data_loader = torch.utils.data.DataLoader(dataset=[X, y], batch_size=10)

    model.fit(data_loader)

    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == (5, 2)


def test_linear_model_get_representation():
    def dummy_train_fn(model, dataloader, construct_kwargs, **kwargs):
        model._construct_model_params(**construct_kwargs)

    model = LinearModel(train_fn=dummy_train_fn, in_features=10, out_features=2)

    X = torch.rand(10, 10)
    y = torch.bernoulli(X)
    data_loader = torch.utils.data.DataLoader(dataset=[X, y], batch_size=10)

    model.fit(data_loader)

    representation = model.get_representation()
    assert representation.shape == (2, 10)


def test_sklearn_linear_model_initialization():
    model = SkLearnLinearModel(sklearn_module="linear_model.LinearRegression")
    assert model.sklearn_module == "linear_model.LinearRegression"


def test_sklearn_lasso():
    sklearn_lasso = SkLearnLasso()

    # return empty values
    assert sklearn_lasso.classes() is None
    assert sklearn_lasso.bias() is None

    sklearn_lasso = SkLearnLasso()

    X = torch.rand(10, 10)
    y = torch.bernoulli(X)

    data_loader = torch.utils.data.DataLoader(dataset=[X, y], batch_size=10)
    sklearn_lasso.fit(train_data=data_loader)

    assert sklearn_lasso.classes() is None
    assert sklearn_lasso.bias() is not None
    assert sklearn_lasso.linear.in_features == 10
    assert sklearn_lasso.linear.out_features == 10
    assert isinstance(sklearn_lasso.get_representation(), torch.Tensor)


def test_linear_model_invalid_norm_type():
    with pytest.raises(ValueError):
        LinearModel(train_fn=lambda x: x, norm_type="invalid_norm")._construct_model_params()


def test_linear_model_missing_features():
    with pytest.raises(ValueError):
        LinearModel(train_fn=lambda x: x)._construct_model_params()


def test_linear_model_bias_value_without_bias():
    with pytest.raises(ValueError):
        LinearModel(train_fn=lambda x: x)._construct_model_params(
            in_features=10, out_features=2, bias=False, bias_value=torch.zeros(2)
        )
