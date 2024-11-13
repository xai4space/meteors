import pytest

import torch


from meteors.models import (
    ExplainableModel,
    SkLearnLasso,
    SkLearnRidge,
    SkLearnLinearRegression,
    SkLearnLogisticRegression,
    SkLearnSGDClassifier,
)

from meteors.models.linear import SkLearnLinearModel, LinearModel


def test_explainable_model():
    explainable_model = ExplainableModel(problem_type="regression", forward_func=lambda x: x.mean(dim=(1, 2, 3)))
    assert explainable_model is not None

    explainable_model = ExplainableModel(problem_type="classification", forward_func=lambda x: x.mean(dim=(1, 2, 3)))
    assert explainable_model is not None

    explainable_model = ExplainableModel(
        problem_type="segmentation",
        forward_func=lambda x: x.mean(dim=(1, 2, 3)),
        postprocessing_output=lambda x: x.mean(dim=(1, 2, 3)),
    )
    assert explainable_model is not None

    with pytest.raises(ValueError):
        explainable_model = ExplainableModel(problem_type="segmentation", forward_func=lambda x: x.mean(dim=(1, 2, 3)))

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
    print(next(iter(data_loader)))

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
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

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
    y = torch.rand(10, 5)

    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    sklearn_lasso.fit(train_data=data_loader)

    assert sklearn_lasso.classes() is None
    assert sklearn_lasso.bias() is not None
    assert sklearn_lasso.linear.in_features == 10
    assert sklearn_lasso.linear.out_features == 5
    assert isinstance(sklearn_lasso.get_representation(), torch.Tensor)


def test_sklearn_ridge():
    sklearn_ridge = SkLearnRidge()

    # return empty values
    assert sklearn_ridge.classes() is None
    assert sklearn_ridge.bias() is None

    sklearn_ridge = SkLearnRidge()

    X = torch.rand(10, 10)
    y = torch.rand(10, 5)

    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    sklearn_ridge.fit(train_data=data_loader)

    assert sklearn_ridge.classes() is None
    assert sklearn_ridge.bias() is not None
    assert sklearn_ridge.linear.in_features == 10
    assert sklearn_ridge.linear.out_features == 5
    assert isinstance(sklearn_ridge.get_representation(), torch.Tensor)


def test_sklearn_linear_regression():
    sklearn_linear_regression = SkLearnLinearRegression()

    # return empty values
    assert sklearn_linear_regression.classes() is None
    assert sklearn_linear_regression.bias() is None

    sklearn_linear_regression = SkLearnLinearRegression()

    X = torch.rand(10, 10)
    y = torch.rand(10, 5)

    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    sklearn_linear_regression.fit(train_data=data_loader)

    assert sklearn_linear_regression.classes() is None
    assert sklearn_linear_regression.bias() is not None
    assert sklearn_linear_regression.linear.in_features == 10
    assert sklearn_linear_regression.linear.out_features == 5
    assert isinstance(sklearn_linear_regression.get_representation(), torch.Tensor)


def test_sklearn_logistic_regression():
    sklearn_logistic_regression = SkLearnLogisticRegression()

    # return empty values
    assert sklearn_logistic_regression.classes() is None
    assert sklearn_logistic_regression.bias() is None

    sklearn_logistic_regression = SkLearnLogisticRegression()

    X = torch.rand(10, 10)
    y = torch.randint(0, 2, (10, 1)).reshape(-1)
    y[0], y[1] = 0, 1

    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    sklearn_logistic_regression.fit(train_data=data_loader)

    assert sklearn_logistic_regression.classes() is not None
    assert sklearn_logistic_regression.bias() is not None
    assert sklearn_logistic_regression.linear.in_features == 10
    assert sklearn_logistic_regression.linear.out_features == 1
    assert isinstance(sklearn_logistic_regression.get_representation(), torch.Tensor)


def test_sklearn_sgd_classifier():
    sklearn_sgd_classifier = SkLearnSGDClassifier()

    # return empty values
    assert sklearn_sgd_classifier.classes() is None
    assert sklearn_sgd_classifier.bias() is None

    sklearn_sgd_classifier = SkLearnSGDClassifier()

    X = torch.rand(10, 10)
    y = torch.randint(0, 2, (10, 1)).reshape(-1)
    y[0], y[1] = 0, 1

    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    sklearn_sgd_classifier.fit(train_data=data_loader)

    assert sklearn_sgd_classifier.classes() is not None
    assert sklearn_sgd_classifier.bias() is not None
    assert sklearn_sgd_classifier.linear.in_features == 10
    assert sklearn_sgd_classifier.linear.out_features == 1
    assert isinstance(sklearn_sgd_classifier.get_representation(), torch.Tensor)


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
