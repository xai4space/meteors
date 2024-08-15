import pytest

import torch


from meteors.utils.models import ExplainableModel, SkLearnLasso


def test_explainable_model():
    explainable_model = ExplainableModel(problem_type="regression", forward_func=lambda x: x.mean(dim=(1, 2, 3)))
    assert explainable_model is not None

    explainable_model.to("cpu")


def test_problem_type_validation():
    with pytest.raises(ValueError):
        ExplainableModel(problem_type="vacuum_cleaning", forward_func=lambda x: x.mean(dim=(1, 2, 3)))

    with pytest.raises(TypeError):
        ExplainableModel(problem_type=2115, forward_func=lambda x: x.mean(dim=(1, 2, 3)))


def test_sklearn_lasso():
    sklearn_lasso = SkLearnLasso()

    # return empty values
    sklearn_lasso.bias()
    sklearn_lasso.classes()

    sklearn_lasso = SkLearnLasso()

    X = torch.rand(10, 10)
    y = torch.bernoulli(X)

    data_loader = torch.utils.data.DataLoader(dataset=[X, y], batch_size=10)
    sklearn_lasso.fit(train_data=data_loader)

    sklearn_lasso.bias()
    sklearn_lasso.get_representation()
