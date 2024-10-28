import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split


import shap

import meteors as mt


def test_shap():
    X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)

    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    explainable_model = mt.utils.models.ExplainableModel(knn.predict_proba, "classification")

    hyper_shap = mt.shap.HyperSHAP(explainable_model, X_train, explainer_type="Kernel")

    explanation = hyper_shap.explain(X_test)

    assert isinstance(explanation, mt.shap.SHAPExplanation)
