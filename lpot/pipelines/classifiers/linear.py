from typing import Tuple

from sklearn import svm
from sklearn import linear_model

from lpot.pipelines.sklearn_wrappers import sklearn_classifier_wrapper


class LogisticFinalElement(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
                sklearn_class=linear_model.LogisticRegression,
            possible_parameters={
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
                "multi_class": ["auto", "ovr"],
                "max_iter": [100]}
        )

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * (x[1] ** 2) + x[1] ** 3, (x[0], 2)


class SVMCFinalElement(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=svm.SVC,
            possible_parameters={
                "kernel": ["linear", "sigmoid", "rbf"],
                "probability": [True],
                "gamma": ["auto"],
                "max_iter": [100],
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] + x[0] ** 3, (x[0], 2)
