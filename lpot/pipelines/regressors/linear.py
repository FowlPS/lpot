from typing import Tuple

from sklearn import svm
from sklearn import linear_model

from lpot.pipelines.sklearn_wrappers import sklearn_regressor_wrapper


class LinearFinalElement(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=linear_model.LinearRegression,
            possible_parameters={
                "normalize": [True, False]}
        )

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * (x[1] ** 2) + x[1] ** 3, (x[0], 2)


class SVMRFinalElement(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=svm.SVR,
            possible_parameters={
                "kernel": ["linear", "sigmoid"],
                "gamma": ["auto"],
                "max_iter": [100],
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] + x[0] ** 3, (x[0], 2)
