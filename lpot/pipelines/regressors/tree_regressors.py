from typing import Tuple

from sklearn import ensemble
from sklearn import tree

from pipelines.sklearn_wrappers import sklearn_regressor_wrapper


class RandomForestRegressor(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=ensemble.RandomForestRegressor,
            possible_parameters={
                "n_estimators": [5, 10, 20, 50, 100],
                "criterion": ["mse", "mae"],
                "max_depth": [
                    # None,
                    3, 5, 10],
                "min_samples_split": [2, 4]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] * self.chosen_parameters["n_estimators"], (x[0], 2)


class DecisionTreeRegressor(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=tree.DecisionTreeRegressor,
            possible_parameters={
                "criterion": ["mse", "mae"],
                "max_depth": [None, 3, 5, 10],
                "min_samples_split": [2, 4]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1], (x[0], 2)


class GradientBoostingRegressor(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=ensemble.GradientBoostingRegressor,
            possible_parameters={
                "n_estimators": [5, 10, 20, 50, 100],
                "criterion": ["mse", "mae"],
                "max_depth": [
                    # None,
                    3, 5, 10],
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] * self.chosen_parameters["n_estimators"], (x[0], 2)
