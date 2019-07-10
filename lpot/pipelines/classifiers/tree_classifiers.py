from typing import Tuple

from sklearn import ensemble
from sklearn import tree

from lpot.pipelines.sklearn_wrappers import sklearn_classifier_wrapper


class RandomForestClassifier(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=ensemble.RandomForestClassifier,
            possible_parameters={
                "n_estimators": [5, 10, 20, 50, 100],
                "criterion": ["gini", "entropy"],
                "max_depth": [
                    # None,
                    3, 5, 10],
                "min_samples_split": [2, 4]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] * self.chosen_parameters["n_estimators"], (x[0], 2)


class DecisionTreeClassifier(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=tree.DecisionTreeClassifier,
            possible_parameters={
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 3, 5, 10],
                "min_samples_split": [2, 4]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1], (x[0], 2)


class GradientBoostingClassifier(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=ensemble.GradientBoostingClassifier,
            possible_parameters={
                "n_estimators": [5, 10, 20, 50, 100],
                "max_depth": [
                    # None,
                    3, 5, 10],
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1] * self.chosen_parameters["n_estimators"], (x[0], 2)
