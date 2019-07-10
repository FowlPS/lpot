from typing import Tuple

from lpot.pipelines.sklearn_wrappers import sklearn_transformer_wrapper

from sklearn import decomposition


class PCAWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=decomposition.PCA,
            possible_parameters={
                "n_components": [
                    2, 3, 5, 10, 15, None
                ]
            })

    def fit(self, x, y):
        if self.chosen_parameters["n_components"] and x.shape[1] <= self.chosen_parameters["n_components"]:
            self.fitted = True
            return
        return super().fit(x, y)

    def transform(self, x):
        if self.chosen_parameters["n_components"] and x.shape[1] <= self.chosen_parameters["n_components"]:
            return x
        return super().transform(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return min(x[1] ** 3, x[0] ** 3), (
            x if not self.chosen_parameters["n_components"] else (x[0], self.chosen_parameters["n_components"]))
