from typing import Tuple

from sklearn import neighbors
from lpot.pipelines.sklearn_wrappers import sklearn_regressor_wrapper


class KNNRFinalElement(sklearn_regressor_wrapper.SklearnRegressorWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=neighbors.KNeighborsRegressor,
            possible_parameters={
                "n_neighbors": [1, 3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1], (x[0], 2)
