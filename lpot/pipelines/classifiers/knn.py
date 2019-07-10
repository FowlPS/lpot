from typing import Tuple

from sklearn import neighbors
from lpot.pipelines.sklearn_wrappers import sklearn_classifier_wrapper

class KNNCFinalElement(sklearn_classifier_wrapper.SklearnClassifierWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=neighbors.KNeighborsClassifier,
            possible_parameters={
                "n_neighbors": [1, 3, 5, 7, 10],
                "p": [1, 2],
                "weights":["uniform", "distance"],
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return (x[0] ** 2) * x[1], (x[0], 2)
