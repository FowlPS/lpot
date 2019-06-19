from typing import Tuple

from pipelines.sklearn_wrappers import sklearn_transformer_wrapper

from sklearn import preprocessing
from scipy import special


class PolynomialFeaturesWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=preprocessing.PolynomialFeatures,
            possible_parameters={
                "degree": [2, 3]
            })

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        n_new_features = special.comb(
            N=x[1],
            k=self.chosen_parameters["degree"],
            exact=True,
            repetition=True)
        return n_new_features * x[0], (x[0], n_new_features)
