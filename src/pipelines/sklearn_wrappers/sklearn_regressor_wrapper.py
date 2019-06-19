from typing import Tuple

from pipelines.sklearn_wrappers import sklearn_transformer_wrapper


class SklearnRegressorWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def transform(self, x):
        return self.sklearn_object.predict(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], 1)
