from typing import Tuple

from pipelines.sklearn_wrappers import sklearn_transformer_wrapper

from sklearn import preprocessing


class StandardScalerWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=preprocessing.StandardScaler,
            possible_parameters={})

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], x[1])


class MinMaxScalerWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=preprocessing.MinMaxScaler,
            possible_parameters={})

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], x[1])


class RobustScalerWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=preprocessing.RobustScaler,
            possible_parameters={})

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], x[1])
