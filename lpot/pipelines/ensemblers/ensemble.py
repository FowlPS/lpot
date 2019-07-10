from lpot.pipelines.layers import layer
from typing import Tuple

class Ensembler(layer.Layer):  # soft/hard voting, logistic regression
    def can_shrink(self) -> bool:
        return False

    def mutate_shrink(self) -> layer.Layer:
        raise NotImplementedError()

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        raise NotImplementedError()

    def transform(self, x):
        raise NotImplementedError()

    def fit(self, x, y, refit: bool = False):
        raise NotImplementedError()

    @staticmethod
    def get_default():  # per class
        raise NotImplementedError()


class ClassififerEnsembler(Ensembler):  #these two will differ in dimensionality #not sklearn unfortunately
    pass


class RegressorEnsembler(Ensembler):
    pass
