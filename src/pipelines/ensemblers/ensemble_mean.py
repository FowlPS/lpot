import numpy as np
from pipelines.ensemblers import ensemble
from pipelines.layers import layer
from typing import List, Tuple


class EnsembleMean(ensemble.RegressorEnsembler):
    def can_shrink(self) -> bool:
        return False

    def mutate_shrink(self) -> layer.Layer:
        pass

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0],)

    def mark_unfit(self):
        super().mark_unfit()

    def randomly_mutate(self):
        return super().randomly_mutate()

    def transform(self, x):
        x = np.mean(x, axis=1)
        return x

    def __init__(self, mutable_to: List[layer.Layer.__class__] = None):
        super().__init__(mutable_to)

    def fit(self, x, y, refit: bool = False):
        self.fitted = True
