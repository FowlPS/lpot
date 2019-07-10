import numpy as np
from lpot.pipelines.ensemblers import ensemble
from lpot.pipelines.layers import layer
from typing import List, Tuple


class VotingSoft(ensemble.ClassififerEnsembler):
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
        x, classlist = x
        x = np.sum(x, axis=1)
        class_order = classlist[0]
        return np.array([class_order[i] for i in np.argmax(x, axis=1)])

    def __init__(self, mutable_to: List[layer.Layer.__class__] = None):
        super().__init__(mutable_to)

    def fit(self, x, y, refit: bool = False):
        self.fitted = True
