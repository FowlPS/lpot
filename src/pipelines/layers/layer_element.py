from typing import Tuple


class LayerElement:
    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __str__(self):
        return self.__repr__()

    def is_mutable(self):
        return True

    def fit(self, x, y):
        raise NotImplementedError()

    def transform(self, x):
        raise NotImplementedError()

    def randomly_mutate(self):
        raise NotImplementedError()  # this only mutates within itself

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        raise NotImplementedError()

class ClassifierElement(LayerElement):
    def get_classes(self):
        raise NotImplementedError()
