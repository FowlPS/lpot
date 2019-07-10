from abc import abstractmethod, ABC

from typing import List, Optional, Type, Tuple

from lpot.pipelines.layers.layer_element import LayerElement

import random
import copy


class Layer(ABC):
    elements: List[LayerElement]

    def __init__(self, mutable_to: 'List[Type[Layer]]' = None):  # in subclasses this will randomize hyperparameters
        self.fitted = False
        self.mutable_to = mutable_to
        self.elements = []

    def __repr__(self):
        return self.__class__.__name__ + "(elements=" + str(self.elements) + ", mutable_to=" + str(
            self.mutable_to) + ")"

    def __str__(self):
        return self.__class__.__name__ + (("(elements=" + str(self.elements)) if self.elements else "(") + ")"

    def can_shrink(self) -> bool:
        return True  # default

    @classmethod
    def get_random(cls, mutable_to: Optional[List['Layer']] = None):
        raise NotImplementedError()

    def randomly_mutate(self):
        if not self.mutable_to:
            return copy.deepcopy(self)
        new_layer_class = random.choice(self.mutable_to)
        return new_layer_class.get_random(
            mutable_to=self.mutable_to)  # in subclasses this will be called only with some chance, usually layer will just shrink within itself

    def mutate_shrink(
            self) -> 'Layer':  # this has to return itself, so that it's possible to replace Layer with IdentityLayer #will only be called from subclass with some chance, like if it cannot reduce number of elements
        return IdentityLayer()

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return sum([i.get_complexity(x)[0] for i in self.elements]), x

    def fit(self, x, y, refit=False):
        if self.fitted and not refit:
            return
        for element in self.elements:
            element.fit(x=x, y=y)
        self.fitted = True

    def mark_unfit(self):
        self.fitted = False

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError()  # probably concat of features generated like above? Actually, it should depend on layer type


class IdentityLayer(Layer):
    def can_shrink(self) -> bool:
        return False

    def mutate_shrink(self) -> Layer:
        raise NotImplementedError()

    def get_complexity(self, x:Tuple) -> Tuple[int, Tuple]:
        return 0, x

    def transform(self, x):
        return x

    def fit(self, x, y, refit=False):
        self.fitted = True
        return

    @classmethod
    def get_random(cls, mutable_to: Optional[List[Type[Layer]]] = None):
        return IdentityLayer(mutable_to=mutable_to)
