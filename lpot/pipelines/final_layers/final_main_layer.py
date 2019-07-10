import numpy as np
from typing import Tuple

from pipelines.layers.layer import Layer
from pipelines.layers.layer_element import LayerElement, ClassifierElement

from typing import List, Type
import random
import copy
import layers_config


class FinalLayer(Layer):
    elements_classes: List[Type[Layer]]

    def __init__(self, elements: List[LayerElement], mutable_to: List[Type[Layer]] = None):
        super().__init__(mutable_to=mutable_to)
        self.elements = elements

    @classmethod
    def get_random(cls, mutable_to=None):
        return cls(elements=[random.choice(cls.elements_classes)()])

    def can_shrink(self) -> bool:
        return len(self.elements) > 1

    def randomly_mutate(self):
        available_mutations = ["add", "replace", "mute_element"]
        if len(self.elements) > 1:
            available_mutations += ["shrink"]
        mutation = random.choice(available_mutations)
        if mutation == "add":
            new_elements = copy.deepcopy(self.elements) + [random.choice(self.elements_classes)()]
            return self.__class__(elements=new_elements,
                                  mutable_to=self.mutable_to)
        elif mutation == "replace":
            new_elements = copy.deepcopy(self.elements)
            replaced_idx = random.randint(0, len(new_elements) - 1)
            new_elements[replaced_idx] = random.choice(self.elements_classes)()
            return self.__class__(elements=new_elements,
                                  mutable_to=self.mutable_to)
        elif mutation == "mute_element":
            new_elements = copy.deepcopy(self.elements)
            replaced_idx = random.randint(0, len(new_elements) - 1)
            new_elements[replaced_idx].randomly_mutate()
            return self.__class__(elements=new_elements,
                                  mutable_to=self.mutable_to)
        elif mutation == "shrink":
            return self.mutate_shrink()

    def mutate_shrink(self) -> 'Layer':
        new_elements = copy.deepcopy(self.elements)
        replaced_idx = random.randint(0, len(new_elements) - 1)
        del new_elements[replaced_idx]
        if not new_elements:
            new_elements = [random.choice(self.elements_classes)()]
        return self.__class__(elements=new_elements,
                              mutable_to=self.mutable_to)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return sum([i.get_complexity(x)[0] for i in self.elements]), (
            x[0], sum([i.get_complexity(x)[1][1] for i in self.elements]),)


class FinalClassifierLayer(FinalLayer):
    elements_classes = layers_config.CLASSIFIER_FINAL_ELEMENTS_LIST
    elements: List[ClassifierElement]


    def transform(self, x):
        return np.transpose(np.asarray([element.transform(x) for element in self.elements]), (1, 0, 2)),\
               [element.get_classes() for element in self.elements]


class FinalRegressorLayer(FinalLayer):
    elements_classes = layers_config.REGRESSOR_FINAL_ELEMENTS_LIST

    def transform(self, x):
        return np.transpose(np.asarray([element.transform(x) for element in self.elements]))
