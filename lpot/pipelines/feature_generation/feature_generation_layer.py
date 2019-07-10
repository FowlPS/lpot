from abc import ABC

import numpy as np
import layers_config

from pipelines.layers.layer import Layer
from pipelines.layers.layer_element import LayerElement
from typing import List, Optional, Type, Tuple
import random
import copy


class FeatureGenerationLayer(Layer, ABC):
    elements_classes: List[Type[LayerElement]]

    def __init__(
            self,
            elements: List[LayerElement],
            mutable_to: List[Type[Layer]] = None):
        super().__init__(mutable_to=mutable_to)
        self.elements = elements

    @classmethod
    def get_random(cls, mutable_to: Optional[List[Layer]] = None):
        return cls(
            elements=[random.choice(cls.elements_classes)()],
            mutable_to=mutable_to)

    def randomly_mutate(self):
        available_mutations = ["add", "shrink", "mute_layer", "mute_element"]
        if len(self.elements) > 1:
            available_mutations += ["replace"]
        mutation = random.choice(available_mutations)
        if mutation == "add":
            return self.__class__(elements=copy.deepcopy(self.elements)
                                           + [random.choice(self.elements_classes)()],
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
        elif mutation == "mute_layer":
            return super().randomly_mutate()

    def mutate_shrink(self) -> 'Layer':
        new_elements = copy.deepcopy(self.elements)
        replaced_idx = random.randint(0, len(new_elements) - 1)
        del new_elements[replaced_idx]
        if new_elements:
            return self.__class__(elements=new_elements,
                                  mutable_to=self.mutable_to)
        else:
            return super().mutate_shrink()

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return sum([element.get_complexity(x)[0] for element in self.elements]), (x[0], x[1] + 2 * len(self.elements))


class FeatureGenerationClassifierLayer(FeatureGenerationLayer):
    elements_classes = layers_config.CLASSIFIER_FEATURE_GENERATION_ELEMENTS_LIST

    def __init__(self, elements: List[LayerElement], mutable_to: List[Type[Layer]] = None):
        super().__init__(
            mutable_to=mutable_to,
            elements=elements
        )

    def transform(self, x):
        generated_features = np.asarray([element.transform(x) for element in self.elements])
        new_features = np.transpose(generated_features, (1, 0, 2)).reshape((x.shape[0], -1))
        return np.concatenate((x, new_features), axis=1)


class FeatureGenerationRegressorLayer(FeatureGenerationLayer):
    elements_classes = layers_config.REGRESSOR_FEATURE_GENERATION_ELEMENTS_LIST

    def __init__(self, elements: List[LayerElement], mutable_to: List[Type[Layer]] = None):
        super().__init__(
            mutable_to=mutable_to,
            elements=elements
        )

    def transform(self, x):
        generated_features = np.asarray([element.transform(x) for element in self.elements])
        new_features = np.transpose(generated_features, (1, 0)).reshape((x.shape[0], -1))
        return np.concatenate((x, new_features), axis=1)
