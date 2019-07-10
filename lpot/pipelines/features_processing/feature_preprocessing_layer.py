from typing import List, Type, Tuple

import random
import copy

import layers_config
from pipelines.layers import layer
from pipelines.layers import layer_element


class FeaturePreprocessingLayer(layer.Layer):
    elements_classes: List[Type[layer_element.LayerElement]]

    def __init__(self, elements: List[layer.LayerElement], mutable_to: List[Type[layer.Layer]] = None):
        super().__init__(mutable_to=mutable_to)
        assert len(elements) == 1
        self.elements = elements

    def transform(self, x):
        return self.elements[0].transform(x)

    @classmethod
    def get_random(cls, mutable_to=None):
        return cls(elements=[random.choice(cls.elements_classes)()])

    def randomly_mutate(self):
        available_mutations = ["replace", "shrink", "change", "mute_element"]
        mutation = random.choice(available_mutations)
        if mutation == "replace":
            return self.__class__(elements=[random.choice(self.elements_classes)()],
                                  mutable_to=self.mutable_to)
        elif mutation == "shrink":
            return super().mutate_shrink()
        elif mutation == "mute_element":
            new_elements = copy.deepcopy(self.elements)
            replaced_idx = random.randint(0, len(new_elements) - 1)
            new_elements[replaced_idx].randomly_mutate()
            return self.__class__(elements=new_elements,
                                  mutable_to=self.mutable_to)
        elif mutation == "change":
            return super().randomly_mutate()

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return self.elements[0].get_complexity(x)


class ScalerLayer(FeaturePreprocessingLayer):
    elements_classes = layers_config.SCALERS_ELEMENTS_LIST


class ClassifierFeatureSelectionLayer(FeaturePreprocessingLayer):
    elements_classes = layers_config.CLASSIFIER_FEATURE_SELECTION_ELEMENTS_LIST


class RegressorFeatureSelectionLayer(FeaturePreprocessingLayer):
    elements_classes = layers_config.REGRESSOR_FEATURE_SELECTION_ELEMENTS_LIST


class TransformationLayer(FeaturePreprocessingLayer):
    elements_classes = layers_config.FEATURES_TRANSFORMATION_ELEMENTS_LIST
