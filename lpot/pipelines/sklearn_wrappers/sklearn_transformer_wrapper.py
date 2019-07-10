from typing import Dict, List, Optional
import random
from pipelines.layers import layer_element
import sklearn_config
from typing import Tuple


class SklearnWrapper(layer_element.LayerElement):
    def __init__(self, sklearn_class, possible_parameters: Optional[Dict[str, List]] = None):
        if not possible_parameters:
            possible_parameters = {}
        self.possible_parameters = possible_parameters
        self.sklearn_class = sklearn_class
        self.chosen_parameters = self._choose_random_parameters(possible_parameters)
        self.sklearn_object = sklearn_class(**self.chosen_parameters)
        for key, value in sklearn_config.sklearn_parameters.items():
            if key in self.sklearn_object.get_params():
                self.sklearn_object.set_params(**{key: value})
        self.fitted = False

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.chosen_parameters) + ")"

    def randomly_mutate(self):
        self.chosen_parameters = self._choose_random_parameters(self.possible_parameters)
        self.sklearn_object = self.sklearn_class(**self.chosen_parameters)

    def transform(self, x):
        return self.sklearn_object.transform(x)

    def fit(self, x, y):
        self.fitted = True
        self.sklearn_object.fit(x, y)

    def is_mutable(self):
        return super().is_mutable()

    def _choose_random_parameters(self, parameters_dict: Dict[str, List]):
        return {k: random.choice(v) for k, v in parameters_dict.items()}

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], x[1])
