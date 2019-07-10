from typing import Tuple

from lpot.pipelines.sklearn_wrappers import sklearn_transformer_wrapper
from lpot.pipelines.layers.layer_element import ClassifierElement


class SklearnClassifierWrapper(sklearn_transformer_wrapper.SklearnWrapper, ClassifierElement):
    def transform(self, x):
        return self.sklearn_object.predict_proba(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], 2)

    def get_classes(self):
        return list(self.sklearn_object.classes_)
