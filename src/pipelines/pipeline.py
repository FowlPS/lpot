from abc import ABC

import numpy as np
import pipeline_config
import copy
from pipelines.layers import layer
from pipelines.final_layers import final_main_layer
from sklearn import metrics
import random
from typing import Callable, Optional, List, Type, Tuple


class Pipeline(ABC):
    type: str
    layers: List[layer.Layer]

    def __init__(self, layers: List[layer.Layer], n_transformer_layers: int,
                 transformer_layers_class_list: List[Type[layer.Layer]]):
        assert (len(layers) == n_transformer_layers + 2)
        self.layers = layers
        self.n_transformer_layers = n_transformer_layers
        self.transformer_layers_class_list = transformer_layers_class_list
        self.valid=True

    def __repr__(self):
        return self.__class__.__name__ + "(layers=" + repr(self.layers) + ")"

    def __str__(self):
        return self.__class__.__name__ + "(layers=\n" + "\n".join([str(i) + "," for i in self.layers]) + ")"

    @classmethod
    def new_pipeline(cls, n_transformer_layers: int, transformer_layers_class_list: List[Type[layer.Layer]]) -> 'Pipeline':  # for now force the use of ensembles
        layers = cls.initialize_layers(n_transformer_layers=n_transformer_layers, mutable_to=transformer_layers_class_list)
        return cls(
            layers=layers,
            n_transformer_layers=n_transformer_layers,
            transformer_layers_class_list=transformer_layers_class_list,
        )

    def fit(self, x, y, refit=False, verbose: int = 1) -> None:
        if verbose > 2:
            print("Fitting: " + self.__str__())
        try:

            for element_layer in self.layers:
                element_layer.fit(x, y, refit=refit)
                x = element_layer.transform(x)
        except Exception as inst:
            print(type(inst))
            print(inst)
            print("Error in fit:")
            print(self.__str__())
            self.valid=False

    def predict(self, x) -> np.array:
        if self.valid:
            for layers_element in self.layers:
                x = layers_element.transform(x)
            return x
        else:
            return [0 for i in x]

    @classmethod
    def initialize_layers(cls, n_transformer_layers: int, mutable_to:Optional[List[Type[layer.Layer]]]=None) -> List[layer.Layer]:
        layers: List[layer.Layer] = [layer.IdentityLayer(mutable_to=mutable_to) for i in range(n_transformer_layers)]
        layers.append(cls.get_final_layers_class().get_random())
        layers.append(cls.get_default_ensembler()())
        return layers

    def get_mutation_point(self, n_epoch: int, annealing: bool, total_epochs: int) -> int:
        if not annealing:
            return random.randint(0, self.n_transformer_layers + 1)
        else:
            distribution = [(1.0 + i / 20) ** int(1 + 10 * float(n_epoch) / total_epochs) for i in
                            range(self.n_transformer_layers + 1)]
            return np.random.choice(
                a=range(self.n_transformer_layers + 1),
                p=[i / sum(distribution) for i in distribution]
            )

    def mutate(self, n_epoch: int, annealing: bool, total_epochs: int) -> 'Pipeline':
        mutation_point = self.get_mutation_point(n_epoch=n_epoch, annealing=annealing, total_epochs=total_epochs)
        new_layers: List[layer.Layer] = copy.deepcopy(self.layers)
        new_layers[mutation_point] = new_layers[mutation_point].randomly_mutate()
        for i in range(mutation_point, self.n_transformer_layers + 2):
            new_layers[i].mark_unfit()
        new_pipeline: 'Pipeline' = self.__class__(
            layers=new_layers,
            n_transformer_layers=self.n_transformer_layers,
            transformer_layers_class_list=self.transformer_layers_class_list
        )
        return new_pipeline

    def mutate_shrink(self) -> 'Pipeline':
        mutable_layer_indices = [idx for idx, element in enumerate(self.layers) if element.can_shrink()]
        mutation_point = random.choice(mutable_layer_indices) if mutable_layer_indices else -2
        new_layers: List[layer.Layer] = copy.deepcopy(self.layers)
        new_layers[mutation_point] = new_layers[mutation_point].mutate_shrink()
        for i in range(mutation_point, self.n_transformer_layers + 2):
            new_layers[i].mark_unfit()
        new_pipeline: 'Pipeline' = self.__class__(
            layers=new_layers,
            n_transformer_layers=self.n_transformer_layers,
            transformer_layers_class_list=self.transformer_layers_class_list
        )
        return new_pipeline

    def crossover(self, other: 'Pipeline', n_epoch: int, annealing: bool,
                  total_epochs: int) -> 'Pipeline':  # first layers from this, last from other
        mutation_point = self.get_mutation_point(n_epoch=n_epoch, annealing=annealing, total_epochs=total_epochs)
        new_layers: List[layer.Layer] = copy.deepcopy(self.layers[:mutation_point]) + copy.deepcopy(
            other.layers[mutation_point:])
        for i in range(mutation_point, self.n_transformer_layers + 2):
            new_layers[i].mark_unfit()
        new_pipeline: 'Pipeline' = self.__class__(
            layers=new_layers,
            n_transformer_layers=self.n_transformer_layers,
            transformer_layers_class_list=self.transformer_layers_class_list
        )
        return new_pipeline

    @staticmethod
    def get_default_transformer_layers_list() -> List[Type[layer.Layer]]:
        raise NotImplementedError()

    @staticmethod
    def get_final_layers_class() -> Type[layer.Layer]:
        raise NotImplementedError()

    @staticmethod
    def get_default_ensembler() -> Type[layer.Layer]:
        raise NotImplementedError()

    @staticmethod
    def get_default_fitness_function() -> Callable:
        raise NotImplementedError()

    def get_complexity(self, x: Tuple) -> int:
        cost = 0
        for layer_ in self.layers:
            layer_cost, x = layer_.get_complexity(x)
            cost += layer_cost
        return cost


class ClassificationPipeline(Pipeline):
    @staticmethod
    def get_default_transformer_layers_list() -> List[Type[layer.Layer]]:
        return pipeline_config.CLASSIFIER_TRANSFORMER_LAYERS_LIST

    @staticmethod
    def get_final_layers_class() -> Type[layer.Layer]:
        return final_main_layer.FinalClassifierLayer

    @staticmethod
    def get_default_ensembler() -> Type[layer.Layer]:
        return pipeline_config.DEFAULT_CLASSIFIER_ENSEMBLER

    @staticmethod
    def get_default_fitness_function() -> Callable:
        def fitness(individual: Pipeline, x, y):
            y_pred = individual.predict(x)
            return metrics.accuracy_score(y_true=y, y_pred=y_pred)

        return fitness


class RegressionPipeline(Pipeline):
    @staticmethod
    def get_default_transformer_layers_list() -> List[Type[layer.Layer]]:
        return pipeline_config.REGRESSOR_TRANSFORMER_LAYERS_LIST

    @staticmethod
    def get_final_layers_class() -> Type[layer.Layer]:
        return final_main_layer.FinalRegressorLayer  # TODO

    @staticmethod
    def get_default_ensembler() -> Type[layer.Layer]:
        return pipeline_config.DEFAULT_REGRESSOR_ENSEMBLER  # TODO

    @staticmethod
    def get_default_fitness_function() -> Callable:
        def fitness(individual: Pipeline, x, y):
            y_pred = individual.predict(x)
            return - metrics.mean_squared_error(y_true=y, y_pred=y_pred)

        return fitness
