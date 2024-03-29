from typing import List, Callable, Type, Optional

from sklearn import model_selection
import functools
from lpot.pipelines.pipeline import Pipeline, ClassificationPipeline, RegressionPipeline
from lpot.pipelines.layers import layer

import random
import math


class EvolutionManager:
    n_layers: int
    transformer_layers_list: List[Type[layer.Layer]]
    max_complexity: int
    fitted: bool
    population: List
    pipeline_class: Type[Pipeline]
    best: Pipeline
    mut_pb: float
    cross_pb: float
    MAX_RETRIES: int

    def __init__(self,
                 n_layers: int = 4,
                 transformer_layers_list: Optional[List[Type[layer.Layer]]] = None,
                 keep_best_fraction=1 / 3,
                 max_complexity: int = 1e11,
                 classification: bool = True,
                 mut_part: float = 0.9,
                 fitness: Callable = None,  # individual, x, y
                 verbosity: int = 3,
                 max_fitness: Optional[float] = None,
                 MAX_RETRIES: int = 100,
                 annealing: bool = True,
                 population_size: int = 10,
                 nb_epochs: int = 50,
                 refit: bool = True,
                 ):

        """
        :param n_layers: Number of non-mandatory transformer layers 
        :param transformer_layers_list: Optional list of allowed transformer layer types. Leave None for all possible layers
        :param keep_best_fraction: Fraction of individuals that is taken from one generation to another
        :param max_complexity: Maximum estimated complexity of a single pipeline.
        :param classification: Boolean indicating whether the problem is a classification one. Change to False for regression
        :param mut_part: Part of individuals created by mutation (rest comes from crossover)
        :param fitness: Fitness function, callable taking individual, x, y, and returns a comparable type, used to rank individuals
        :param verbosity: Verbosity
        :param max_fitness: Maximum possible fitness. Used for early stopping (i.e. 1.0 for accuracy) 
        :param MAX_RETRIES: Max retries of shrinking a pipeline to allowed complexity. Increase if trying to significantly lower max_complexity.
        :param annealing: Whether to use annealing optimization (slightly faster, slightly lowers results)
        :param population_size: Number of individuals in each generation
        :param nb_epochs: Number of generations
        """
        self.pipeline_class = ClassificationPipeline if classification else RegressionPipeline
        if transformer_layers_list is None:
            transformer_layers_list = self.pipeline_class.get_default_transformer_layers_list()
        self.n_layers = n_layers
        self.transformer_layers_list = transformer_layers_list
        self.max_complexity = max_complexity
        self.fitted = False
        self.fitness = fitness if fitness else self.pipeline_class.get_default_fitness_function()
        self.population = []  # ordered by fitness?
        self.best: Optional[Pipeline] = None
        self.mut_pb = mut_part
        self.cross_pb = 1 - mut_part
        self.verbosity = verbosity
        self.max_fitness = max_fitness
        self.keep_best_fraction = keep_best_fraction
        self.MAX_RETRIES = MAX_RETRIES
        self.annealing = annealing
        self.population_size = population_size
        self.nb_epochs = nb_epochs
        self.refit = refit

    def reset(self):
        """
        clears population - use to free memory
        :return: 
        """
        self.population = []

    def fit(self,
            data,
            labels,
            population_size: Optional[int] = None,
            nb_epochs: Optional[int] = None
            ):
        """
        Performs search for best pipeline architecture
        :param data: Data  
        :param labels: Labels
        :param population_size: population size (overrides init) 
        :param nb_epochs: number of generations (overrides init)
        """
        population_size = population_size if population_size else self.population_size
        nb_epochs = nb_epochs if nb_epochs else self.nb_epochs
        self.population = []
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.25,
                                                                            random_state=42)

        test_fitness = functools.partial(
            self.fitness,
            x=x_test,
            y=y_test,
        )

        for i in range(population_size):
            new_pipeline = self.pipeline_class.new_pipeline(
                transformer_layers_class_list=self.transformer_layers_list,
                n_transformer_layers=self.n_layers
            )  # add parameters
            new_pipeline = self._fit_individual(
                individual=new_pipeline,
                data=x_train,
                labels=y_train,
            )
            self.population.append(new_pipeline)
        self.population = sorted(self.population, reverse=True, key=lambda ind: test_fitness(
            ind))  # note: how about a test fitness, which would be a partial of fitness?
        for i in range(nb_epochs):
            self._run_epoch(train_data=x_train, train_labels=y_train, n_epoch=i, fitness_fun=test_fitness,
                            total_epochs=nb_epochs)
            if self.max_fitness is not None and math.isclose(test_fitness(self.best), self.max_fitness):
                print("Found best possible solution, ending evolution")
                break
        if self.refit:
            self.best.fit(data, labels, refit=True, verbose=self.verbosity)
        self.fitted = True

    def _run_epoch(self,
                   train_data,
                   train_labels,
                   n_epoch: int,
                   fitness_fun: Callable,
                   total_epochs: int):
        self.population = self._generate_new_population(population=self.population, data=train_data, labels=train_labels,
                                                        fitness_fun=fitness_fun, n_epoch=n_epoch,
                                                        total_epochs=total_epochs)
        self.best = self._select_k_best(self.population, 1)[0]
        if self.verbosity > 0:
            print("Epoch {epoch_nr} finished, best val fitness={fitness_value}".format(
                epoch_nr=n_epoch,
                fitness_value=fitness_fun(self.best)
            ))
        if self.verbosity > 1:
            print("Current best: {best}".format(best=str(self.best)))
        if self.verbosity > 4:
            print("Current population:")
            for i in self.population:
                print(str(i) + " fitness: " + str(fitness_fun(i)))

    def _generate_new_population(self, population, data, labels, fitness_fun: Callable, n_epoch: int, total_epochs: int):
        population_to_keep = self._select_k_best(population, max(int(len(population) * self.keep_best_fraction), 2))
        if self.verbosity > 3:
            print("Population kept:")
            for i in population_to_keep:
                print(str(i))
                if self.verbosity > 5:
                    print("Fitness: " + str(fitness_fun(i)))
        new_population = [] + population_to_keep
        new_population += self._create_new_individuals(population_to_keep, len(population) - len(population_to_keep),
                                                       data=data,
                                                       labels=labels,
                                                       n_epoch=n_epoch,
                                                       total_epochs=total_epochs)
        for individual in new_population:
            if self.verbosity > 10:
                print("Fitting: " + str(individual))
        new_population = [self._fit_individual(
            individual=individual,
            data=data,
            labels=labels) for individual in new_population]
        return sorted(new_population, reverse=True, key=lambda ind: (fitness_fun(ind), -ind.get_complexity(data.shape)))

    def _select_k_best(self, population: List[Pipeline], k: int) -> List[Pipeline]:
        return population[:k]

    def _fit_individual(self, individual: Pipeline, data, labels) -> Pipeline:
        retries = 0
        while retries < self.MAX_RETRIES:
            if individual.get_complexity(data.shape) <= self.max_complexity:
                individual.fit(data, labels, verbose=self.verbosity)
                return individual
            else:
                individual = individual.mutate_shrink()
                retries += 1
        raise ValueError("Failed to create simple enough individual. Try increasing max_complexity or MAX_RETRIES")

    def _create_new_individuals(self, population: List[Pipeline], k: int, data, labels, n_epoch: int, total_epochs: int):
        new_mutated_population = random.choices(population, k=int(math.floor(k * self.mut_pb)))
        new_crossed_parents = [random.sample(population, k=2) for i in range(int(math.ceil(k * self.cross_pb)))]
        updated_population = []

        for individual in new_mutated_population:
            mutated_individual = individual.mutate(annealing=self.annealing, n_epoch=n_epoch, total_epochs=total_epochs)
            mutated_individual = self._fit_individual(
                individual=mutated_individual,
                data=data,
                labels=labels
            )
            updated_population.append(mutated_individual)

        for [individual_1, individual_2] in new_crossed_parents:
            crossed_individual = individual_1.crossover(other=individual_2, annealing=self.annealing, n_epoch=n_epoch,
                                                        total_epochs=total_epochs)
            crossed_individual = self._fit_individual(
                individual=crossed_individual,
                data=data,
                labels=labels
            )
            updated_population.append(crossed_individual)

        updated_population = updated_population + random.choices(population, k=(k - len(updated_population)))

        return updated_population

    def get_best(self) -> Pipeline:
        """
        Returns best found pipeline
        :return: Best found pipeline 
        """
        return self.best

    def predict(self, x):
        """
        This one predicts using the best found model
        :param: Data for prediction
        :return: predictions
        """
        assert self.fitted, "Call of predict before fit"
        return self.best.predict(x)
