# LPOT - Layer-based Pipeline Optimization Tool
This is still an early version of the code, I am aware that it requires a refactoring, docs and tests.

## Installation
Easiest way is to install the software via pip:

```bash
pip install git+https://github.com/FowlPS/lpot
```

## Introduction

LPOT is sklearn-based machine learning pipeline optimization tool.

## Sample usage

```python


# Get the dataset 
from sklearn import datasets
from sklearn import model_selection


iris = datasets.load_iris()

x_train, x_test, y_train, y_test = model_selection.train_test_split(iris["data"], iris["target"])

# Create LPOT

from lpot.evolution import evolution_manager

lpot_instance = evolution_manager.EvolutionManager(
 n_layers=4,
 verbosity=2,
 population_size=10,
 nb_epochs=10,
 max_fitness=1.0)


# Perform fitting

lpot_instance.fit(x_train, y_train)

# evaluate
from sklearn import metrics

lpot_prediction = lpot_instance.predict(x_test)

print("LPOT accuracy: " + str(metrics.accuracy_score(y_test, lpot_prediction)))
```