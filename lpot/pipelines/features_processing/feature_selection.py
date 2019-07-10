from typing import Tuple

from lpot.pipelines.sklearn_wrappers import sklearn_transformer_wrapper

from sklearn import feature_selection
from sklearn import svm
from sklearn import linear_model


class SelectKBestClassifierWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=feature_selection.SelectKBest,
            possible_parameters={
                "k": [5, 10, 15]
            })

    def fit(self, x, y):
        if x.shape[1] <= self.chosen_parameters["k"]:
            self.fitted = True
            return
        return super().fit(x, y)

    def transform(self, x):
        if x.shape[1] <= self.chosen_parameters["k"]:
            return x
        return super().transform(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], min(x[1], self.chosen_parameters["k"]))


class SelectKBestRegressorWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=feature_selection.SelectKBest,
            possible_parameters={
                "k": [5, 10, 15],
                "score_func": [feature_selection.f_regression]
            })

    def fit(self, x, y):
        if x.shape[1] <= self.chosen_parameters["k"]:
            self.fitted = True
            return
        return super().fit(x, y)

    def transform(self, x):
        if x.shape[1] <= self.chosen_parameters["k"]:
            return x
        return super().transform(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], min(x[1], self.chosen_parameters["k"]))


class SelectPercentileClassifierWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=feature_selection.SelectPercentile,
            possible_parameters={
                "percentile": [10, 30, 50, 70],
            })

    def transform(self, x):
        transformed =  super().transform(x)
        if transformed.shape[1] == 0:
            return x
        return transformed


    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], max(int(self.chosen_parameters["percentile"] * x[1] / 100), 1))


class SelectPercentileRegressorWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=feature_selection.SelectPercentile,
            possible_parameters={
                "percentile": [10, 30, 50, 70],
                "score_func": [feature_selection.f_regression]
            })

    def transform(self, x):
        transformed =  super().transform(x)
        if transformed.shape[1] == 0:
            return x
        return transformed

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], (x[0], max(int(self.chosen_parameters["percentile"] * x[1] / 100), 1))


class RFEClasssifierWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        svm_estimator = svm.SVC(kernel="linear")
        logistic_estimator = linear_model.LogisticRegression(solver="lbfgs")
        super().__init__(
            sklearn_class=feature_selection.RFE,
            possible_parameters={
                "estimator": [
                    # svm_estimator,
                    logistic_estimator
                ],
                "n_features_to_select": [None, 5, 10]
            })

    def fit(self, x, y):
        if self.chosen_parameters["n_features_to_select"] and x.shape[1] <= self.chosen_parameters[
            "n_features_to_select"] or (not  self.chosen_parameters["n_features_to_select"] and x.shape[1] <= 1):
            self._complexity = 0
            self.fitted = True
            return
        return super().fit(x, y)

    def transform(self, x):
        if self.chosen_parameters["n_features_to_select"] and x.shape[1] <= self.chosen_parameters[
            "n_features_to_select"] or (not  self.chosen_parameters["n_features_to_select"] and x.shape[1] <= 1):
            return x
        return super().transform(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        n_features = self.chosen_parameters["n_features_to_select"]
        return (x[0] * (x[1] ** 2) + x[1] ** 3) * x[1], \
               (x[0], min(x[1], n_features) if n_features else max(1, int(x[1] / 2)))


class RFERegressorWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        svm_estimator = svm.SVR(kernel="linear")
        linear_estimator = linear_model.LinearRegression()
        super().__init__(
            sklearn_class=feature_selection.RFE,
            possible_parameters={
                "estimator": [
                    # svm_estimator,
                    linear_estimator
                ],
                "n_features_to_select": [None, 5, 10]
            })

    def fit(self, x, y):
        if self.chosen_parameters["n_features_to_select"] and x.shape[1] <= self.chosen_parameters[
            "n_features_to_select"] or (not  self.chosen_parameters["n_features_to_select"] and x.shape[1] <= 1):
            self._complexity = 0
            self.fitted = True
            return
        return super().fit(x, y)

    def transform(self, x):
        if self.chosen_parameters["n_features_to_select"] and x.shape[1] <= self.chosen_parameters[
            "n_features_to_select"] or (not  self.chosen_parameters["n_features_to_select"] and x.shape[1] <= 1):
            return x
        return super().transform(x)

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        n_features = self.chosen_parameters["n_features_to_select"]
        return (x[0] * (x[1] ** 2) + x[1] ** 3) * x[1], \
               (x[0], min(x[1], n_features) if n_features else max(1, int(x[1] / 2)))


class VarianceThresholdWrapper(sklearn_transformer_wrapper.SklearnWrapper):
    def __init__(self):
        super().__init__(
            sklearn_class=feature_selection.VarianceThreshold,
            possible_parameters={}
        )

    def transform(self, x):
        transformed =  super().transform(x)
        if transformed.shape[1] == 0:
            return x
        return transformed

    def get_complexity(self, x: Tuple) -> Tuple[int, Tuple]:
        return x[0] * x[1], x
