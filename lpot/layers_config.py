from lpot.pipelines.classifiers import linear as linearcls
from lpot.pipelines.classifiers import tree_classifiers
from lpot.pipelines.classifiers import knn as knnc

from lpot.pipelines.regressors import knn as knnr
from lpot.pipelines.regressors import linear as linearreg
from lpot.pipelines.regressors import tree_regressors

from lpot.pipelines.features_processing import scalers
from lpot.pipelines.features_processing import polynomial_featuers
from lpot.pipelines.features_processing import pca
from lpot.pipelines.features_processing import feature_selection


CLASSIFIER_FINAL_ELEMENTS_LIST = [
    linearcls.LogisticFinalElement,
    linearcls.SVMCFinalElement,
    tree_classifiers.RandomForestClassifier,
    tree_classifiers.DecisionTreeClassifier,
    tree_classifiers.GradientBoostingClassifier,
    knnc.KNNCFinalElement,
]

CLASSIFIER_FEATURE_GENERATION_ELEMENTS_LIST = [
    linearcls.LogisticFinalElement,
    linearcls.SVMCFinalElement,
    tree_classifiers.RandomForestClassifier,
    tree_classifiers.DecisionTreeClassifier,
    tree_classifiers.GradientBoostingClassifier,
    knnc.KNNCFinalElement,
]

REGRESSOR_FINAL_ELEMENTS_LIST = [
    knnr.KNNRFinalElement,
    linearreg.LinearFinalElement,
    linearreg.SVMRFinalElement,
    tree_regressors.GradientBoostingRegressor,
    tree_regressors.DecisionTreeRegressor,
    tree_regressors.RandomForestRegressor,
]

REGRESSOR_FEATURE_GENERATION_ELEMENTS_LIST = [
    knnr.KNNRFinalElement,
    linearreg.LinearFinalElement,
    linearreg.SVMRFinalElement,
    tree_regressors.GradientBoostingRegressor,
    tree_regressors.DecisionTreeRegressor,
    tree_regressors.RandomForestRegressor,
]

CLASSIFIER_FEATURE_SELECTION_ELEMENTS_LIST = [
    feature_selection.SelectKBestClassifierWrapper,
    feature_selection.SelectPercentileClassifierWrapper,
    feature_selection.RFEClasssifierWrapper,
    feature_selection.VarianceThresholdWrapper,
]

REGRESSOR_FEATURE_SELECTION_ELEMENTS_LIST = [
    feature_selection.RFERegressorWrapper,
    feature_selection.VarianceThresholdWrapper,
    feature_selection.SelectKBestRegressorWrapper,
    feature_selection.SelectPercentileRegressorWrapper,
]

FEATURES_TRANSFORMATION_ELEMENTS_LIST = [
    polynomial_featuers.PolynomialFeaturesWrapper,
    pca.PCAWrapper,
]

SCALERS_ELEMENTS_LIST = [
    scalers.MinMaxScalerWrapper,
    scalers.RobustScalerWrapper,
    scalers.StandardScalerWrapper,
]
