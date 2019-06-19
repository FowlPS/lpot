from src.pipelines.layers import layer
from src.pipelines.feature_generation import feature_generation_layer
from src.pipelines.features_processing import feature_preprocessing_layer
from src.pipelines.ensemblers import voting_soft
from src.pipelines.ensemblers import ensemble_mean

CLASSIFIER_TRANSFORMER_LAYERS_LIST = [
    feature_generation_layer.FeatureGenerationClassifierLayer,
    layer.IdentityLayer,
    feature_preprocessing_layer.ClassifierFeatureSelectionLayer,
    feature_preprocessing_layer.ScalerLayer,
    feature_preprocessing_layer.TransformationLayer,
]

REGRESSOR_TRANSFORMER_LAYERS_LIST = [
    feature_generation_layer.FeatureGenerationRegressorLayer,
    layer.IdentityLayer,
    feature_preprocessing_layer.RegressorFeatureSelectionLayer,
    feature_preprocessing_layer.ScalerLayer,
    feature_preprocessing_layer.TransformationLayer,
]

DEFAULT_CLASSIFIER_ENSEMBLER = voting_soft.VotingSoft

DEFAULT_REGRESSOR_ENSEMBLER = ensemble_mean.EnsembleMean