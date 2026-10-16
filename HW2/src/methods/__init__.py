from .base import PredictiveModel, Prediction, prediction_from_samples
from .ensemble import DeepEnsemble
from ..config import DatasetConfig, ExperimentConfig
from ..data import DatasetBundle


def build_method(name: str, data: DatasetBundle, ds_cfg: DatasetConfig, cfg: ExperimentConfig) -> PredictiveModel:
    """Construct an untrained method from the run config."""
    if name == "ensemble":
        return DeepEnsemble(data.feature_shape, data.n_classes, members=cfg.members, hidden_dim=ds_cfg.hidden_dim, epochs=ds_cfg.epochs, batch_size=ds_cfg.batch_size, learning_rate=ds_cfg.learning_rate, weight_decay=ds_cfg.weight_decay, seed=cfg.seed)
    if name == "mc_dropout":
        from .mc_dropout import MCDropoutPredictor
        return MCDropoutPredictor(data.feature_shape, data.n_classes)
    raise ValueError(f"Unknown method '{name}'.")


__all__ = ["PredictiveModel", "Prediction", "prediction_from_samples", "DeepEnsemble", "build_method"]
