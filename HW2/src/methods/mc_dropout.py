"""Reserved implementation point for the MC-dropout comparison."""
from pathlib import Path
import torch
from .base import PredictiveModel, Prediction


class MCDropoutPredictor(PredictiveModel):
    """Implement this class using ClassifierMLP(dropout=...) and T stochastic passes.

    Keep dropout active at prediction time, stack T probability tensors, then
    build the result with `prediction_from_samples` exactly as DeepEnsemble.predict
    does. `predict(x, n_samples=k)` should use k stochastic passes so the pipeline's
    size sweep works unchanged. This makes plots, metrics, and the pipeline
    method-agnostic.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MC dropout is scaffolded but not implemented for this stage of HW2.")

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> Prediction:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    def load(self, directory: Path) -> None:
        raise NotImplementedError
