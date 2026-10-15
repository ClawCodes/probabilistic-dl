"""Reserved implementation point for the MC-dropout comparison."""
from .base import PredictiveModel, Prediction


class MCDropoutPredictor(PredictiveModel):
    """Implement this class using ClassifierMLP(dropout=...) and T stochastic passes.

    Keep dropout active at prediction time, stack T probability tensors, then
    construct Prediction exactly as DeepEnsemble.predict does. This makes plots,
    metrics, and the experiment runner method-agnostic.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MC dropout is scaffolded but not implemented for this stage of HW2.")

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x) -> Prediction:
        raise NotImplementedError
