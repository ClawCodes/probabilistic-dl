from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Prediction:
    probabilities: torch.Tensor
    predictive_entropy: torch.Tensor
    expected_entropy: torch.Tensor
    mutual_information: torch.Tensor


def prediction_from_samples(samples: torch.Tensor) -> Prediction:
    """Summarize stacked per-sample probabilities of shape (S, N, C)."""
    mean = samples.mean(dim=0)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(dim=1)
    member_entropy = -(samples * samples.clamp_min(1e-12).log()).sum(dim=2).mean(dim=0)
    return Prediction(mean, entropy, member_entropy, entropy - member_entropy)


class PredictiveModel(ABC):
    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Train and return (per-epoch history rows, final per-member rows)."""

    @abstractmethod
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> Prediction:
        """Predict with the first `n_samples` members / stochastic passes (all if None)."""

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> None: ...
