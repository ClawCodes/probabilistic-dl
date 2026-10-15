from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class Prediction:
    probabilities: torch.Tensor
    predictive_entropy: torch.Tensor
    expected_entropy: torch.Tensor
    mutual_information: torch.Tensor


class PredictiveModel(ABC):
    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> list[dict[str, float]]: ...

    @abstractmethod
    def predict(self, x: torch.Tensor) -> Prediction: ...
