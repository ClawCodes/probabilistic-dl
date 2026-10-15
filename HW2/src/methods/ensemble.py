from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import Prediction, PredictiveModel
from ..models import ClassifierMLP


class DeepEnsemble(PredictiveModel):
    def __init__(self, input_shape: tuple[int, ...], n_classes: int, *, members: int, hidden_dim: int, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int):
        self.input_shape, self.n_classes, self.hidden_dim = input_shape, n_classes, hidden_dim
        self.members = members
        self.models = []
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.weight_decay, self.seed = weight_decay, seed

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> list[dict[str, float]]:
        histories = []
        for member in range(self.members):
            # Set this before construction so each member's initialization is
            # explicitly tied to a distinct, reproducible seed.
            torch.manual_seed(self.seed + member)
            model = ClassifierMLP(self.input_shape, self.n_classes, self.hidden_dim)
            self.models.append(model)
            # Standard deep ensembles differ only in random initialization and
            # stochastic training order; every member sees the same data.
            train_x, train_y = x, y
            loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
            # weight_decay is the L2 penalty, equivalent to an isotropic
            # zero-mean Gaussian prior over weights (up to scaling convention).
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()
            for _ in range(self.epochs):
                model.train()
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                model.eval()
                final_loss = criterion(model(train_x), train_y).item()
                accuracy = (model(train_x).argmax(1) == train_y).float().mean().item()
            histories.append({"member": member, "train_loss": final_loss, "train_accuracy": accuracy})
        return histories

    def predict(self, x: torch.Tensor) -> Prediction:
        with torch.no_grad():
            probabilities = []
            for model in self.models:
                model.eval()
                probabilities.append(model(x).softmax(dim=1))
            samples = torch.stack(probabilities)
            mean = samples.mean(dim=0)
            entropy = -(mean * mean.clamp_min(1e-12).log()).sum(dim=1)
            member_entropy = -(samples * samples.clamp_min(1e-12).log()).sum(dim=2).mean(dim=0)
        return Prediction(mean, entropy, member_entropy, entropy - member_entropy)
