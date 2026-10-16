from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import Prediction, PredictiveModel, prediction_from_samples
from ..models import ClassifierMLP


class DeepEnsemble(PredictiveModel):
    def __init__(self, input_shape: tuple[int, ...], n_classes: int, *, members: int, hidden_dim: int, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int):
        self.input_shape, self.n_classes, self.hidden_dim = input_shape, n_classes, hidden_dim
        self.members = members
        self.models = []
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.weight_decay, self.seed = weight_decay, seed

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        self.models = []
        history, finals = [], []
        for member in range(self.members):
            # Set this before construction so each member's initialization is
            # explicitly tied to a distinct, reproducible seed. Member k is
            # therefore identical for any ensemble size, so smaller ensembles
            # are prefixes of a larger one.
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
            for epoch in range(self.epochs):
                model.train()
                total_loss, correct = 0.0, 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_y)
                    correct += (logits.argmax(1) == batch_y).sum().item()
                history.append({"member": member, "epoch": epoch + 1, "train_loss": total_loss / len(train_y), "train_accuracy": correct / len(train_y)})
            with torch.no_grad():
                model.eval()
                logits = model(train_x)
                final_loss = criterion(logits, train_y).item()
                accuracy = (logits.argmax(1) == train_y).float().mean().item()
            finals.append({"member": member, "train_loss": final_loss, "train_accuracy": accuracy})
        return history, finals

    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> Prediction:
        models = self.models[:n_samples]
        with torch.no_grad():
            probabilities = []
            for model in models:
                model.eval()
                probabilities.append(model(x).softmax(dim=1))
            return prediction_from_samples(torch.stack(probabilities))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for member, model in enumerate(self.models):
            torch.save(model.state_dict(), directory / f"member_{member:02d}.pt")

    def load(self, directory: Path) -> None:
        paths = sorted(directory.glob("member_*.pt"))
        if len(paths) != self.members:
            raise FileNotFoundError(f"Expected {self.members} member weights in {directory}, found {len(paths)}.")
        self.models = []
        for path in paths:
            model = ClassifierMLP(self.input_shape, self.n_classes, self.hidden_dim)
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            self.models.append(model)
