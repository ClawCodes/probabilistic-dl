"""MC dropout: one dropout network, with T stochastic passes as ensemble "members"."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import PredictiveModel
from ..metrics import split_metrics
from ..models import ClassifierMLP

# Seed offset for prediction-time dropout masks, kept far from training seeds.
PASS_SEED_OFFSET = 1_000_000


class MCDropoutPredictor(PredictiveModel):
    def __init__(self, input_shape: tuple[int, ...], n_classes: int, *, hidden_dim: int, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, seed: int, dropout: float, samples: int):
        self.input_shape, self.n_classes, self.hidden_dim = input_shape, n_classes, hidden_dim
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.weight_decay, self.seed = weight_decay, seed
        self.dropout, self.samples = dropout, samples
        self.model = None

    def _build(self) -> ClassifierMLP:
        return ClassifierMLP(self.input_shape, self.n_classes, self.hidden_dim, dropout=self.dropout)

    def _stochastic_passes(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Stack n forward passes with dropout active: (n, N, C)."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        # Dropout draws from the global RNG. Forking keeps evaluation from
        # perturbing training, and a fixed seed makes pass t reproducible.
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(self.seed + PASS_SEED_OFFSET)
            return torch.stack([self.model(x).softmax(dim=1) for _ in range(n)])

    def fit(self, x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor) -> list[dict]:
        splits = {"train": (x, y), "test": (x_test, y_test)}
        torch.manual_seed(self.seed)
        self.model = model = self._build()
        loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        # Same L2/Gaussian-prior regularization as the deep ensemble; with
        # dropout this is the approximate variational objective of Gal & Ghahramani.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        history = []
        for epoch in range(self.epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            # Each stochastic pass is a "member"; their average is the MC prediction.
            for split, (sx, sy) in splits.items():
                samples = self._stochastic_passes(sx, self.samples)
                for t, probs in enumerate(samples):
                    history.append({"member": t, "epoch": epoch + 1, "split": split, **split_metrics(probs, sy)})
                history.append({"member": "ensemble", "epoch": epoch + 1, "split": split, **split_metrics(samples.mean(0), sy)})
        return history

    def sample_probabilities(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        return self._stochastic_passes(x, n_samples or self.samples)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")

    def load(self, directory: Path) -> None:
        path = directory / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Expected MC-dropout weights at {path}.")
        self.model = self._build()
        self.model.load_state_dict(torch.load(path, weights_only=True))
