import torch
from torch import nn


class ClassifierMLP(nn.Module):
    """Baseline classifier. `dropout` allows direct reuse by MC-dropout later."""
    def __init__(self, input_shape: tuple[int, ...], n_classes: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        input_dim = int(torch.tensor(input_shape).prod().item())
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
