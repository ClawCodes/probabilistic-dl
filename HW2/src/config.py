from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "latex" / "figs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "two_moons"
    method: str = "ensemble"
    seed: int = 7
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    members: int = 5
