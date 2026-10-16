import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

KNOWN_DATASETS = ("two_moons", "fashion_mnist", "mnist")
KNOWN_METHODS = ("ensemble", "mc_dropout")


@dataclass(frozen=True)
class DatasetConfig:
    """Training hyperparameters that may differ between datasets."""
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2/Gaussian-prior precision coefficient
    hidden_dim: int = 64


def default_datasets() -> dict[str, DatasetConfig]:
    return {
        "two_moons": DatasetConfig(),
        "fashion_mnist": DatasetConfig(epochs=20, batch_size=256, hidden_dim=256),
    }


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    methods: tuple[str, ...] = ("ensemble",)
    # Members are trained once for max(ensemble_sizes); smaller sizes are prefixes.
    ensemble_sizes: tuple[int, ...] = (1, 2, 5, 10)
    datasets: dict[str, DatasetConfig] = field(default_factory=default_datasets)

    @property
    def members(self) -> int:
        return max(self.ensemble_sizes)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "methods": list(self.methods), "ensemble_sizes": list(self.ensemble_sizes),
                "datasets": {name: asdict(ds) for name, ds in self.datasets.items()}}

    @classmethod
    def from_dict(cls, values: dict) -> "ExperimentConfig":
        _check_keys(values, {f.name for f in fields(cls)}, "experiment config")
        values = dict(values)
        if "datasets" in values:
            datasets = {}
            for name, ds in values["datasets"].items():
                if name not in KNOWN_DATASETS:
                    raise ValueError(f"Unknown dataset '{name}'; expected one of {KNOWN_DATASETS}.")
                _check_keys(ds, {f.name for f in fields(DatasetConfig)}, f"datasets.{name}")
                datasets[name] = DatasetConfig(**ds)
            values["datasets"] = datasets
        for key in ("methods", "ensemble_sizes"):
            if key in values:
                values[key] = tuple(values[key])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        unknown = set(self.methods) - set(KNOWN_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods {sorted(unknown)}; expected a subset of {KNOWN_METHODS}.")
        if not self.methods or not self.datasets:
            raise ValueError("Config needs at least one method and one dataset.")
        if not self.ensemble_sizes or min(self.ensemble_sizes) < 1:
            raise ValueError("ensemble_sizes must be non-empty positive integers.")

    def with_override(self, key: str, raw_value: str) -> "ExperimentConfig":
        """Apply a `dataset.field=value` override, casting to the field's type."""
        dataset, _, name = key.partition(".")
        if dataset not in self.datasets:
            raise ValueError(f"Cannot override '{key}': dataset '{dataset}' is not in the config.")
        types = {f.name: f.type for f in fields(DatasetConfig)}
        if name not in types:
            raise ValueError(f"Cannot override '{key}': unknown field '{name}'.")
        cast = {"int": int, "float": float}.get(getattr(types[name], "__name__", types[name]), str)
        datasets = dict(self.datasets)
        datasets[dataset] = replace(datasets[dataset], **{name: cast(raw_value)})
        return replace(self, datasets=datasets)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(path.read_text()))


def _check_keys(values: dict, allowed: set[str], where: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys {sorted(unknown)} in {where}; allowed: {sorted(allowed)}.")


def resolve_run_dir(name: str | Path) -> Path:
    """Accept `run_3`, `outputs/run_3`, or an absolute path."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = OUTPUT_DIR / path
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} is not a run directory (missing config.json).")
    return path


def next_run_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    taken = [int(p.name.removeprefix("run_")) for p in OUTPUT_DIR.glob("run_*") if p.name.removeprefix("run_").isdigit()]
    path = OUTPUT_DIR / f"run_{max(taken, default=0) + 1}"
    path.mkdir()
    return path
