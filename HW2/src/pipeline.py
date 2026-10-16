"""Single entry point: every artifact lives under outputs/run_N next to its config.json.

Steps run in order train -> plot; `--step` picks where to start.

    python -m src.pipeline                              # new run with defaults
    python -m src.pipeline --set two_moons.epochs=50    # new run with overrides
    python -m src.pipeline --dir run_1                  # retrain run_1 from its config
    python -m src.pipeline --dir run_1 --step plot      # replot run_1 from saved weights
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from .config import ExperimentConfig, next_run_dir, resolve_run_dir
from .data import DatasetBundle, load_dataset
from .methods import PredictiveModel, build_method
from .plotting import plot_2d_surface, plot_ensemble_size_sweep

STEPS = ("train", "plot")
CONFIG_FLAGS = ("config", "seed", "methods", "ensemble_sizes", "set")

# Files each step owns inside run_N/<dataset>/<method>/; a step wipes only its own.
TRAIN_ARTIFACTS = ("weights", "history.csv", "members.csv", "metrics.json")
PLOT_ARTIFACTS = ("size_sweep.csv", "figures")

Models = dict[tuple[str, str], PredictiveModel]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HW2 uncertainty pipeline.")
    parser.add_argument("--step", choices=STEPS, default="train", help="pipeline step to start from")
    parser.add_argument("--dir", help="existing run directory (e.g. run_1); its config.json is used as is")
    parser.add_argument("--config", type=Path, help="JSON config to start a new run from")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--ensemble-sizes", type=int, nargs="+")
    parser.add_argument("--set", action="append", default=[], metavar="DATASET.FIELD=VALUE", help="per-dataset override, repeatable")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    values = json.loads(args.config.read_text()) if args.config else {}
    if args.seed is not None:
        values["seed"] = args.seed
    if args.methods:
        values["methods"] = args.methods
    if args.ensemble_sizes:
        values["ensemble_sizes"] = args.ensemble_sizes
    cfg = ExperimentConfig.from_dict(values)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects DATASET.FIELD=VALUE, got '{item}'.")
        cfg = cfg.with_override(key, value)
    return cfg


def method_dir(run_dir: Path, dataset: str, method: str) -> Path:
    return run_dir / dataset / method


def clear(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = directory / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def accuracy(probabilities: torch.Tensor, y: torch.Tensor) -> float:
    return (probabilities.argmax(1) == y).float().mean().item()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(run_dir: Path, cfg: ExperimentConfig, datasets: dict[str, DatasetBundle]) -> Models:
    models: Models = {}
    for dataset_name, ds_cfg in cfg.datasets.items():
        data = datasets[dataset_name]
        for method in cfg.methods:
            seed_everything(cfg.seed)
            out = method_dir(run_dir, dataset_name, method)
            out.mkdir(parents=True, exist_ok=True)
            clear(out, TRAIN_ARTIFACTS + PLOT_ARTIFACTS)  # plot artifacts would be stale
            print(f"[train] {dataset_name}/{method}")
            model = build_method(method, data, ds_cfg, cfg)
            history, members = model.fit(data.x_train, data.y_train)
            model.save(out / "weights")
            write_csv(out / "history.csv", history)
            write_csv(out / "members.csv", members)
            train_pred, test_pred = model.predict(data.x_train), model.predict(data.x_test)
            metrics = {
                "train_accuracy": accuracy(train_pred.probabilities, data.y_train),
                "test_accuracy": accuracy(test_pred.probabilities, data.y_test),
                "test_predictive_entropy": test_pred.predictive_entropy.mean().item(),
                "test_mutual_information": test_pred.mutual_information.mean().item(),
            }
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"[train] {dataset_name}/{method}: test accuracy {metrics['test_accuracy']:.3f}")
            models[dataset_name, method] = model
    return models


def size_sweep(model: PredictiveModel, data: DatasetBundle, sizes: tuple[int, ...], members: list[dict[str, float]]) -> list[dict[str, float]]:
    rows = []
    for size in sorted(sizes):
        train_pred, test_pred = model.predict(data.x_train, n_samples=size), model.predict(data.x_test, n_samples=size)
        rows.append({"members": size, "train_accuracy": accuracy(train_pred.probabilities, data.y_train), "test_accuracy": accuracy(test_pred.probabilities, data.y_test), "test_predictive_entropy": test_pred.predictive_entropy.mean().item(), "test_mutual_information": test_pred.mutual_information.mean().item(), "mean_member_train_accuracy": sum(row["train_accuracy"] for row in members[:size]) / size})
    return rows


def plot(run_dir: Path, cfg: ExperimentConfig, datasets: dict[str, DatasetBundle], models: Models | None = None) -> None:
    models = models or {}
    for dataset_name, ds_cfg in cfg.datasets.items():
        data = datasets[dataset_name]
        for method in cfg.methods:
            out = method_dir(run_dir, dataset_name, method)
            clear(out, PLOT_ARTIFACTS)
            model = models.get((dataset_name, method))
            if model is None:
                model = build_method(method, data, ds_cfg, cfg)
                model.load(out / "weights")
            rows = size_sweep(model, data, cfg.ensemble_sizes, read_csv(out / "members.csv"))
            write_csv(out / "size_sweep.csv", rows)
            figures = [plot_ensemble_size_sweep(rows, out / "figures" / "size_sweep.png")]
            if data.feature_shape == (2,):
                figures.append(plot_2d_surface(data, model, out / "figures" / "surface.png"))
            for figure in figures:
                print(f"[plot] saved {figure.relative_to(run_dir.parent)}")


def main() -> None:
    args = parse_args()
    if args.dir:
        given = [flag for flag in CONFIG_FLAGS if getattr(args, flag)]
        if given:
            raise SystemExit(f"--dir uses the run's config.json; drop {', '.join('--' + f.replace('_', '-') for f in given)} or start a new run.")
        run_dir = resolve_run_dir(args.dir)
        cfg = ExperimentConfig.load(run_dir / "config.json")
    else:
        if args.step != STEPS[0]:
            raise SystemExit(f"--step {args.step} needs --dir pointing at an existing run.")
        cfg = build_config(args)
        run_dir = next_run_dir()
        cfg.save(run_dir / "config.json")
    print(f"Run directory: {run_dir}")

    datasets = {name: load_dataset(name, cfg.seed) for name in cfg.datasets}
    models = None
    steps = STEPS[STEPS.index(args.step):]
    if "train" in steps:
        models = train(run_dir, cfg, datasets)
    if "plot" in steps:
        plot(run_dir, cfg, datasets, models)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}")
