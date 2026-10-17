"""Single entry point: every artifact lives under outputs/run_N next to its config.json.

Steps run in order train -> plot; `--step` picks where to start.

    python -m src.pipeline                              # new run with defaults
    python -m src.pipeline --set two_moons.epochs=50    # new run with overrides
    python -m src.pipeline --dir run_1                  # retrain run_1 from its config
    python -m src.pipeline --dir run_1 --step plot      # replot run_1 from saved weights
"""
import argparse
import csv
import itertools
import math
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from .config import ExperimentConfig, next_run_dir, resolve_run_dir
from .data import DatasetBundle, load_dataset
from .methods import PredictiveModel, build_method, prediction_from_samples
from .metrics import accuracy, ece, error_rate, nll, reliability_bins
from .plotting import ENSEMBLE_COLOR, MEMBER_COLOR, SWEEP_PANELS, plot_2d_surface, plot_ensemble_size_sweep, plot_reliability, plot_training_curves

STEPS = ("train", "plot")
CONFIG_FLAGS = ("config", "seed", "methods", "ensemble_sizes", "set")

# Files each step owns inside run_N/<dataset>/<method>/; a step wipes only its own.
TRAIN_ARTIFACTS = ("weights", "history.csv", "metrics.json")
PLOT_ARTIFACTS = ("size_sweep.csv", "size_sweep_summary.csv", "figures")
STRING_COLUMNS = ("member", "split")

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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [{key: value if key in STRING_COLUMNS else float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def sample_metrics(samples: torch.Tensor, y: torch.Tensor, ece_bins: int) -> dict[str, float]:
    """Error, NLL, ECE and mean uncertainty of the averaged prediction over `samples` (S, N, C)."""
    prediction = prediction_from_samples(samples)
    probs = prediction.probabilities
    return {"accuracy": accuracy(probs, y), "error": error_rate(probs, y), "nll": nll(probs, y), "ece": ece(probs, y, ece_bins),
            "entropy": prediction.predictive_entropy.mean().item(), "mi": prediction.mutual_information.mean().item()}


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
            history = model.fit(data.x_train, data.y_train, data.x_test, data.y_test)
            model.save(out / "weights")
            write_csv(out / "history.csv", history)
            metrics = {}
            for split, (x, y) in (("train", (data.x_train, data.y_train)), ("test", (data.x_test, data.y_test))):
                for name, value in sample_metrics(model.sample_probabilities(x), y, cfg.ece_bins).items():
                    metrics[f"{split}_{name}"] = value
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"[train] {dataset_name}/{method}: test accuracy {metrics['test_accuracy']:.3f}")
            models[dataset_name, method] = model
    return models


def member_subsets(n_members: int, size: int, draws: int, generator: torch.Generator) -> list[tuple[int, ...]]:
    """k-member subsets for error bars: every member at k=1, every combination when
    there are at most `draws` of them (so k=M is one draw), else `draws` random ones."""
    if size == 1:
        return [(member,) for member in range(n_members)]
    if math.comb(n_members, size) <= draws:
        return list(itertools.combinations(range(n_members), size))
    return [tuple(sorted(torch.randperm(n_members, generator=generator)[:size].tolist())) for _ in range(draws)]


def size_sweep(samples: dict[str, tuple[torch.Tensor, torch.Tensor]], cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """Per-subset rows and their mean/std summary for every ensemble size and split."""
    generator = torch.Generator().manual_seed(cfg.seed)
    rows, summary = [], []
    for size in sorted(cfg.ensemble_sizes):
        subsets = member_subsets(cfg.members, size, cfg.subset_draws, generator)
        for split, (split_samples, y) in samples.items():
            split_rows = [{"members": size, "draw": draw, "split": split, "subset": " ".join(map(str, subset)), **sample_metrics(split_samples[list(subset)], y, cfg.ece_bins)}
                          for draw, subset in enumerate(subsets)]
            rows.extend(split_rows)
            entry = {"members": size, "split": split, "draws": len(subsets)}
            for metric in [metric for metric, _ in SWEEP_PANELS] + ["accuracy"]:
                values = torch.tensor([row[metric] for row in split_rows])
                entry[f"{metric}_mean"], entry[f"{metric}_std"] = values.mean().item(), values.std(unbiased=False).item()
            summary.append(entry)
    return rows, summary


def plot(run_dir: Path, cfg: ExperimentConfig, datasets: dict[str, DatasetBundle], models: Models | None = None) -> None:
    models = models or {}
    for dataset_name, ds_cfg in cfg.datasets.items():
        data = datasets[dataset_name]
        for method in cfg.methods:
            out = method_dir(run_dir, dataset_name, method)
            history = read_csv(out / "history.csv")
            if not history or "split" not in history[0]:
                raise ValueError(f"{out / 'history.csv'} uses the old schema; retrain with --dir {run_dir.name}.")
            clear(out, PLOT_ARTIFACTS)
            model = models.get((dataset_name, method))
            if model is None:
                model = build_method(method, data, ds_cfg, cfg)
                model.load(out / "weights")
            # Run the networks once per split; subsets are then just indexing.
            samples = {"train": (model.sample_probabilities(data.x_train, cfg.members), data.y_train),
                       "test": (model.sample_probabilities(data.x_test, cfg.members), data.y_test)}
            rows, summary = size_sweep(samples, cfg)
            write_csv(out / "size_sweep.csv", rows)
            write_csv(out / "size_sweep_summary.csv", summary)
            test_samples, y_test = samples["test"]
            reliability = {
                "single member": {"probs": test_samples[0], "color": MEMBER_COLOR},
                f"ensemble (k={cfg.members})": {"probs": test_samples.mean(0), "color": ENSEMBLE_COLOR},
            }
            for curve in reliability.values():
                curve["bins"], curve["ece"] = reliability_bins(curve["probs"], y_test, cfg.ece_bins), ece(curve["probs"], y_test, cfg.ece_bins)
            figures = [
                plot_ensemble_size_sweep(summary, out / "figures" / "size_sweep.png"),
                plot_training_curves(history, out / "figures" / "training_curves.png"),
                plot_reliability(reliability, out / "figures" / "reliability.png"),
            ]
            if data.feature_shape == (2,): #TODO: can make apply to just two moons will also have fashion-MNIST as well
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
