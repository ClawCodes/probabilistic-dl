import argparse
import csv
import json
import random
import numpy as np
import torch
from .config import ExperimentConfig, FIG_DIR, OUTPUT_DIR
from .data import load_dataset
from .methods import DeepEnsemble
from .plotting import plot_2d_surface


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run HW2 uncertainty experiments.")
    parser.add_argument("--dataset", default="two_moons")
    parser.add_argument("--method", choices=["ensemble", "mc_dropout"], default="ensemble")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2/Gaussian-prior precision coefficient")
    values = parser.parse_args()
    return ExperimentConfig(**vars(values))


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if cfg.method == "mc_dropout":
        from .methods.mc_dropout import MCDropoutPredictor
        MCDropoutPredictor()  # Produces a deliberate, informative extension-point error.
        return
    data = load_dataset(cfg.dataset, cfg.seed)
    model = DeepEnsemble(data.feature_shape, data.n_classes, members=cfg.members, hidden_dim=cfg.hidden_dim, epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, seed=cfg.seed)
    history = model.fit(data.x_train, data.y_train)
    prediction = model.predict(data.x_test)
    accuracy = (prediction.probabilities.argmax(1) == data.y_test).float().mean().item()
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / f"{cfg.method}_{cfg.dataset}_metrics.json").open("w") as handle:
        json.dump({"test_accuracy": accuracy, "config": vars(cfg)}, handle, indent=2)
    with (OUTPUT_DIR / f"{cfg.method}_{cfg.dataset}_members.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    if data.feature_shape == (2,):
        figure = plot_2d_surface(data, model, FIG_DIR, cfg.method)
        print(f"Saved figure: {figure}")
    print(f"Test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
