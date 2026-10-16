"""Run the required comparison across ensemble sizes with a shared config."""
import argparse
import csv
from dataclasses import replace
from .config import ExperimentConfig, OUTPUT_DIR
from .data import load_dataset
from .methods import DeepEnsemble
from .config import FIG_DIR
from .plotting import plot_ensemble_size_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="two_moons")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    cfg = ExperimentConfig(dataset=args.dataset, epochs=args.epochs)
    data = load_dataset(cfg.dataset, cfg.seed)
    rows = []
    for size in args.sizes:
        run = replace(cfg, members=size)
        model = DeepEnsemble(data.feature_shape, data.n_classes, members=run.members, hidden_dim=run.hidden_dim, epochs=run.epochs, batch_size=run.batch_size, learning_rate=run.learning_rate, weight_decay=run.weight_decay, seed=run.seed)
        history = model.fit(data.x_train, data.y_train)
        train_prediction, test_prediction = model.predict(data.x_train), model.predict(data.x_test)
        rows.append({"members": size, "train_accuracy": (train_prediction.probabilities.argmax(1) == data.y_train).float().mean().item(), "test_accuracy": (test_prediction.probabilities.argmax(1) == data.y_test).float().mean().item(), "test_predictive_entropy": test_prediction.predictive_entropy.mean().item(), "test_mutual_information": test_prediction.mutual_information.mean().item(), "mean_member_train_accuracy": sum(item["train_accuracy"] for item in history) / size})
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"ensemble_size_sweep_{args.dataset}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    figure = plot_ensemble_size_sweep(rows, FIG_DIR, args.dataset)
    print(f"Saved: {path}")
    print(f"Saved figure: {figure}")


if __name__ == "__main__":
    main()
