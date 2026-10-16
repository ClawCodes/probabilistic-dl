from pathlib import Path
import matplotlib.pyplot as plt
import torch
from .data import DatasetBundle
from .methods.base import Prediction


def plot_2d_surface(dataset: DatasetBundle, predictor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max = dataset.x_train[:, 0].min().item() - .7, dataset.x_train[:, 0].max().item() + .7
    y_min, y_max = dataset.x_train[:, 1].min().item() - .7, dataset.x_train[:, 1].max().item() + .7
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 250), torch.linspace(y_min, y_max, 250), indexing="xy")
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    pred = predictor.predict(grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].contourf(xx.numpy(), yy.numpy(), pred.probabilities[:, 1].reshape(xx.shape).numpy(), levels=30, cmap="RdBu", alpha=.8)
    axes[0].scatter(dataset.x_train[:, 0], dataset.x_train[:, 1], c=dataset.y_train, cmap="RdBu", edgecolor="white", s=18)
    axes[0].set(title="Predictive probability", xlabel="$x_1$", ylabel="$x_2$")
    image = axes[1].contourf(xx.numpy(), yy.numpy(), pred.mutual_information.reshape(xx.shape).numpy(), levels=30, cmap="magma")
    axes[1].scatter(dataset.x_train[:, 0], dataset.x_train[:, 1], c=dataset.y_train, cmap="RdBu", edgecolor="white", s=18)
    axes[1].set(title="Epistemic uncertainty (MI)", xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(image, ax=axes[1], label="nats")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ensemble_size_sweep(rows: list[dict[str, float]], path: Path) -> Path:
    """Plot the required ensemble-size accuracy/uncertainty comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = [row["members"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(sizes, [row["train_accuracy"] for row in rows], "o-", label="train")
    axes[0].plot(sizes, [row["test_accuracy"] for row in rows], "o-", label="test")
    axes[0].set(xlabel="number of members", ylabel="accuracy", title="Performance vs. ensemble size")
    axes[0].legend()
    axes[1].plot(sizes, [row["test_predictive_entropy"] for row in rows], "o-", label="predictive entropy")
    axes[1].plot(sizes, [row["test_mutual_information"] for row in rows], "o-", label="mutual information")
    axes[1].set(xlabel="number of members", ylabel="nats", title="Test uncertainty vs. ensemble size")
    axes[1].legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
