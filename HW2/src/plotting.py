from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from .data import DatasetBundle

# Categorical slots, validated together (all-pairs CVD/normal-vision pass).
# Aqua is below 3:1 contrast, so every series here also has a legend and its own marker.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8a8984"
TEXT_SECONDARY = "#52514e"
# Fixed entity colors, used the same way in every figure.
MEMBER_COLOR, ENSEMBLE_COLOR = BLUE, ORANGE
SPLIT_STYLE = {"test": {"color": BLUE, "marker": "o"}, "train": {"color": AQUA, "marker": "s"}}
METHOD_STYLE = {"ensemble": {"color": BLUE, "marker": "o", "label": "deep ensemble"},
                "mc_dropout": {"color": ORANGE, "marker": "s", "label": "MC dropout"}}


def _style(ax) -> None:
    """Recessive axes: no top/right spines, light grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


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


SWEEP_PANELS = (("error", "error rate"), ("nll", "NLL (nats)"), ("ece", "ECE"), ("entropy", "predictive entropy (nats)"), ("mi", "mutual information (nats)"))


def plot_ensemble_size_sweep(summary: list[dict], path: Path) -> Path:
    """Metrics vs. number of members, with +/-1 sd over k-member subsets.

    `summary` rows: members, split, and <metric>_mean / <metric>_std for each panel.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(row["members"]) for row in summary})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    flat = axes.ravel()
    for ax, (metric, label) in zip(flat, SWEEP_PANELS):
        _style(ax)
        single = [row for row in summary if row["split"] == "test" and int(row["members"]) == 1]
        if metric in ("error", "nll") and single:
            # Reference band: the spread of individual members on the test set (k=1).
            mean, std = single[0][f"{metric}_mean"], single[0][f"{metric}_std"]
            ax.axhspan(mean - std, mean + std, color=NEUTRAL, alpha=0.15, linewidth=0, label="single member, test (±1 sd)")
        for split in ("train", "test"):
            rows = sorted((row for row in summary if row["split"] == split), key=lambda row: row["members"])
            ax.errorbar([row["members"] for row in rows], [row[f"{metric}_mean"] for row in rows], yerr=[row[f"{metric}_std"] for row in rows],
                        linewidth=2, markersize=6, capsize=3, label=split, **SPLIT_STYLE[split])
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.minorticks_off()
        ax.set_xlabel("number of members", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(label, fontsize=10, loc="left")
    # The sixth cell holds the shared legend.
    flat[-1].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[-1].legend(handles, labels, loc="center", frameon=False, title="error bars: ±1 sd over\nk-member subsets", fontsize=9, title_fontsize=9)
    fig.suptitle("Performance and uncertainty vs. ensemble size", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(history: list[dict], path: Path) -> Path:
    """Loss and accuracy per epoch on train and test: member mean +/-1 sd and the ensemble."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted({int(row["epoch"]) for row in history})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for col, split in enumerate(("train", "test")):
        rows = [row for row in history if row["split"] == split]
        for row_index, (metric, label) in enumerate((("loss", "NLL (nats)"), ("accuracy", "accuracy"))):
            ax = axes[row_index, col]
            _style(ax)
            members = torch.tensor([[row[metric] for row in rows if row["member"] != "ensemble" and int(row["epoch"]) == epoch] for epoch in epochs])
            mean, std = members.mean(1), members.std(1, unbiased=False)
            ensemble = [next(row[metric] for row in rows if row["member"] == "ensemble" and int(row["epoch"]) == epoch) for epoch in epochs]
            ax.fill_between(epochs, mean - std, mean + std, color=MEMBER_COLOR, alpha=0.2, linewidth=0, label="members ±1 sd")
            ax.plot(epochs, mean, color=MEMBER_COLOR, linewidth=2, label=f"member mean (n={members.shape[1]})")
            ax.plot(epochs, ensemble, color=ENSEMBLE_COLOR, linewidth=2, label="ensemble")
            ax.set_title(f"{split} {label}", fontsize=10, loc="left")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if row_index == 1:
                ax.set_xlabel("epoch", fontsize=9, color=TEXT_SECONDARY)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training curves", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reliability(curves: dict[str, dict], path: Path, min_count: int = 10) -> Path:
    """Reliability diagram (top) and confidence histogram (bottom) on the test set.

    `curves` maps a label to {"bins": reliability_bins(...) output, "ece": float, "color": hex}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.2, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    for ax in (top, bottom):
        _style(ax)
    top.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1, label="perfect calibration")
    n_curves = len(curves)
    for index, (label, curve) in enumerate(curves.items()):
        bins = curve["bins"]
        # Nearly empty bins give noisy accuracies; the histogram still shows them.
        filled = bins["count"] >= min_count
        top.plot(bins["confidence"][filled], bins["accuracy"][filled], color=curve["color"], linewidth=2, marker="o", markersize=6, label=f"{label} (ECE {curve['ece']:.3f})")
        n_bins = len(bins["count"])
        width = 1 / n_bins / n_curves
        lefts = torch.arange(n_bins) / n_bins + index * width
        bottom.bar(lefts, bins["count"] / bins["count"].sum(), width=width * 0.9, align="edge", color=curve["color"])
    top.set(xlim=(0, 1), ylim=(0, 1))
    top.set_ylabel("accuracy", fontsize=9, color=TEXT_SECONDARY)
    top.set_title(f"Test reliability (bins with ≥{min_count} samples)", fontsize=10, loc="left")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.set_ylabel("fraction", fontsize=9, color=TEXT_SECONDARY)
    bottom.set_xlabel("confidence (max probability)", fontsize=9, color=TEXT_SECONDARY)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_function_space(points: dict[str, torch.Tensor], explained: torch.Tensor, disagreement: dict[str, torch.Tensor], errors: dict[str, float], path: Path) -> Path:
    """Members of every method in one shared PCA of their test predictions (left), and
    each method's pairwise top-1 disagreement on a shared color scale (right).

    `points` maps method -> (S, 2) PCA coordinates, `explained` is the variance ratio of
    each component, `disagreement` maps method -> (S, S), `errors` maps method -> test error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(points)
    fig, axes = plt.subplots(1, 1 + len(methods), figsize=(4.6 + 3.4 * len(methods), 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35] + [1] * len(methods)})
    scatter = axes[0]
    _style(scatter)
    for method, coords in points.items():
        style = METHOD_STYLE.get(method, {"color": NEUTRAL, "marker": "o", "label": method})
        scatter.scatter(coords[:, 0], coords[:, 1], s=48, color=style["color"], marker=style["marker"], edgecolor="white", linewidth=1,
                        label=f"{style['label']} members", zorder=3)
        # Averaging probabilities is linear, so the method's prediction sits at its members' centroid.
        center = coords.mean(0)
        scatter.scatter(center[0], center[1], s=110, facecolor="none", edgecolor=style["color"], marker=style["marker"], linewidth=2,
                        label=f"{style['label']} average", zorder=4)
    scatter.set_aspect("equal", adjustable="datalim")
    scatter.set_xlabel(f"PC 1 ({explained[0]:.0%} of variance)", fontsize=9, color=TEXT_SECONDARY)
    scatter.set_ylabel(f"PC 2 ({explained[1]:.0%} of variance)", fontsize=9, color=TEXT_SECONDARY)
    scatter.set_title("Members in function space (test predictions)", fontsize=10, loc="left")
    # Below the axes, so it never covers a member or centroid.
    scatter.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    vmax = max(matrix.max().item() for matrix in disagreement.values())
    for ax, (method, matrix) in zip(axes[1:], disagreement.items()):
        n = matrix.shape[0]
        off_diagonal = ~torch.eye(n, dtype=torch.bool)
        shown = matrix.clone()
        shown[~off_diagonal] = float("nan")  # self-disagreement is always 0
        image = ax.imshow(shown.numpy(), cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("member", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("member", fontsize=9, color=TEXT_SECONDARY)
        label = METHOD_STYLE.get(method, {"label": method})["label"]
        ax.set_title(f"{label}: mean disagreement {matrix[off_diagonal].mean():.3f}\n(averaged-model test error {errors[method]:.3f})", fontsize=9, loc="left")
    fig.colorbar(image, ax=axes[1:].tolist(), shrink=0.8, label="fraction of test inputs with different top-1 class")
    fig.suptitle("Functional diversity: how differently do members predict?", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
