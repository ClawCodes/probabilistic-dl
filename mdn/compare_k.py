"""
Compare the MDN (across K) against the deterministic RegressionNet baseline
on the bimodal toy regression task from `networks._demo_general` (same
generative process: two linear modes mixed 50/50, plus small Gaussian noise).

Produces the following plots, saved directly under report/figs/:
  0. synthetic_raw_data.png - scatter of the raw training data, x vs each output
     dimension, no model overlay.
  1. synthetic_training.png - training NLL vs epoch (MDN only, one curve
     per K - RegressionNet's MSE isn't on a comparable scale so it's left
     off this panel) and training RMSE vs epoch (MDN curves plus the
     RegressionNet baseline - RMSE is the fair cross-model comparison).
  2. synthetic_holdout_vs_k.png - held-out RMSE vs K, same comparison.
  3. density_grid_2d.png - grid of p(y|x) plots (rows: K, columns: x in
     X_SLICES) showing each weighted component density and their sum.
  4. synthetic_curve_overlay.png - RegressionNet's predicted curve vs each
     MDN's predicted component mean +/- std curves, swept over x, overlaid
     on the raw training scatter. One row per K.

Model-level statistics and per-epoch training histories are saved under
report/csvs/ and consumed directly by the LaTeX report.

MDN point predictions (for accuracy/R^2) use the mixture mean, i.e. the
alpha-weighted average of the K component means.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from mdn.networks import MDN, RegressionNet, mdn_nll
from mdn.reporting import write_latex_table

K_VALUES = [1, 2, 3, 4, 5]
X_SLICES = [-2.0, 0.0, 2.0]
EPOCHS = 300
N_TRAIN, N_HOLDOUT = 4000, 1000
IN_DIM, OUT_DIM = 1, 1
HIDDEN, RANK, LR = 128, 1, 1e-3
BATCH_SIZE = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figs"
CSV_DIR = PROJECT_ROOT / "report" / "csvs"


def make_bimodal_data(n, in_dim, out_dim, W1, W2, generator):
    """Same generative process as networks._demo_general: y is drawn from one
    of two linear modes of x (50/50), plus small Gaussian observation noise."""
    x = torch.randn(n, in_dim, generator=generator)
    mode = (torch.rand(n, generator=generator) < 0.5).float().unsqueeze(-1)
    noise = 0.1 * torch.randn(n, out_dim, generator=generator)
    y = mode * (x @ W1) + (1 - mode) * (x @ W2) + noise
    return x, y


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def r_squared(pred, target):
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot).mean().item()


def mdn_mixture_mean(alpha, mu):
    return (alpha.unsqueeze(-1) * mu).sum(dim=1)


def train_mdn(K, x_train, y_train):
    """Minibatch training (BATCH_SIZE, reshuffled every epoch). The logged
    per-epoch loss/accuracy is evaluated once on the full training set after
    that epoch's minibatch updates, not the last minibatch's loss - so
    'epoch' still means one point per curve, same as before batching."""
    torch.manual_seed(0)
    model = MDN(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN, K=K, rank=RANK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    n = x_train.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            alpha, mu, Lambda = model(x_train[idx])
            loss = mdn_nll(alpha, mu, Lambda, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            alpha, mu, Lambda = model(x_train)
            epoch_loss = mdn_nll(alpha, mu, Lambda, y_train).item()
            epoch_rmse = rmse(mdn_mixture_mean(alpha, mu), y_train)
        loss_curve.append(epoch_loss)
        rmse_curve.append(epoch_rmse)
    return model, loss_curve, rmse_curve


def train_regression(x_train, y_train):
    """Minibatch training, same epoch-level logging convention as train_mdn."""
    torch.manual_seed(0)
    model = RegressionNet(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    n = x_train.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            pred = model(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = model(x_train)
            epoch_loss = F.mse_loss(pred, y_train).item()
            epoch_rmse = rmse(pred, y_train)
        loss_curve.append(epoch_loss)
        rmse_curve.append(epoch_rmse)
    return model, loss_curve, rmse_curve


def evaluate_mdn(model, x, y):
    with torch.no_grad():
        alpha, mu, Lambda = model(x)
        pred = mdn_mixture_mean(alpha, mu)
        return {
            "nll": mdn_nll(alpha, mu, Lambda, y).item(),
            "rmse": rmse(pred, y),
            "r2": r_squared(pred, y),
        }


def evaluate_regression(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return {
            "mse": F.mse_loss(pred, y).item(),
            "rmse": rmse(pred, y),
            "r2": r_squared(pred, y),
        }


def metric(value):
    """Format a metric consistently for both CSV output and LaTeX display."""
    return "N/A" if value is None else f"{value:.6f}"


def plot_raw_data(x_data, y_data):
    """Scatter x vs each output dimension of y, no model overlay."""
    out_dim = y_data.shape[1]
    fig, axes = plt.subplots(out_dim, 1, figsize=(8, 2.5 * out_dim), sharex=True)
    axes = [axes] if out_dim == 1 else axes
    for j, ax in enumerate(axes):
        ax.scatter(x_data.squeeze(-1).numpy(), y_data[:, j].numpy(), s=6, alpha=0.3, color="gray")
        ax.set_ylabel(f"y[{j}]")
    axes[-1].set_xlabel("x")
    fig.suptitle("Raw training data")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_raw_data.png", dpi=150)
    plt.close(fig)


def univariate_density(y_grid, mu, std):
    return torch.exp(-0.5 * ((y_grid - mu) / std) ** 2) / (std * (2 * torch.pi) ** 0.5)


def plot_density_grid(models_by_k, x_slices, n_grid=300):
    """Grid of p(y|x) plots: one row per K, one column per x in x_slices.
    Each subplot shows the K weighted component densities plus their sum
    (the mixture density), as a function of y."""
    k_values = sorted(models_by_k)
    fig, axes = plt.subplots(len(k_values), len(x_slices),
                              figsize=(4 * len(x_slices), 2.5 * len(k_values)),
                              squeeze=False)
    colors = plt.cm.tab10.colors

    for row, K in enumerate(k_values):
        model = models_by_k[K]
        for col, x_val in enumerate(x_slices):
            ax = axes[row][col]
            x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
            with torch.no_grad():
                alpha, mu, Lambda = model(x_tensor)
            alpha_k = alpha[0]                          # (K,)
            mu_k = mu[0, :, 0]                           # (K,)
            std_k = Lambda[0, :, 0, 0].reciprocal().sqrt()  # (K,)

            y_min = (mu_k - 4 * std_k).min().item()
            y_max = (mu_k + 4 * std_k).max().item()
            y_grid = torch.linspace(y_min, y_max, n_grid)

            mixture_density = torch.zeros(n_grid)
            for k in range(K):
                weighted = alpha_k[k] * univariate_density(y_grid, mu_k[k], std_k[k])
                mixture_density += weighted
                ax.plot(y_grid.numpy(), weighted.numpy(), color=colors[k % len(colors)],
                        linewidth=1.2, label=f"component {k}" if col == 0 else None)
            ax.plot(y_grid.numpy(), mixture_density.numpy(), color="black", linewidth=1.8,
                    linestyle="--", label="mixture" if col == 0 else None)

            ax.set_title(f"K={K}, x={x_val:g}", fontsize=10)
            if row == len(k_values) - 1:
                ax.set_xlabel("y")
            if col == 0:
                ax.set_ylabel("p(y|x)")
        axes[row][0].legend(fontsize=7, loc="upper right")

    fig.suptitle("Predicted mixture density p(y|x) across K and x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "density_grid_2d.png", dpi=150)
    plt.close(fig)


def plot_curve_overlay(models_by_k, reg_model, x_data, y_data, n_grid=200):
    """Sweep x over its observed range and overlay:
      - RegressionNet's single predicted curve (black dashed)
      - each MDN's K component mean +/- 1 std curves (colored)
    on top of a scatter of the real (x, y) data. One row per K."""
    k_values = sorted(models_by_k)
    x_flat, y_flat = x_data[:, 0], y_data[:, 0]
    x_grid = torch.linspace(x_flat.min().item(), x_flat.max().item(), n_grid).unsqueeze(-1)

    with torch.no_grad():
        reg_pred = reg_model(x_grid)  # (n_grid, 1)

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(k_values), 1, figsize=(8, 3.5 * len(k_values)), squeeze=False)
    axes = [ax[0] for ax in axes]

    # Fix the y-range to the real data span (with padding) rather than
    # autoscaling: a spare/near-degenerate component can have a wildly
    # inflated predicted std in regions it doesn't own, which would
    # otherwise blow out the axis and hide the actual fit.
    y_min, y_max = y_flat.min().item(), y_flat.max().item()
    padding = 0.1 * (y_max - y_min)
    y_lo, y_hi = y_min - padding, y_max + padding

    for row, K in enumerate(k_values):
        ax = axes[row]
        ax.scatter(x_flat.numpy(), y_flat.numpy(), s=6, alpha=0.15, color="gray",
                   label="data" if row == 0 else None)
        ax.plot(x_grid[:, 0].numpy(), reg_pred[:, 0].numpy(), color="black",
                linewidth=1.8, linestyle="--", label="RegressionNet" if row == 0 else None)

        with torch.no_grad():
            alpha, mu, Lambda = models_by_k[K](x_grid)          # (n_grid,K), (n_grid,K,1), (n_grid,K,1,1)
        mu_k = mu[:, :, 0]                                       # (n_grid, K)
        std_k = Lambda[:, :, 0, 0].reciprocal().sqrt()           # (n_grid, K)

        for k in range(K):
            color = colors[k % len(colors)]
            mean_k, std_k_np = mu_k[:, k].numpy(), std_k[:, k].numpy()
            avg_weight = alpha[:, k].mean().item()
            ax.plot(x_grid[:, 0].numpy(), mean_k, color=color, linewidth=1.5,
                    label=f"MDN component {k} (avg weight={avg_weight:.2f})" if row == 0 else None)
            ax.fill_between(x_grid[:, 0].numpy(), mean_k - std_k_np, mean_k + std_k_np,
                            color=color, alpha=0.15)

        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("y")
        ax.set_title(f"K={K}", fontsize=10)
    axes[-1].set_xlabel("x")
    axes[0].legend(fontsize=7, loc="upper left")

    fig.suptitle("RegressionNet vs MDN predicted components")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_curve_overlay.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    W1, W2 = torch.randn(IN_DIM, OUT_DIM), torch.randn(IN_DIM, OUT_DIM)
    x_train, y_train = make_bimodal_data(N_TRAIN, IN_DIM, OUT_DIM, W1, W2,
                                          generator=torch.Generator().manual_seed(1))
    x_holdout, y_holdout = make_bimodal_data(N_HOLDOUT, IN_DIM, OUT_DIM, W1, W2,
                                              generator=torch.Generator().manual_seed(2))

    plot_raw_data(x_train, y_train)

    reg_model, reg_loss_curve, reg_rmse_curve = train_regression(x_train, y_train)
    reg_train = evaluate_regression(reg_model, x_train, y_train)
    reg_test = evaluate_regression(reg_model, x_holdout, y_holdout)
    print(f"RegressionNet  holdout R^2={reg_test['r2']:.4f}  "
          f"holdout RMSE={reg_test['rmse']:.4f}")

    mdn_results = {}
    for K in K_VALUES:
        model, loss_curve, rmse_curve = train_mdn(K, x_train, y_train)
        train_metrics = evaluate_mdn(model, x_train, y_train)
        test_metrics = evaluate_mdn(model, x_holdout, y_holdout)
        mdn_results[K] = dict(
            model=model,
            loss_curve=loss_curve,
            rmse_curve=rmse_curve,
            train=train_metrics,
            test=test_metrics,
        )
        print(f"MDN K={K}  holdout R^2={test_metrics['r2']:.4f}  "
              f"holdout RMSE={test_metrics['rmse']:.4f}")

    summary_rows = [{
        "model": "Deterministic",
        "k": "---",
        "train_nll": "N/A",
        "test_nll": "N/A",
        "train_rmse": metric(reg_train["rmse"]),
        "test_rmse": metric(reg_test["rmse"]),
        "train_r2": metric(reg_train["r2"]),
        "test_r2": metric(reg_test["r2"]),
        "train_mse": metric(reg_train["mse"]),
        "test_mse": metric(reg_test["mse"]),
    }]
    for K in K_VALUES:
        train_metrics = mdn_results[K]["train"]
        test_metrics = mdn_results[K]["test"]
        summary_rows.append({
            "model": "Single Gaussian" if K == 1 else "MDN",
            "k": str(K),
            "train_nll": metric(train_metrics["nll"]),
            "test_nll": metric(test_metrics["nll"]),
            "train_rmse": metric(train_metrics["rmse"]),
            "test_rmse": metric(test_metrics["rmse"]),
            "train_r2": metric(train_metrics["r2"]),
            "test_r2": metric(test_metrics["r2"]),
            "train_mse": metric(train_metrics["rmse"] ** 2),
            "test_mse": metric(test_metrics["rmse"] ** 2),
        })
    summary_path = CSV_DIR / "synthetic_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    write_latex_table(summary_path)

    history_rows = []
    for epoch, (loss, rmse_val) in enumerate(zip(reg_loss_curve, reg_rmse_curve), start=1):
        history_rows.append({
            "model": "Deterministic", "k": "", "epoch": epoch,
            "loss_name": "MSE", "train_loss": loss, "train_rmse": rmse_val,
        })
    for K in K_VALUES:
        model_name = "Single Gaussian" if K == 1 else "MDN"
        for epoch, (loss, rmse_val) in enumerate(
                zip(mdn_results[K]["loss_curve"], mdn_results[K]["rmse_curve"]), start=1):
            history_rows.append({
                "model": model_name, "k": K, "epoch": epoch,
                "loss_name": "NLL", "train_loss": loss, "train_rmse": rmse_val,
            })
    pd.DataFrame(history_rows).to_csv(
        CSV_DIR / "synthetic_training_history.csv", index=False)

    plot_density_grid({K: mdn_results[K]["model"] for K in K_VALUES}, X_SLICES)
    plot_curve_overlay({K: mdn_results[K]["model"] for K in K_VALUES}, reg_model, x_train, y_train)

    epochs_axis = range(EPOCHS)

    # RegressionNet is deliberately left off this panel: its MSE and the
    # MDN's NLL are different loss functions on different scales (NLL can be
    # negative, MSE can't), so overlaying them on one axis would invite a
    # cross-model "lower curve wins" reading that isn't valid. The RMSE
    # panel below is the fair cross-model comparison (same units for every
    # model regardless of training objective); this panel is only for
    # checking each MDN's own NLL is converging.
    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(12, 5))
    for K in K_VALUES:
        ax_loss.plot(epochs_axis, mdn_results[K]["loss_curve"], label=f"MDN K={K}")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training NLL")
    ax_loss.set_title("Training NLL vs epoch")
    ax_loss.legend()

    for K in K_VALUES:
        ax_rmse.plot(epochs_axis, mdn_results[K]["rmse_curve"], label=f"MDN K={K}")
    ax_rmse.plot(epochs_axis, reg_rmse_curve, label="RegressionNet", color="black", linestyle="--")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.set_ylabel("training RMSE")
    ax_rmse.set_title("Training RMSE vs epoch")
    ax_rmse.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_training.png", dpi=150)
    plt.close(fig)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.plot(K_VALUES, [mdn_results[K]["test"]["rmse"] for K in K_VALUES],
             marker="o", label="MDN")
    ax3.axhline(reg_test["rmse"], color="black", linestyle="--", label="RegressionNet")
    ax3.set_xlabel("K (number of mixture components)")
    ax3.set_ylabel("held-out RMSE")
    ax3.set_title("Held-out RMSE vs K")
    ax3.set_xticks(K_VALUES)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "synthetic_holdout_vs_k.png", dpi=150)
    plt.close(fig3)

    print(f"Plots saved to {FIG_DIR}")
    print(f"Statistics saved to {CSV_DIR}")


if __name__ == "__main__":
    main()
