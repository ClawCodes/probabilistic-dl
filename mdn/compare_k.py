"""
Compare the MDN (across K) against the deterministic RegressionNet baseline
on the bimodal toy regression task from `networks._demo_general` (same
generative process: two linear modes mixed 50/50, plus small Gaussian noise).

Produces the following plots, saved under mdn/plots/:
  0. raw_data.png - scatter of the raw training data, x vs each output
     dimension, no model overlay.
  1. loss_accuracy_vs_epoch.png - training loss (MDN: NLL, Regression: MSE)
     and training accuracy (negative RMSE) vs epoch, one curve per K plus the
     RegressionNet baseline.
  2. holdout_accuracy_vs_k.png - held-out accuracy (negative RMSE) vs K,
     same comparison.
  3. density_grid.png - grid of p(y|x) plots (rows: K, columns: x in
     X_SLICES) showing each weighted component density and their sum.

MDN point predictions (for accuracy/R^2) use the mixture mean, i.e. the
alpha-weighted average of the K component means.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mdn.networks import MDN, RegressionNet, mdn_nll

K_VALUES = [1, 2, 3, 4, 5]
X_SLICES = [-2.0, 0.0, 2.0]
EPOCHS = 1000
N_TRAIN, N_HOLDOUT = 4000, 1000
IN_DIM, OUT_DIM = 1, 1
HIDDEN, RANK, LR = 128, 1, 1e-3
PLOT_DIR = "mdn/plots"


def make_bimodal_data(n, in_dim, out_dim, W1, W2, generator):
    """Same generative process as networks._demo_general: y is drawn from one
    of two linear modes of x (50/50), plus small Gaussian observation noise."""
    x = torch.randn(n, in_dim, generator=generator)
    mode = (torch.rand(n, generator=generator) < 0.5).float().unsqueeze(-1)
    noise = 0.1 * torch.randn(n, out_dim, generator=generator)
    y = mode * (x @ W1) + (1 - mode) * (x @ W2) + noise
    return x, y


def neg_rmse(pred, target):
    return -torch.sqrt(F.mse_loss(pred, target)).item()


def r_squared(pred, target):
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot).mean().item()


def mdn_mixture_mean(alpha, mu):
    return (alpha.unsqueeze(-1) * mu).sum(dim=1)


def train_mdn(K, x_train, y_train):
    torch.manual_seed(0)
    model = MDN(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN, K=K, rank=RANK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, acc_curve = [], []
    for _ in range(EPOCHS):
        alpha, mu, Lambda = model(x_train)
        loss = mdn_nll(alpha, mu, Lambda, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = neg_rmse(mdn_mixture_mean(alpha, mu), y_train)
        loss_curve.append(loss.item())
        acc_curve.append(acc)
    return model, loss_curve, acc_curve


def train_regression(x_train, y_train):
    torch.manual_seed(0)
    model = RegressionNet(in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, acc_curve = [], []
    for _ in range(EPOCHS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = neg_rmse(pred, y_train)
        loss_curve.append(loss.item())
        acc_curve.append(acc)
    return model, loss_curve, acc_curve


def evaluate_mdn(model, x, y):
    with torch.no_grad():
        alpha, mu, _ = model(x)
        pred = mdn_mixture_mean(alpha, mu)
        return r_squared(pred, y), neg_rmse(pred, y)


def evaluate_regression(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return r_squared(pred, y), neg_rmse(pred, y)


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
    fig.savefig(f"{PLOT_DIR}/raw_data.png", dpi=150)
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
    fig.savefig(f"{PLOT_DIR}/density_grid.png", dpi=150)
    plt.close(fig)


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    torch.manual_seed(0)
    W1, W2 = torch.randn(IN_DIM, OUT_DIM), torch.randn(IN_DIM, OUT_DIM)
    x_train, y_train = make_bimodal_data(N_TRAIN, IN_DIM, OUT_DIM, W1, W2,
                                          generator=torch.Generator().manual_seed(1))
    x_holdout, y_holdout = make_bimodal_data(N_HOLDOUT, IN_DIM, OUT_DIM, W1, W2,
                                              generator=torch.Generator().manual_seed(2))

    plot_raw_data(x_train, y_train)

    reg_model, reg_loss_curve, reg_acc_curve = train_regression(x_train, y_train)
    reg_r2, reg_acc = evaluate_regression(reg_model, x_holdout, y_holdout)
    print(f"RegressionNet  holdout R^2={reg_r2:.4f}  holdout neg-RMSE={reg_acc:.4f}")

    mdn_results = {}
    for K in K_VALUES:
        model, loss_curve, acc_curve = train_mdn(K, x_train, y_train)
        r2, acc = evaluate_mdn(model, x_holdout, y_holdout)
        mdn_results[K] = dict(model=model, loss_curve=loss_curve, acc_curve=acc_curve, r2=r2, acc=acc)
        print(f"MDN K={K}  holdout R^2={r2:.4f}  holdout neg-RMSE={acc:.4f}")

    plot_density_grid({K: mdn_results[K]["model"] for K in K_VALUES}, X_SLICES)

    epochs_axis = range(EPOCHS)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    for K in K_VALUES:
        ax_loss.plot(epochs_axis, mdn_results[K]["loss_curve"], label=f"MDN K={K}")
    ax_loss.plot(epochs_axis, reg_loss_curve, label="RegressionNet", color="black", linestyle="--")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Training loss (MDN: NLL, Regression: MSE)")
    ax_loss.legend()

    for K in K_VALUES:
        ax_acc.plot(epochs_axis, mdn_results[K]["acc_curve"], label=f"MDN K={K}")
    ax_acc.plot(epochs_axis, reg_acc_curve, label="RegressionNet", color="black", linestyle="--")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("training accuracy (negative RMSE)")
    ax_acc.set_title("Training accuracy vs epoch")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/loss_accuracy_vs_epoch.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.plot(K_VALUES, [mdn_results[K]["acc"] for K in K_VALUES], marker="o", label="MDN")
    ax3.axhline(reg_acc, color="black", linestyle="--", label="RegressionNet")
    ax3.set_xlabel("K (number of mixture components)")
    ax3.set_ylabel("held-out accuracy (negative RMSE)")
    ax3.set_title("Held-out accuracy vs K")
    ax3.set_xticks(K_VALUES)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f"{PLOT_DIR}/holdout_accuracy_vs_k.png", dpi=150)

    print(f"Plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
