"""
Train the MDN (across K) and RegressionNet baseline on the real wind turbine
SCADA dataset (data/wind-turbine-scada-dataset/T1.csv), predicting LV
ActivePower from Wind Speed and Wind Direction only (Theoretical_Power_Curve
is dropped since it's a near-deterministic transform of Wind Speed).

Inputs and target are standardized (zero mean, unit std) using training-set
statistics before training; RMSE is reported back in the original kW units.

Produces:
  - report/figs/wind_turbine_training.png: training loss and RMSE vs epoch, one curve per K
    plus the RegressionNet baseline.
  - report/figs/density_grid_wind_turbine.png: grid of
    p(ActivePower | wind speed, direction) plots
    (rows: K, columns: representative (speed, direction) slices) showing
    each weighted component density and their sum, in real kW units.
  - report/csvs/wind_turbine_summary.csv: final train/test metrics.
  - report/csvs/wind_turbine_training_history.csv: per-epoch metrics.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wind-turbine-scada-dataset" / "T1.csv"
TARGET = "LV ActivePower (kW)"
FEATURES = ["Wind Speed (m/s)", "Wind Direction (°)"]
FIG_DIR = PROJECT_ROOT / "report" / "figs"
CSV_DIR = PROJECT_ROOT / "report" / "csvs"

# (wind speed m/s, wind direction deg) slices spanning the ramp region where
# the EDA scatter showed the clearest normal-vs-curtailed bimodality, all at
# the dominant wind direction cluster (~50 deg).
X_SLICES_RAW = [(5.0, 50.0), (9.0, 50.0), (13.0, 50.0)]

K_VALUES = [1, 2, 3, 4, 5]
EPOCHS = 300
HIDDEN, RANK, LR = 128, 1, 1e-3
TRAIN_FRACTION = 0.8
SEED = 0


def load_data():
    df = pd.read_csv(DATA_PATH)
    x = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(df[[TARGET]].values, dtype=torch.float32)

    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(x.shape[0], generator=generator)
    n_train = int(TRAIN_FRACTION * x.shape[0])
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]
    return x[train_idx], y[train_idx], x[holdout_idx], y[holdout_idx]


def standardize(train, *others):
    mean, std = train.mean(dim=0, keepdim=True), train.std(dim=0, keepdim=True)
    scaled_others = [(o - mean) / std for o in others]
    return (train - mean) / std, scaled_others, mean, std


def unstandardize(z, mean, std):
    return z * std + mean


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def r_squared(pred, target):
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot).mean().item()


def mdn_mixture_mean(alpha, mu):
    return (alpha.unsqueeze(-1) * mu).sum(dim=1)


def univariate_density(y_grid, mu, std):
    return torch.exp(-0.5 * ((y_grid - mu) / std) ** 2) / (std * (2 * torch.pi) ** 0.5)


def train_mdn(K, x_train, y_train, y_mean, y_std):
    torch.manual_seed(SEED)
    model = MDN(in_dim=x_train.shape[1], out_dim=1, hidden=HIDDEN, K=K, rank=RANK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    target_kw = unstandardize(y_train, y_mean, y_std)
    for _ in range(EPOCHS):
        alpha, mu, Lambda = model(x_train)
        loss = mdn_nll(alpha, mu, Lambda, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred_kw = unstandardize(mdn_mixture_mean(alpha, mu), y_mean, y_std)
        loss_curve.append(loss.item())
        rmse_curve.append(rmse(pred_kw, target_kw))
    return model, loss_curve, rmse_curve


def train_regression(x_train, y_train, y_mean, y_std):
    torch.manual_seed(SEED)
    model = RegressionNet(in_dim=x_train.shape[1], out_dim=1, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    target_kw = unstandardize(y_train, y_mean, y_std)
    for _ in range(EPOCHS):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred_kw = unstandardize(pred, y_mean, y_std)
        loss_curve.append(loss.item())
        rmse_curve.append(rmse(pred_kw, target_kw))
    return model, loss_curve, rmse_curve


def evaluate_mdn(model, x, y, y_mean, y_std):
    with torch.no_grad():
        alpha, mu, Lambda = model(x)
        pred_kw = unstandardize(mdn_mixture_mean(alpha, mu), y_mean, y_std)
        target_kw = unstandardize(y, y_mean, y_std)
        return {
            "nll": mdn_nll(alpha, mu, Lambda, y).item(),
            "rmse": rmse(pred_kw, target_kw),
            "r2": r_squared(pred_kw, target_kw),
        }


def evaluate_regression(model, x, y, y_mean, y_std):
    with torch.no_grad():
        pred = model(x)
        pred_kw = unstandardize(pred, y_mean, y_std)
        target_kw = unstandardize(y, y_mean, y_std)
        return {
            "mse": F.mse_loss(pred, y).item(),
            "rmse": rmse(pred_kw, target_kw),
            "r2": r_squared(pred_kw, target_kw),
        }


def metric(value):
    """Format a metric consistently for both CSV output and LaTeX display."""
    return "N/A" if value is None else f"{value:.6f}"


def plot_density_grid(models_by_k, x_slices_raw, x_mean, x_std, y_mean, y_std, n_grid=300):
    """Grid of p(y|x) plots: one row per K, one column per (speed, direction)
    slice. Each subplot shows the K weighted component densities plus their
    sum (the mixture density), in real kW units on the y-axis."""
    k_values = sorted(models_by_k)
    fig, axes = plt.subplots(len(k_values), len(x_slices_raw),
                              figsize=(4.5 * len(x_slices_raw), 2.5 * len(k_values)),
                              squeeze=False)
    colors = plt.cm.tab10.colors
    y_mean_v, y_std_v = y_mean.item(), y_std.item()

    for row, K in enumerate(k_values):
        model = models_by_k[K]
        for col, x_raw in enumerate(x_slices_raw):
            ax = axes[row][col]
            x_tensor = (torch.tensor([x_raw], dtype=torch.float32) - x_mean) / x_std
            with torch.no_grad():
                alpha, mu, Lambda = model(x_tensor)
            alpha_k = alpha[0]                                    # (K,)
            mu_kw = mu[0, :, 0] * y_std_v + y_mean_v               # (K,), real kW
            std_kw = Lambda[0, :, 0, 0].reciprocal().sqrt() * y_std_v  # (K,), real kW

            y_min = (mu_kw - 4 * std_kw).min().item()
            y_max = (mu_kw + 4 * std_kw).max().item()
            y_grid = torch.linspace(y_min, y_max, n_grid)

            mixture_density = torch.zeros(n_grid)
            for k in range(K):
                weighted = alpha_k[k] * univariate_density(y_grid, mu_kw[k], std_kw[k])
                mixture_density += weighted
                ax.plot(y_grid.numpy(), weighted.numpy(), color=colors[k % len(colors)],
                        linewidth=1.2, label=f"component {k}" if col == 0 else None)
            ax.plot(y_grid.numpy(), mixture_density.numpy(), color="black", linewidth=1.8,
                    linestyle="--", label="mixture" if col == 0 else None)

            speed, direction = x_raw
            ax.set_title(f"K={K}, speed={speed:g} m/s, dir={direction:g}°", fontsize=9)
            if row == len(k_values) - 1:
                ax.set_xlabel("ActivePower (kW)")
            if col == 0:
                ax.set_ylabel("p(y|x)")
        axes[row][0].legend(fontsize=7, loc="upper right")

    fig.suptitle("Predicted mixture density p(ActivePower | wind speed, direction) across K")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "density_grid_wind_turbine.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    x_train_raw, y_train_raw, x_holdout_raw, y_holdout_raw = load_data()
    x_train, (x_holdout,), x_mean, x_std = standardize(x_train_raw, x_holdout_raw)
    y_train, (y_holdout,), y_mean, y_std = standardize(y_train_raw, y_holdout_raw)

    reg_model, reg_loss_curve, reg_rmse_curve = train_regression(x_train, y_train, y_mean, y_std)
    reg_train = evaluate_regression(reg_model, x_train, y_train, y_mean, y_std)
    reg_test = evaluate_regression(reg_model, x_holdout, y_holdout, y_mean, y_std)
    print(f"RegressionNet  holdout RMSE = {reg_test['rmse']:.2f} kW")

    mdn_results = {}
    for K in K_VALUES:
        model, loss_curve, rmse_curve = train_mdn(K, x_train, y_train, y_mean, y_std)
        train_metrics = evaluate_mdn(model, x_train, y_train, y_mean, y_std)
        test_metrics = evaluate_mdn(model, x_holdout, y_holdout, y_mean, y_std)
        mdn_results[K] = dict(
            model=model,
            loss_curve=loss_curve,
            rmse_curve=rmse_curve,
            train=train_metrics,
            test=test_metrics,
        )
        print(f"MDN K={K}       holdout RMSE = {test_metrics['rmse']:.2f} kW")

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
            "train_mse": metric((train_metrics["rmse"] / y_std.item()) ** 2),
            "test_mse": metric((test_metrics["rmse"] / y_std.item()) ** 2),
        })
    summary_path = CSV_DIR / "wind_turbine_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    write_latex_table(summary_path)

    history_rows = []
    for epoch, (loss, train_rmse) in enumerate(
            zip(reg_loss_curve, reg_rmse_curve), start=1):
        history_rows.append({
            "model": "Deterministic", "k": "", "epoch": epoch,
            "loss_name": "MSE", "train_loss": loss, "train_rmse_kw": train_rmse,
        })
    for K in K_VALUES:
        model_name = "Single Gaussian" if K == 1 else "MDN"
        for epoch, (loss, train_rmse) in enumerate(
                zip(mdn_results[K]["loss_curve"], mdn_results[K]["rmse_curve"]), start=1):
            history_rows.append({
                "model": model_name, "k": K, "epoch": epoch,
                "loss_name": "NLL", "train_loss": loss,
                "train_rmse_kw": train_rmse,
            })
    pd.DataFrame(history_rows).to_csv(
        CSV_DIR / "wind_turbine_training_history.csv", index=False)

    epochs_axis = range(EPOCHS)
    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(12, 5))

    for K in K_VALUES:
        ax_loss.plot(epochs_axis, mdn_results[K]["loss_curve"], label=f"MDN K={K}")
    ax_loss.plot(epochs_axis, reg_loss_curve, label="RegressionNet (MSE)", color="black", linestyle="--")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Training loss vs epoch")
    ax_loss.legend(fontsize=8)

    for K in K_VALUES:
        ax_rmse.plot(epochs_axis, mdn_results[K]["rmse_curve"], label=f"MDN K={K}")
    ax_rmse.plot(epochs_axis, reg_rmse_curve, label="RegressionNet", color="black", linestyle="--")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.set_ylabel("training RMSE (kW)")
    ax_rmse.set_title("Training RMSE vs epoch")
    ax_rmse.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "wind_turbine_training.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {FIG_DIR / 'wind_turbine_training.png'}")

    plot_density_grid({K: mdn_results[K]["model"] for K in K_VALUES},
                       X_SLICES_RAW, x_mean, x_std, y_mean, y_std)
    print(f"Plot saved to {FIG_DIR / 'density_grid_wind_turbine.png'}")
    print(f"Statistics saved to {CSV_DIR}")


if __name__ == "__main__":
    main()
