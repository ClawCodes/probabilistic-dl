"""
Train the MDN (across K) and RegressionNet baseline on the processed,
higher-dimensional wind turbine dataset (data/processed_data.csv, produced
by mdn.process_wind_turbine_data), predicting LV ActivePower from 6 inputs:
Wind Speed, Wind Direction, and cyclical (sin/cos) encodings of hour-of-day
and day-of-year.

Inputs and target are standardized (zero mean, unit std) using training-set
statistics before training; RMSE is reported back in the original kW units.

hour_sin/hour_cos and doy_sin/doy_cos are a matched pair each, constrained
to the unit circle - they can't be swept independently (that would query
the model at physically impossible points off the training manifold), so
every sweep plot here sweeps the *underlying angle* (hour 0-24, or
day-of-year 1-365) and recomputes both sin and cos together. Wind Speed and
Wind Direction remain simple scalar sweeps.

Produces:
  - report/figs/wind_turbine_training.png: training NLL vs epoch (MDN only,
    one curve per K - RegressionNet's MSE isn't on a comparable scale so
    it's left off this panel) and training RMSE vs epoch (MDN curves plus
    the RegressionNet baseline - RMSE is the fair cross-model comparison).
    Train-set only - see wind_turbine_holdout_vs_k.png for held-out
    performance.
  - report/figs/wind_turbine_holdout_vs_k.png: held-out RMSE vs K, one
    point per MDN plus a reference line for RegressionNet - the per-epoch
    curves above are train-only, so this is the only plot showing whether
    adding mixture components actually helps on unseen data.
  - report/figs/density_grid_wind_turbine.png: grid of
    p(ActivePower | x) plots (rows: K, columns: representative wind-speed
    slices, other features held at representative values) showing each
    weighted component density and their sum, in real kW units.
  - report/figs/curve_overlay_{speed,direction,hour,doy}_wind_turbine.png:
    RegressionNet's predicted curve vs each MDN's predicted component mean
    +/- std curves, swept over one conceptual feature at a time (other
    features held at representative values), overlaid on the raw scatter
    restricted to a matching band of a companion feature. One row per K.
  - report/figs/feature_separation_wind_turbine.png: bar chart of the
    maximum gap between MDN component means achieved while sweeping each
    of the 4 conceptual features, taken as the max over K - a single
    quantitative measure of which features the MDN actually uses to
    separate modes.
  - report/csvs/wind_turbine_summary.csv: final train/test metrics.
  - report/csvs/wind_turbine_training_history.csv: per-epoch metrics.
  - report/csvs/wind_turbine_density_grid_component_weights.csv: mixture
    weight (alpha), mean, and std of every component at every
    density-grid slice (K x slice x component) - the same (model, slice)
    pairs plotted in density_grid_wind_turbine.png.
  - report/csvs/wind_turbine_sweep_component_weights.csv: for each of the
    4 curve-overlay sweeps and each K, the mean and max (over the sweep
    grid) weight carried by the second-most-probable MDN component - a
    quantitative complement to the curve-overlay plots for judging
    whether an apparent second mode is a genuine, non-trivial fraction of
    predicted probability mass or a near-zero-weight, effectively unused
    component.
"""

import math

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
DATA_PATH = PROJECT_ROOT / "data" / "processed_data.csv"
TARGET = "LV ActivePower (kW)"
FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "Wind Speed (m/s)", "Wind Direction (°)"]
FIG_DIR = PROJECT_ROOT / "report" / "figs"
CSV_DIR = PROJECT_ROOT / "report" / "csvs"

K_VALUES = [1, 2, 3, 4, 5]
EPOCHS = 300
HIDDEN, RANK, LR = 128, 1, 1e-3
BATCH_SIZE = 64
TRAIN_FRACTION = 0.8
SEED = 0

# Representative operating point used to hold every *other* feature fixed
# while sweeping one at a time: mid-ramp wind speed and the dominant
# direction cluster (both established in the EDA), noon, and mid-year.
# hour/doy values don't matter much given the EDA finding that neither
# shows a real marginal effect - these are just fixed, stated defaults.
REP_SPEED, REP_DIRECTION, REP_HOUR, REP_DOY = 9.0, 50.0, 12.0, 182.0


def hour_to_raw(hour):
    angle = 2 * math.pi * hour / 24.0
    return {"hour_sin": torch.sin(angle) if torch.is_tensor(angle) else math.sin(angle),
            "hour_cos": torch.cos(angle) if torch.is_tensor(angle) else math.cos(angle)}


def doy_to_raw(day):
    angle = 2 * math.pi * day / 365.0
    return {"doy_sin": torch.sin(angle) if torch.is_tensor(angle) else math.sin(angle),
            "doy_cos": torch.cos(angle) if torch.is_tensor(angle) else math.cos(angle)}


def hour_from_raw(x):
    sin, cos = x[:, FEATURES.index("hour_sin")], x[:, FEATURES.index("hour_cos")]
    return torch.atan2(sin, cos) % (2 * math.pi) * 24.0 / (2 * math.pi)


def doy_from_raw(x):
    sin, cos = x[:, FEATURES.index("doy_sin")], x[:, FEATURES.index("doy_cos")]
    return torch.atan2(sin, cos) % (2 * math.pi) * 365.0 / (2 * math.pi)


# Fixed representative value for every raw column, used whenever that
# column isn't the one being swept.
REPRESENTATIVE_RAW = {
    "Wind Speed (m/s)": REP_SPEED,
    "Wind Direction (°)": REP_DIRECTION,
    **hour_to_raw(REP_HOUR),
    **doy_to_raw(REP_DOY),
}

# Percentile used to trim the speed/direction sweep domains to the
# well-supported range of the training data. Sweeping all the way to the
# hard extremes (e.g. exactly 0 or 25 m/s) queries the model where it has
# almost no training examples, which produces wild, physically impossible
# extrapolated component means (checked: e.g. a component mean of -6800 kW
# at speed=0, or +6800 kW at speed=25) - noise that would otherwise dominate
# both the curve-overlay plots and the separation metric below.
DOMAIN_TRIM_PERCENTILE = 0.01

# Each conceptual feature's sweep spec: display label, natural-unit sweep
# domain, natural->raw and raw->natural conversions, and the companion
# feature (+ band) used to filter the real-data scatter. "speed" and
# "direction" domains are placeholders here - set_data_driven_domains()
# overwrites them with the actual training data's [1st, 99th] percentile
# range before any plotting happens. hour/doy are left at their full
# natural range (0-24, 1-365) since every value is well-represented.
SWEEP_SPECS = {
    "speed": dict(
        label="Wind Speed (m/s)", domain=None,
        to_raw=lambda v: {"Wind Speed (m/s)": v},
        from_raw=lambda x: x[:, FEATURES.index("Wind Speed (m/s)")],
        filter_column="Wind Direction (°)", filter_value=REP_DIRECTION, filter_tolerance=10.0,
        output_name="curve_overlay_speed_wind_turbine.png",
    ),
    "direction": dict(
        label="Wind Direction (°)", domain=None,
        to_raw=lambda v: {"Wind Direction (°)": v},
        from_raw=lambda x: x[:, FEATURES.index("Wind Direction (°)")],
        filter_column="Wind Speed (m/s)", filter_value=REP_SPEED, filter_tolerance=1.0,
        output_name="curve_overlay_direction_wind_turbine.png",
    ),
    "hour": dict(
        label="Hour of Day", domain=(0.0, 24.0),
        to_raw=hour_to_raw, from_raw=hour_from_raw,
        filter_column="Wind Speed (m/s)", filter_value=REP_SPEED, filter_tolerance=1.0,
        output_name="curve_overlay_hour_wind_turbine.png",
    ),
    "doy": dict(
        label="Day of Year", domain=(1.0, 365.0),
        to_raw=doy_to_raw, from_raw=doy_from_raw,
        filter_column="Wind Speed (m/s)", filter_value=REP_SPEED, filter_tolerance=1.0,
        output_name="curve_overlay_doy_wind_turbine.png",
    ),
}

# Density-grid slices: vary wind speed across the ramp region, everything
# else held at the representative operating point.
DENSITY_GRID_SLICES = [
    {"Wind Speed (m/s)": 5.0},
    {"Wind Speed (m/s)": 9.0},
    {"Wind Speed (m/s)": 13.0},
]


def set_data_driven_domains(x_raw):
    """Fill in SWEEP_SPECS['speed']/['direction']['domain'] from the actual
    training data's [DOMAIN_TRIM_PERCENTILE, 1-DOMAIN_TRIM_PERCENTILE]
    quantile range, so sweeps stay inside the well-supported region."""
    for name, key in [("Wind Speed (m/s)", "speed"), ("Wind Direction (°)", "direction")]:
        col = x_raw[:, FEATURES.index(name)]
        q = torch.quantile(col, torch.tensor([DOMAIN_TRIM_PERCENTILE, 1 - DOMAIN_TRIM_PERCENTILE]))
        SWEEP_SPECS[key]["domain"] = (q[0].item(), q[1].item())


def build_raw_row(overrides):
    row = dict(REPRESENTATIVE_RAW)
    row.update(overrides)
    return [row[name] for name in FEATURES]


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
    """Minibatch training (BATCH_SIZE, reshuffled every epoch). The logged
    per-epoch loss/RMSE is evaluated once on the full training set after
    that epoch's minibatch updates, not the last minibatch's loss - so
    'epoch' still means one point per curve/history-row downstream, same
    as before batching was added."""
    torch.manual_seed(SEED)
    model = MDN(in_dim=x_train.shape[1], out_dim=1, hidden=HIDDEN, K=K, rank=RANK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    target_kw = unstandardize(y_train, y_mean, y_std)
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
            pred_kw = unstandardize(mdn_mixture_mean(alpha, mu), y_mean, y_std)
        loss_curve.append(epoch_loss)
        rmse_curve.append(rmse(pred_kw, target_kw))
    return model, loss_curve, rmse_curve


def train_regression(x_train, y_train, y_mean, y_std):
    """Minibatch training, same epoch-level logging convention as train_mdn."""
    torch.manual_seed(SEED)
    model = RegressionNet(in_dim=x_train.shape[1], out_dim=1, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve, rmse_curve = [], []
    target_kw = unstandardize(y_train, y_mean, y_std)
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
            pred_kw = unstandardize(pred, y_mean, y_std)
        loss_curve.append(epoch_loss)
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


def plot_density_grid(models_by_k, slices, x_mean, x_std, y_mean, y_std, n_grid=300):
    """Grid of p(y|x) plots: one row per K, one column per override dict in
    slices (every unlisted feature held at its representative value). Each
    subplot shows the K weighted component densities plus their sum (the
    mixture density), in real kW units on the y-axis."""
    k_values = sorted(models_by_k)
    fig, axes = plt.subplots(len(k_values), len(slices),
                              figsize=(4.5 * len(slices), 2.5 * len(k_values)),
                              squeeze=False)
    colors = plt.cm.tab10.colors
    y_mean_v, y_std_v = y_mean.item(), y_std.item()

    for row, K in enumerate(k_values):
        model = models_by_k[K]
        for col, overrides in enumerate(slices):
            ax = axes[row][col]
            x_tensor = (torch.tensor([build_raw_row(overrides)], dtype=torch.float32) - x_mean) / x_std
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

            slice_desc = ", ".join(f"{k}={v:g}" for k, v in overrides.items())
            ax.set_title(f"K={K}, {slice_desc}", fontsize=9)
            if row == len(k_values) - 1:
                ax.set_xlabel("ActivePower (kW)")
            if col == 0:
                ax.set_ylabel("p(y|x)")
        axes[row][0].legend(fontsize=7, loc="upper right")

    fig.suptitle("Predicted mixture density p(ActivePower | x) across K "
                 "(wind speed varied, other features at representative values)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "density_grid_wind_turbine.png", dpi=150)
    plt.close(fig)


def build_sweep_grid(spec, n_grid, x_mean, x_std):
    sweep_grid = torch.linspace(spec["domain"][0], spec["domain"][1], n_grid)
    sweep_raw_cols = spec["to_raw"](sweep_grid)

    x_grid_raw = torch.empty(n_grid, len(FEATURES))
    for j, name in enumerate(FEATURES):
        x_grid_raw[:, j] = sweep_raw_cols[name] if name in sweep_raw_cols else REPRESENTATIVE_RAW[name]
    return sweep_grid, (x_grid_raw - x_mean) / x_std


def plot_curve_overlay(models_by_k, reg_model, x_scatter_raw, y_scatter_raw,
                        x_mean, x_std, y_mean, y_std, spec, n_grid=200):
    """Hold every feature except one fixed at its representative value and
    sweep the remaining one (spec) across spec['domain'], overlaying:
      - RegressionNet's single predicted curve (black dashed)
      - each MDN's K component mean +/- 1 std curves (colored)
    on top of a scatter of the real data restricted to a band of the
    spec's companion (filter) feature. One row per K, all in real kW."""
    k_values = sorted(models_by_k)

    filter_idx = FEATURES.index(spec["filter_column"])
    in_band = (x_scatter_raw[:, filter_idx] - spec["filter_value"]).abs() <= spec["filter_tolerance"]
    x_band, power_band = x_scatter_raw[in_band], y_scatter_raw[in_band, 0]
    sweep_band = spec["from_raw"](x_band)

    sweep_grid, x_grid = build_sweep_grid(spec, n_grid, x_mean, x_std)
    y_mean_v, y_std_v = y_mean.item(), y_std.item()

    with torch.no_grad():
        reg_pred_kw = reg_model(x_grid) * y_std_v + y_mean_v  # (n_grid, 1)

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(k_values), 1, figsize=(8, 3.5 * len(k_values)), squeeze=False)
    axes = [ax[0] for ax in axes]

    # Fix the y-range to the real data span (with padding) rather than
    # autoscaling: a spare/near-degenerate component can have a wildly
    # inflated predicted std in regions it doesn't own, which would
    # otherwise blow out the axis and hide the actual fit.
    power_min, power_max = power_band.min().item(), power_band.max().item()
    padding = 0.1 * (power_max - power_min)
    y_lo, y_hi = power_min - padding, power_max + padding

    for row, K in enumerate(k_values):
        ax = axes[row]
        ax.scatter(sweep_band.numpy(), power_band.numpy(), s=6, alpha=0.15,
                   color="gray", label="data" if row == 0 else None)
        ax.plot(sweep_grid.numpy(), reg_pred_kw[:, 0].numpy(), color="black",
                linewidth=1.8, linestyle="--", label="RegressionNet" if row == 0 else None)

        with torch.no_grad():
            alpha, mu, Lambda = models_by_k[K](x_grid)             # (n_grid,K), (n_grid,K,1), (n_grid,K,1,1)
        mu_kw = mu[:, :, 0] * y_std_v + y_mean_v                     # (n_grid, K)
        std_kw = Lambda[:, :, 0, 0].reciprocal().sqrt() * y_std_v    # (n_grid, K)

        for k in range(K):
            color = colors[k % len(colors)]
            mean_k, std_k = mu_kw[:, k].numpy(), std_kw[:, k].numpy()
            avg_weight = alpha[:, k].mean().item()
            ax.plot(sweep_grid.numpy(), mean_k, color=color, linewidth=1.5,
                    label=f"MDN component {k} (avg weight={avg_weight:.2f})" if row == 0 else None)
            ax.fill_between(sweep_grid.numpy(), mean_k - std_k, mean_k + std_k, color=color, alpha=0.15)

        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("ActivePower (kW)")
        ax.set_title(f"K={K}", fontsize=10)
    axes[-1].set_xlabel(spec["label"])
    axes[0].legend(fontsize=7, loc="upper left")

    fig.suptitle(f"RegressionNet vs MDN predicted components "
                 f"({spec['filter_column']}={spec['filter_value']:g}±{spec['filter_tolerance']:g})",
                 wrap=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / spec["output_name"], dpi=150)
    plt.close(fig)


def plot_feature_separation(models_by_k, x_mean, x_std, y_mean, y_std, n_grid=200):
    """For each conceptual feature, sweep it (other features held at their
    representative values) and compute the max gap between the most
    extreme MDN component means, taken as the max over K. A single bar
    chart ranking all 4 features by this separation metric - a
    quantitative complement to eyeballing the curve-overlay plots."""
    y_mean_v, y_std_v = y_mean.item(), y_std.item()
    labels, separations = [], []

    for spec in SWEEP_SPECS.values():
        _, x_grid = build_sweep_grid(spec, n_grid, x_mean, x_std)
        best = 0.0
        for K in sorted(models_by_k):
            with torch.no_grad():
                _, mu, _ = models_by_k[K](x_grid)
            mu_kw = mu[:, :, 0] * y_std_v + y_mean_v          # (n_grid, K)
            gap = (mu_kw.max(dim=1).values - mu_kw.min(dim=1).values).max().item()
            best = max(best, gap)
        labels.append(spec["label"])
        separations.append(best)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, separations, color="lightsteelblue", edgecolor="black")
    ax.set_ylabel("max component-mean gap (kW)")
    ax.set_title("MDN component separation by swept feature (max over K)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_separation_wind_turbine.png", dpi=150)
    plt.close(fig)
    return dict(zip(labels, separations))


def plot_holdout_vs_k(mdn_results, reg_test):
    """Held-out RMSE vs K: one point per MDN, plus a horizontal reference
    line for RegressionNet. The per-epoch curves in wind_turbine_training.png
    are train-set only (evaluated once per epoch on training data as it
    trains); this is the only plot that shows held-out performance as a
    function of K."""
    k_values = sorted(mdn_results)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k_values, [mdn_results[K]["test"]["rmse"] for K in k_values],
            marker="o", label="MDN")
    ax.axhline(reg_test["rmse"], color="black", linestyle="--", label="RegressionNet")
    ax.set_xlabel("K (number of mixture components)")
    ax.set_ylabel("held-out RMSE (kW)")
    ax.set_title("Held-out RMSE vs K")
    ax.set_xticks(k_values)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wind_turbine_holdout_vs_k.png", dpi=150)
    plt.close(fig)


def save_density_grid_component_weights(models_by_k, slices, x_mean, x_std, y_mean, y_std):
    """Mixture weight/mean/std of every component at every density-grid
    slice, for every K - the same (model, slice) pairs plot_density_grid
    plots, saved as a table for the report."""
    y_mean_v, y_std_v = y_mean.item(), y_std.item()
    rows = []
    for K in sorted(models_by_k):
        model = models_by_k[K]
        for overrides in slices:
            x_tensor = (torch.tensor([build_raw_row(overrides)], dtype=torch.float32) - x_mean) / x_std
            with torch.no_grad():
                alpha, mu, Lambda = model(x_tensor)
            slice_desc = ", ".join(f"{k}={v:g}" for k, v in overrides.items())
            for k in range(K):
                rows.append({
                    "slice": slice_desc,
                    "k": K,
                    "component": k,
                    "weight": alpha[0, k].item(),
                    "mean_kw": (mu[0, k, 0] * y_std_v + y_mean_v).item(),
                    "std_kw": (Lambda[0, k, 0, 0].reciprocal().sqrt() * y_std_v).item(),
                })
    path = CSV_DIR / "wind_turbine_density_grid_component_weights.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_sweep_component_weights(models_by_k, x_mean, x_std, y_mean, y_std, n_grid=200):
    """For each curve-overlay sweep, record the second-largest MDN
    component weight (share of predicted probability mass held by the
    second most probable mode) at every swept grid point, using the same
    models and sweep grids as plot_curve_overlay. Summarized as mean/max
    over the sweep, plus where the max occurs - the diagnostic used to
    tell a genuine competing mode (non-trivial weight) apart from a
    near-zero-weight, effectively unused component."""
    rows = []
    for spec in SWEEP_SPECS.values():
        sweep_grid, x_grid = build_sweep_grid(spec, n_grid, x_mean, x_std)
        for K in sorted(models_by_k):
            with torch.no_grad():
                alpha, _, _ = models_by_k[K](x_grid)
            if K == 1:
                second_weight = torch.zeros(n_grid)
            else:
                sorted_w, _ = torch.sort(alpha, dim=1, descending=True)
                second_weight = sorted_w[:, 1]
            max_idx = second_weight.argmax().item()
            rows.append({
                "feature": spec["label"],
                "k": K,
                "mean_second_weight": second_weight.mean().item(),
                "max_second_weight": second_weight[max_idx].item(),
                "x_at_max": sweep_grid[max_idx].item(),
            })
    path = CSV_DIR / "wind_turbine_sweep_component_weights.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    x_train_raw, y_train_raw, x_holdout_raw, y_holdout_raw = load_data()
    x_train, (x_holdout,), x_mean, x_std = standardize(x_train_raw, x_holdout_raw)
    y_train, (y_holdout,), y_mean, y_std = standardize(y_train_raw, y_holdout_raw)
    set_data_driven_domains(x_train_raw)

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

    # RegressionNet is deliberately left off this panel: its MSE and the
    # MDN's NLL are different loss functions on different scales (NLL can be
    # negative, MSE can't), so overlaying them on one axis would invite a
    # cross-model "lower curve wins" reading that isn't valid. The RMSE
    # panel below is the fair cross-model comparison (same kW units for
    # every model regardless of training objective); this panel is only
    # for checking each MDN's own NLL is converging.
    for K in K_VALUES:
        ax_loss.plot(epochs_axis, mdn_results[K]["loss_curve"], label=f"MDN K={K}")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training NLL")
    ax_loss.set_title("Training NLL vs epoch")
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

    plot_holdout_vs_k(mdn_results, reg_test)
    print(f"Plot saved to {FIG_DIR / 'wind_turbine_holdout_vs_k.png'}")

    models_by_k = {K: mdn_results[K]["model"] for K in K_VALUES}

    plot_density_grid(models_by_k, DENSITY_GRID_SLICES, x_mean, x_std, y_mean, y_std)
    print(f"Plot saved to {FIG_DIR / 'density_grid_wind_turbine.png'}")

    density_weights_path = save_density_grid_component_weights(
        models_by_k, DENSITY_GRID_SLICES, x_mean, x_std, y_mean, y_std)
    print(f"Weights saved to {density_weights_path}")

    for spec in SWEEP_SPECS.values():
        plot_curve_overlay(models_by_k, reg_model, x_train_raw, y_train_raw,
                            x_mean, x_std, y_mean, y_std, spec)
        print(f"Plot saved to {FIG_DIR / spec['output_name']}")

    sweep_weights_path = save_sweep_component_weights(models_by_k, x_mean, x_std, y_mean, y_std)
    print(f"Weights saved to {sweep_weights_path}")

    separations = plot_feature_separation(models_by_k, x_mean, x_std, y_mean, y_std)
    print(f"Plot saved to {FIG_DIR / 'feature_separation_wind_turbine.png'}")
    print("Max component-mean separation by feature:")
    for label, gap in separations.items():
        print(f"  {label:20s} {gap:8.1f} kW")

    print(f"Statistics saved to {CSV_DIR}")


if __name__ == "__main__":
    main()
