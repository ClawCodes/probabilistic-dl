"""
EDA helpers for demonstrating that P(y|x) is empirically multimodal, purely
from raw data - no trained model involved.

`plot_conditional_histograms` is deliberately generic (takes plain x/y
arrays, a list of x slices, and a tolerance band) so it can be reused for
different datasets. `run_synthetic` runs it on the toy bimodal dataset from
`mdn.compare_k`; `run_wind_turbine` runs it on the real, processed wind
turbine dataset (data/processed_data.csv).

These are deliberately marginal (single-feature) slices - they do not
control for other features. hour-of-day and day-of-year are not included:
neither shows a clean bimodal split in the raw feature-vs-target scatter the
way wind speed and wind direction do, so they're not useful for this
"multimodality is visible in the raw marginal" demonstration.

That does NOT mean day-of-year (or hour-of-day) has no real effect on
power - a check that isolated two narrow, disjoint day-of-year windows
(same doy_sin band, opposite doy_cos sign, so ~early March vs. ~late
April/early May) found those two windows have visibly different power
distributions (early-March-band mean ~2280 kW vs. late-April-band mean
~710 kW), not the same shape. With only one year of data that's confounded
with "this particular year's weather in those two weeks" and can't be
taken as proof of a seasonal effect, but it rules out the tidier "it's
just wind-speed variance within any multi-day window" explanation. The
trained MDN sweeps in mdn.train_wind_turbine (see
report/figs/feature_separation_wind_turbine.png) show both hour-of-day and
day-of-year driving non-trivial MDN component separation once wind
speed/direction are held fixed - smaller than wind speed or direction, but
not negligible.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from mdn.compare_k import make_bimodal_data, IN_DIM, OUT_DIM, N_TRAIN, X_SLICES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figs"
DATA_PATH = PROJECT_ROOT / "data" / "processed_data.csv"

TOLERANCE = 0.2

WIND_SPEED_SLICES = [5.0, 9.0, 13.0]
WIND_SPEED_TOLERANCE = 0.5

WIND_DIRECTION_SLICES = [50.0, 200.0]
WIND_DIRECTION_TOLERANCE = 10.0


def plot_conditional_histograms(x, y, slices, tolerance, xlabel, ylabel, title,
                                 output_path, bins=30):
    """Grid of histograms of y restricted to a narrow band around each x0 in
    slices (|x - x0| <= tolerance). Unlike a scatter plot, this makes
    bimodality in P(y|x0) explicit as separated peaks rather than something
    the reader has to infer from an "X" shape."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4), sharey=True)
    axes = [axes] if len(slices) == 1 else axes
    for ax, x0 in zip(axes, slices):
        mask = np.abs(x - x0) <= tolerance
        ax.hist(y[mask], bins=bins, color="lightsteelblue", edgecolor="black")
        ax.set_title(f"{xlabel}={x0:g}  (n={mask.sum()})", fontsize=10)
        ax.set_xlabel(ylabel)
    axes[0].set_ylabel("count")

    fig.suptitle(title, wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_synthetic():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    W1, W2 = torch.randn(IN_DIM, OUT_DIM), torch.randn(IN_DIM, OUT_DIM)
    x_train, y_train = make_bimodal_data(N_TRAIN, IN_DIM, OUT_DIM, W1, W2,
                                          generator=torch.Generator().manual_seed(1))

    output_path = FIG_DIR / "synthetic_conditional_histograms.png"
    plot_conditional_histograms(
        x_train[:, 0], y_train[:, 0], slices=X_SLICES, tolerance=TOLERANCE,
        xlabel="x", ylabel="y",
        title="Empirical P(y|x) at fixed x slices (toy bimodal data)",
        output_path=output_path,
    )
    print(f"Plot saved to {output_path}")


def run_wind_turbine():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    speed, direction = df["Wind Speed (m/s)"], df["Wind Direction (°)"]
    power = df["LV ActivePower (kW)"]

    speed_path = FIG_DIR / "wind_turbine_conditional_histograms_speed.png"
    plot_conditional_histograms(
        speed, power, slices=WIND_SPEED_SLICES, tolerance=WIND_SPEED_TOLERANCE,
        xlabel="Wind Speed (m/s)", ylabel="ActivePower (kW)",
        title="Empirical P(ActivePower | wind speed) at fixed speed slices",
        output_path=speed_path,
    )
    print(f"Plot saved to {speed_path}")

    direction_path = FIG_DIR / "wind_turbine_conditional_histograms_direction.png"
    plot_conditional_histograms(
        direction, power, slices=WIND_DIRECTION_SLICES, tolerance=WIND_DIRECTION_TOLERANCE,
        xlabel="Wind Direction (deg)", ylabel="ActivePower (kW)",
        title="Empirical P(ActivePower | wind direction) at fixed direction slices "
              "(marginal, not controlling for wind speed)",
        output_path=direction_path,
    )
    print(f"Plot saved to {direction_path}")


if __name__ == "__main__":
    run_synthetic()
    run_wind_turbine()
