"""
Exploratory check: plot every numeric feature of the wind turbine SCADA
dataset against LV ActivePower, to visually validate whether the
feature -> power relationship is multimodal (a good candidate for an MDN).

Date/Time is excluded since it's a timestamp, not a continuous feature -
there's no meaningful scatter of "power vs raw timestamp" to check for
multimodality.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/wind-turbine-scada-dataset/T1.csv"
TARGET = "LV ActivePower (kW)"
PLOT_DIR = "mdn/plots/wind_turbine"


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c not in (TARGET, "Date/Time")]

    fig, axes = plt.subplots(len(features), 1, figsize=(7, 4 * len(features)))
    axes = [axes] if len(features) == 1 else axes
    for ax, feature in zip(axes, features):
        ax.scatter(df[feature], df[TARGET], s=3, alpha=0.15, color="gray")
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {feature}")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/feature_vs_active_power.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {PLOT_DIR}/feature_vs_active_power.png")


if __name__ == "__main__":
    main()
