"""
Exploratory check: plot every numeric feature of the processed wind turbine
dataset (data/processed_data.csv, produced by mdn.process_wind_turbine_data)
against LV ActivePower, to visually validate whether the feature -> power
relationship is multimodal (a good candidate for an MDN).

processed_data.csv already carries only numeric, model-ready columns (the
cyclical hour/day-of-year encodings plus Wind Speed, Wind Direction, and the
target) - there's no raw timestamp column left to exclude.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed_data.csv"
TARGET = "LV ActivePower (kW)"
FIG_DIR = PROJECT_ROOT / "report" / "figs"
CSV_DIR = PROJECT_ROOT / "report" / "csvs"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.describe().transpose().to_csv(CSV_DIR / "wind_turbine_data_summary.csv")
    features = [c for c in df.columns if c != TARGET]

    n_cols = 3
    n_rows = -(-len(features) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, feature in zip(axes_flat, features):
        ax.scatter(df[feature], df[TARGET], s=3, alpha=0.15, color="gray")
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {feature}")
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path = FIG_DIR / "wind_turbine_features.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")
    print(f"Data summary saved to {CSV_DIR / 'wind_turbine_data_summary.csv'}")


if __name__ == "__main__":
    main()
