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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wind-turbine-scada-dataset" / "T1.csv"
TARGET = "LV ActivePower (kW)"
FIG_DIR = PROJECT_ROOT / "report" / "figs"
CSV_DIR = PROJECT_ROOT / "report" / "csvs"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.describe().transpose().to_csv(CSV_DIR / "wind_turbine_data_summary.csv")
    features = [c for c in df.columns if c not in (TARGET, "Date/Time")]

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    axes = [axes] if len(features) == 1 else axes
    for ax, feature in zip(axes, features):
        ax.scatter(df[feature], df[TARGET], s=3, alpha=0.15, color="gray")
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {feature}")
    fig.tight_layout()
    output_path = FIG_DIR / "wind_turbine_features.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")
    print(f"Data summary saved to {CSV_DIR / 'wind_turbine_data_summary.csv'}")


if __name__ == "__main__":
    main()
