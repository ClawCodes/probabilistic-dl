"""
Feature-engineer the wind turbine SCADA dataset: replace the raw Date/Time
timestamp with cyclical (sin/cos) encodings of hour-of-day and day-of-year,
so a model sees time as a position on a repeating cycle rather than a
discontinuous linear count (e.g. hour 23 and hour 0 should be "close").

Hour-of-day uses the fractional hour (hour + minute/60) since the data is
recorded at 10-minute resolution, not whole hours. Day-of-year is encoded
against the actual number of days in that row's year (365 or 366), to stay
correct across leap years.

Keeps only: hour_sin, hour_cos, doy_sin, doy_cos, LV ActivePower (kW),
Wind Speed (m/s), Wind Direction (deg). Writes the result to
data/processed_data.csv.
"""

import calendar

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wind-turbine-scada-dataset" / "T1.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed_data.csv"


def add_cyclical_time_features(df):
    dt = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")

    fractional_hour = dt.dt.hour + dt.dt.minute / 60.0
    hour_angle = 2 * np.pi * fractional_hour / 24.0

    days_in_year = dt.dt.year.apply(lambda year: 366 if calendar.isleap(year) else 365)
    doy_angle = 2 * np.pi * dt.dt.dayofyear / days_in_year

    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"] = np.sin(doy_angle)
    df["doy_cos"] = np.cos(doy_angle)
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_cyclical_time_features(df)

    columns = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "LV ActivePower (kW)", "Wind Speed (m/s)", "Wind Direction (°)",
    ]
    df[columns].to_csv(OUTPUT_PATH, index=False)
    print(f"Processed data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
