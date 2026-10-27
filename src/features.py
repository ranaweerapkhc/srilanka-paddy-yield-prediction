"""
Feature engineering shared by the notebooks and the Streamlit app.

Must match notebooks/02_modelling.ipynb, which trained the saved models.
"""

import warnings

import numpy as np
import pandas as pd

CATEGORICAL = ["district", "season"]
NUMERIC = ["year", "log_sown_total", "sown_share_major", "sown_share_minor", "sown_share_rainfed",
           "rain_total_mm", "rain_min_month_mm", "rain_max_month_mm", "rain_cv", "rain_pre_season_mm",
           "temp_mean_c", "temp_max_c", "temp_min_c", "humidity_pct", "solar_mj_m2_day", "wind_m_s",
           "lag_prev_season", "lag_same_season", "roll3_same_season"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "yield_total"

ZONES = {
    "Dry": ["ANURADHAPURA", "POLONNARUWA", "HAMBANTOTA", "AMPARA", "TRINCOMALEE", "BATTICALOA", "JAFFNA",
            "MANNAR", "VAVUNIYA", "MULLATIVU", "KILLINOCHCHI", "PUTTALAM", "MONARAGALA", "MAHAWELI H", "UDAWALAWE"],
    "Intermediate": ["KURUNEGALA", "MATALE", "BADULLA"],
    "Wet": ["COLOMBO", "GAMPAHA", "KALUTARA", "GALLE", "MATARA", "RATNAPURA", "KEGALLE", "KANDY", "NUWARA ELIYA"],
}
ZONE_OF = {d: z for z, ds in ZONES.items() for d in ds}


def season_index(year, season):
    """Chronological season index: ... Maha Y -> Yala Y -> Maha Y+1 ..."""
    return np.asarray(year) * 2 + (np.asarray(season) == "Yala").astype(int)


def add_features(df):
    """Add log sown area and the three lag features (past yields known before the season)."""
    df = df.copy()
    df["log_sown_total"] = np.log1p(df["sown_total"])
    df["t"] = season_index(df["year"], df["season"])
    y_at = df.set_index(["district", "t"])[TARGET]

    def lookup(d, t):
        return y_at.get((d, t), np.nan)

    df["lag_prev_season"] = [lookup(d, t - 1) for d, t in zip(df.district, df.t)]
    df["lag_same_season"] = [lookup(d, t - 2) for d, t in zip(df.district, df.t)]
    past3 = np.array([[lookup(d, t - k) for k in (2, 4, 6)] for d, t in zip(df.district, df.t)])
    enough = np.sum(~np.isnan(past3), axis=1) >= 2
    with warnings.catch_warnings():  # rows with no past values give "mean of empty slice"
        warnings.simplefilter("ignore", RuntimeWarning)
        df["roll3_same_season"] = np.where(enough, np.nanmean(past3, axis=1), np.nan)
    df["zone"] = df["district"].map(ZONE_OF)
    return df


def load_dataset(root):
    """Read the processed dataset and add the model features."""
    return add_features(pd.read_csv(root / "data_processed" / "paddy_dataset.csv"))
