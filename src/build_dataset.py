"""
Build the modelling dataset: one row per district x season x year.

Inputs : data_raw/dcs/*.xlsx                       (DCS paddy statistics)
         data_raw/nasa_power/weather_monthly.csv   (from src/fetch_weather.py)
Output : data_processed/paddy_dataset.csv

Run from the project root:
    python src/build_dataset.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DCS_DIR = ROOT / "data_raw" / "dcs"
WEATHER_CSV = ROOT / "data_raw" / "nasa_power" / "weather_monthly.csv"
OUT_CSV = ROOT / "data_processed" / "paddy_dataset.csv"

DCS_FILES = {
    # (measure, season): file
    ("yield", "Maha"): "Average_Yeild-Maha_Season-1979-2025.xlsx",
    ("yield", "Yala"): "Average_Yeild-Yala_Season-1979-2025.xlsx",
    ("sown", "Maha"): "Sown_Extent-Maha_Seasons-1979-2025.xlsx",
    ("sown", "Yala"): "Sown_Extent-Yala_Seasons-1979-2025.xlsx",
    ("harvested", "Maha"): "Harvested_Extent-Maha_Seasons-1979-2025.xlsx",
    ("harvested", "Yala"): "Harvested_Extent-Yala_Seasons-1979-2025.xlsx",
}
IRRIGATION = ["major", "minor", "rainfed", "total"]

# Growing months of each season. Maha months Oct-Feb span two calendar years
# (offset -1 = the year before the harvest year).
SEASON_MONTHS = {
    "Maha": [(-1, 10), (-1, 11), (-1, 12), (0, 1), (0, 2)],
    "Yala": [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8)],
}
# Month before planting (land preparation / soil moisture).
PRE_SEASON_MONTH = {"Maha": (-1, 9), "Yala": (0, 3)}

# Seasons hit by the chemical fertilizer import ban (May 2021) and the
# shortage that followed (keyed by harvest year). 2021 Yala is excluded:
# it was planted before the ban and national yield did not drop.
FERTILIZER_SHOCK = {("Maha", 2022), ("Yala", 2022)}


def clean_district(name):
    """Normalise district names that are spelled differently across files."""
    name = re.sub(r"[^A-Z ]", "", str(name).upper())
    name = re.sub(r"\s+", " ", name).strip()
    return "MAHAWELI H" if name.replace(" ", "") == "MAHAWELIH" else name


def to_number(value):
    """DCS uses '-' for not available; some cells may contain commas."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def harvest_year(label, season):
    """'1978/1979' (Maha) -> 1979 ; '1979' (Yala) -> 1979. Returns None if unreadable."""
    years = re.findall(r"\d{4}", str(label))
    if not years:
        return None
    return int(years[-1]) if season == "Maha" else int(years[0])


def parse_dcs(path, measure, season):
    """Parse a DCS sheet where every season has 4 columns: Major, Minor, Rainfed, Total."""
    raw = pd.read_excel(path, header=None)

    # Header row = the row whose second column says 'District'.
    header_row = raw.index[raw[1].astype(str).str.strip().str.upper() == "DISTRICT"][0]
    labels = raw.iloc[header_row, 2:].tolist()

    # Seasons are consecutive, one 4-column block each. Use the first readable
    # label as the anchor and count forward; this also fixes typos such as '198'.
    first = next(harvest_year(l, season) for l in labels[::4] if harvest_year(l, season))
    n_blocks = len(labels) // 4

    rows = []
    for r in range(header_row + 1, len(raw)):
        district = raw.iloc[r, 1]
        if pd.isna(district) or str(district).strip() == "":
            continue
        district = clean_district(district)
        if district in {"SRI LANKA", "DISTRICT"}:
            continue
        for b in range(n_blocks):
            values = [to_number(raw.iloc[r, 2 + 4 * b + k]) for k in range(4)]
            rows.append({
                "district": district,
                "season": season,
                "year": first + b,
                **{f"{measure}_{irr}": v for irr, v in zip(IRRIGATION, values)},
            })
    return pd.DataFrame(rows)


def load_dcs():
    """Merge yield, sown and harvested tables for both seasons."""
    frames = {}
    for (measure, season), fname in DCS_FILES.items():
        frames.setdefault(season, []).append(parse_dcs(DCS_DIR / fname, measure, season))

    seasons = []
    for season, parts in frames.items():
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on=["district", "season", "year"], how="outer")
        seasons.append(merged)
    return pd.concat(seasons, ignore_index=True)


def season_weather(weather):
    """Aggregate monthly weather into one row per district x season x year."""
    w = weather.set_index(["district", "year", "month"]).sort_index()
    days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    rows = []
    for district in w.index.get_level_values(0).unique():
        d = w.loc[district]
        years = d.index.get_level_values(0).unique()
        for season, months in SEASON_MONTHS.items():
            for year in years:
                keys = [(year + off, m) for off, m in months]
                if not all(k in d.index for k in keys):
                    continue  # season not fully covered by weather data
                s = d.loc[keys]
                monthly_rain = s["rain_mm_day"].values * np.array([days[m] for _, m in keys])
                pre_off, pre_m = PRE_SEASON_MONTH[season]
                pre_key = (year + pre_off, pre_m)
                rows.append({
                    "district": district,
                    "season": season,
                    "year": year,
                    "rain_total_mm": monthly_rain.sum(),
                    "rain_min_month_mm": monthly_rain.min(),
                    "rain_max_month_mm": monthly_rain.max(),
                    "rain_cv": monthly_rain.std() / monthly_rain.mean() if monthly_rain.mean() else np.nan,
                    "rain_pre_season_mm": d.loc[pre_key, "rain_mm_day"] * days[pre_m] if pre_key in d.index else np.nan,
                    "temp_mean_c": s["temp_mean_c"].mean(),
                    "temp_max_c": s["temp_max_c"].mean(),
                    "temp_min_c": s["temp_min_c"].mean(),
                    "humidity_pct": s["humidity_pct"].mean(),
                    "solar_mj_m2_day": s["solar_mj_m2_day"].mean(),  # NaN before 1984
                    "wind_m_s": s["wind_m_s"].mean(),
                })
    return pd.DataFrame(rows)


def main():
    dcs = load_dcs()
    print(f"DCS rows parsed: {len(dcs):,}")

    weather = season_weather(pd.read_csv(WEATHER_CSV))
    print(f"Season-weather rows: {len(weather):,}")

    df = dcs.merge(weather, on=["district", "season", "year"], how="inner")

    # Target must exist; rows with no yield (e.g. war years in the North) are dropped.
    df = df[df["yield_total"].notna() & (df["yield_total"] > 0)].copy()

    # In the irrigation split, '-' next to reported values means "none of this
    # type" (e.g. Colombo has no major irrigation), so it becomes 0. Rows with
    # no split at all (before 1983/84) stay missing.
    for measure in ["sown", "harvested", "yield"]:
        cols = [f"{measure}_{irr}" for irr in ["major", "minor", "rainfed"]]
        has_split = df[cols].notna().any(axis=1)
        if measure != "yield":  # a missing yield is not a zero yield
            df.loc[has_split, cols] = df.loc[has_split, cols].fillna(0)

    # Engineered features
    df["sown_share_major"] = df["sown_major"] / df["sown_total"]
    df["sown_share_minor"] = df["sown_minor"] / df["sown_total"]
    df["sown_share_rainfed"] = df["sown_rainfed"] / df["sown_total"]
    df["harvest_ratio"] = df["harvested_total"] / df["sown_total"]  # EDA only (known after harvest)
    df["fertilizer_shock"] = [int((s, y) in FERTILIZER_SHOCK) for s, y in zip(df["season"], df["year"])]
    df["season_label"] = np.where(
        df["season"] == "Maha",
        (df["year"] - 1).astype(str) + "/" + df["year"].astype(str).str[-2:] + " Maha",
        df["year"].astype(str) + " Yala",
    )

    df = df.sort_values(["year", "season", "district"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {OUT_CSV}")


if __name__ == "__main__":
    main()
