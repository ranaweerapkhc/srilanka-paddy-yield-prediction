"""
Fetch monthly weather for each paddy district from the NASA POWER API.

Source : NASA POWER (https://power.larc.nasa.gov), Agroclimatology community.
Output : data_raw/nasa_power/<district>.json   (raw API response, one per district)
         data_raw/nasa_power/weather_monthly.csv (tidy table: district, year, month, variables)

Run from the project root:
    python src/fetch_weather.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data_raw" / "nasa_power"

START_YEAR = 1981  # first year NASA POWER monthly data is available
END_YEAR = 2025

# Variables requested (all monthly averages unless noted)
PARAMETERS = {
    "PRECTOTCORR": "rain_mm_day",       # precipitation, mm/day
    "T2M": "temp_mean_c",               # mean air temperature at 2 m, °C
    "T2M_MAX": "temp_max_c",            # max air temperature, °C
    "T2M_MIN": "temp_min_c",            # min air temperature, °C
    "RH2M": "humidity_pct",             # relative humidity at 2 m, %
    "ALLSKY_SFC_SW_DWN": "solar_mj_m2_day",  # solar radiation, MJ/m²/day (from 1984)
    "WS2M": "wind_m_s",                 # wind speed at 2 m, m/s
}

# Approximate centre of the main paddy-growing area of each district (lat, lon).
# NASA POWER cells are ~0.5° x 0.625°, so neighbouring small districts may share a cell.
DISTRICTS = {
    "COLOMBO": (6.88, 80.02),
    "GAMPAHA": (7.09, 80.01),
    "KALUTARA": (6.60, 80.20),
    "KANDY": (7.29, 80.63),
    "MATALE": (7.47, 80.62),
    "NUWARA ELIYA": (6.97, 80.77),
    "GALLE": (6.15, 80.30),
    "MATARA": (6.05, 80.55),
    "HAMBANTOTA": (6.20, 81.05),
    "JAFFNA": (9.70, 80.05),
    "MANNAR": (8.85, 80.05),
    "VAVUNIYA": (8.75, 80.50),
    "MULLATIVU": (9.15, 80.65),
    "KILLINOCHCHI": (9.40, 80.40),
    "BATTICALOA": (7.75, 81.60),
    "AMPARA": (7.30, 81.65),
    "TRINCOMALEE": (8.45, 81.10),
    "KURUNEGALA": (7.60, 80.25),
    "PUTTALAM": (8.05, 79.95),
    "ANURADHAPURA": (8.30, 80.45),
    "POLONNARUWA": (7.95, 81.00),
    "BADULLA": (7.00, 81.05),
    "MONARAGALA": (6.85, 81.35),
    "RATNAPURA": (6.65, 80.45),
    "KEGALLE": (7.25, 80.35),
    "UDAWALAWE": (6.45, 80.90),
    "MAHAWELI H": (8.15, 80.40),
}

API_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_district(name, lat, lon, retries=3):
    """Download (or reuse a cached copy of) one district's monthly data."""
    cache = OUT_DIR / f"{name.replace(' ', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": START_YEAR,
        "end": END_YEAR,
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            cache.write_text(json.dumps(data))
            return data
        except requests.RequestException as err:
            print(f"  {name}: attempt {attempt} failed ({err})")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {name}")


def to_rows(name, data):
    """Turn the API's {param: {'YYYYMM': value}} layout into tidy rows."""
    series = data["properties"]["parameter"]
    rows = {}
    for param, values in series.items():
        for key, value in values.items():
            month = int(key[4:])
            if month == 13:  # annual average, not needed
                continue
            row = rows.setdefault(key, {"district": name, "year": int(key[:4]), "month": month})
            row[PARAMETERS[param]] = None if value == -999 else value  # -999 = missing
    return list(rows.values())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, (lat, lon) in DISTRICTS.items():
        print(f"Fetching {name} ({lat}, {lon})")
        data = fetch_district(name, lat, lon)
        all_rows.extend(to_rows(name, data))
        time.sleep(1)  # be polite to the API

    df = pd.DataFrame(all_rows).sort_values(["district", "year", "month"])
    out = OUT_DIR / "weather_monthly.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
