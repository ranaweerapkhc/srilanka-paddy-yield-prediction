"""
Sri Lanka Paddy Yield Predictor — Streamlit front end for the trained EBM.

Run from the project root:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, NUMERIC, TARGET, ZONE_OF, load_dataset, season_index  # noqa: E402

# Test RMSE of the EBM on normal (non-shock) seasons 2021-2025, from notebook 02
EBM_TEST_RMSE = 492

LABEL = {"district": "District", "season": "Season", "year": "Year (trend)", "log_sown_total": "Sown area",
         "sown_share_major": "Major irrigation share", "sown_share_minor": "Minor irrigation share",
         "sown_share_rainfed": "Rainfed share", "rain_total_mm": "Season rainfall",
         "rain_min_month_mm": "Driest month rain", "rain_max_month_mm": "Wettest month rain",
         "rain_cv": "Rainfall unevenness", "rain_pre_season_mm": "Pre-season rain",
         "temp_mean_c": "Mean temperature", "temp_max_c": "Max temperature", "temp_min_c": "Min temperature",
         "humidity_pct": "Humidity", "solar_mj_m2_day": "Solar radiation", "wind_m_s": "Wind speed",
         "lag_prev_season": "Previous season yield", "lag_same_season": "Same season last year",
         "roll3_same_season": "3-yr same-season average"}

BLUE, ORANGE, RED, INK, INK2, GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e4e3df"

st.set_page_config(page_title="Paddy Yield Predictor", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    ebm = joblib.load(ROOT / "models" / "ebm.joblib")
    rf = joblib.load(ROOT / "models" / "random_forest.joblib")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    assert meta["features"] == FEATURES, "Saved model was trained with a different feature list"
    return ebm, rf


@st.cache_data
def load_data():
    return load_dataset(ROOT)


ebm, rf = load_models()
df = load_data()
yield_at = df.set_index(["district", "t"])[TARGET]


def past_yield(district, t):
    v = yield_at.get((district, t), np.nan)
    return None if pd.isna(v) else float(v)


def defaults_for(district, season, year):
    """Default inputs for a season: actual values if it exists, else recent climatology and latest yields."""
    row = df[(df.district == district) & (df.season == season) & (df.year == year)]
    if len(row):
        r = row.iloc[0]
        return {f: r[f] for f in NUMERIC} | {"sown_total": r["sown_total"]}, float(r[TARGET])

    hist = df[(df.district == district) & (df.season == season)].sort_values("year").tail(10)
    d = {f: float(hist[f].median()) for f in NUMERIC if f in hist and hist[f].notna().any()}
    d["sown_total"] = float(hist["sown_total"].iloc[-1])
    t = int(season_index(year, season))
    d["lag_prev_season"] = past_yield(district, t - 1)
    d["lag_same_season"] = past_yield(district, t - 2)
    past3 = [v for v in (past_yield(district, t - k) for k in (2, 4, 6)) if v is not None]
    d["roll3_same_season"] = float(np.mean(past3)) if len(past3) >= 2 else None
    return d, None


# ---------------------------------------------------------------- header
st.title("🌾 Sri Lanka Paddy Yield Predictor")
st.caption("Predicts district-level paddy yield (kg/ha) for a Maha or Yala season with an "
           "**Explainable Boosting Machine**, and shows exactly how each input changed the prediction. "
           "Trained on Department of Census and Statistics paddy data and NASA POWER weather, 1981–2020.")

tab_predict, tab_model, tab_about = st.tabs(["Predict", "What the model learned", "About"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    c1, c2, c3 = st.columns([2, 1, 1])
    districts = sorted(df.district.unique())
    district = c1.selectbox("District", districts, index=districts.index("POLONNARUWA"))
    season = c2.radio("Season", ["Maha", "Yala"], horizontal=True)
    year = c3.number_input("Harvest year", min_value=1985, max_value=2030, value=2025, step=1,
                           help="For Maha, the harvest year: 2024/25 Maha → 2025.")

    defaults, actual = defaults_for(district, season, int(year))
    key = f"{district}|{season}|{year}"  # resets the inputs when the selection changes
    zone = ZONE_OF.get(district, "?")
    label = f"{int(year) - 1}/{str(int(year))[-2:]} Maha" if season == "Maha" else f"{int(year)} Yala"
    if actual is not None:
        st.info(f"**{district.title()} {label}** is in the dataset ({zone} zone). The inputs below are the real "
                f"recorded values, and the actual yield was **{actual:,.0f} kg/ha**. Change any input to explore what-if scenarios.")
    else:
        st.info(f"**{district.title()} {label}** is not in the dataset ({zone} zone). The inputs are pre-filled with the "
                "district's recent typical weather and its latest recorded yields. Edit them for your scenario.")

    def num(col, label, feat, lo, hi, step, fmt="%.0f", default=None, help=None):
        v = defaults.get(feat) if default is None else default
        v = float(np.clip(v if v is not None and not pd.isna(v) else (lo + hi) / 2, lo, hi))
        return col.number_input(label, lo, hi, v, step, format=fmt, key=f"{key}|{feat}", help=help)

    with st.form("inputs"):
        st.subheader("1 · Cultivation")
        a1, a2, a3 = st.columns(3)
        sown = num(a1, "Sown area (ha)", "sown_total", 10.0, 200000.0, 100.0)
        major = a2.slider("Major irrigation (%)", 0, 100, int(round(100 * (defaults.get("sown_share_major") or 0))),
                          key=f"{key}|major")
        minor = a3.slider("Minor irrigation (%)", 0, 100, int(round(100 * (defaults.get("sown_share_minor") or 0))),
                          key=f"{key}|minor")

        st.subheader("2 · Season weather")
        w = st.columns(4)
        rain = num(w[0], "Season rainfall (mm)", "rain_total_mm", 0.0, 3000.0, 10.0,
                   help="Maha: Oct–Feb, Yala: Apr–Aug")
        rmax = num(w[1], "Wettest month rain (mm)", "rain_max_month_mm", 0.0, 1500.0, 10.0)
        rmin = num(w[2], "Driest month rain (mm)", "rain_min_month_mm", 0.0, 500.0, 5.0)
        rpre = num(w[3], "Pre-season month rain (mm)", "rain_pre_season_mm", 0.0, 800.0, 5.0,
                   help="Maha: September, Yala: March")
        w = st.columns(4)
        tmean = num(w[0], "Mean temp (°C)", "temp_mean_c", 15.0, 35.0, 0.1, "%.1f")
        tmax = num(w[1], "Max temp (°C)", "temp_max_c", 18.0, 42.0, 0.1, "%.1f")
        tmin = num(w[2], "Min temp (°C)", "temp_min_c", 10.0, 30.0, 0.1, "%.1f")
        hum = num(w[3], "Humidity (%)", "humidity_pct", 40.0, 100.0, 0.5, "%.1f")
        w = st.columns(4)
        solar = num(w[0], "Solar radiation (MJ/m²/day)", "solar_mj_m2_day", 10.0, 28.0, 0.1, "%.1f")
        wind = num(w[1], "Wind (m/s)", "wind_m_s", 0.0, 10.0, 0.1, "%.1f")
        cv = num(w[2], "Rainfall unevenness (CV)", "rain_cv", 0.0, 3.0, 0.05, "%.2f",
                 help="Standard deviation ÷ mean of the monthly rainfall. Higher means the rain was more uneven.")

        st.subheader("3 · Recent yields (kg/ha)")
        p = st.columns(3)
        lag_prev = num(p[0], "Previous season yield", "lag_prev_season", 500.0, 9000.0, 10.0)
        lag_same = num(p[1], "Same season last year", "lag_same_season", 500.0, 9000.0, 10.0)
        roll3 = num(p[2], "3-yr same-season average", "roll3_same_season", 500.0, 9000.0, 10.0)

        submitted = st.form_submit_button("Predict yield", type="primary", width="stretch")

    if major + minor > 100:
        st.error(f"Major + minor irrigation cannot exceed 100%. Minor irrigation was reduced to {100 - major}% for this "
                 "prediction. The rest of the area is counted as rainfed.")
        minor = 100 - major

    x = pd.DataFrame([{
        "district": district, "season": season, "year": int(year), "log_sown_total": np.log1p(sown),
        "sown_share_major": major / 100, "sown_share_minor": minor / 100, "sown_share_rainfed": (100 - major - minor) / 100,
        "rain_total_mm": rain, "rain_min_month_mm": rmin, "rain_max_month_mm": rmax, "rain_cv": cv,
        "rain_pre_season_mm": rpre, "temp_mean_c": tmean, "temp_max_c": tmax, "temp_min_c": tmin,
        "humidity_pct": hum, "solar_mj_m2_day": solar, "wind_m_s": wind,
        "lag_prev_season": lag_prev, "lag_same_season": lag_same, "roll3_same_season": roll3,
    }])[FEATURES]

    pred = float(ebm.predict(x)[0])
    loc = ebm.explain_local(x).data(0)
    base = float(loc["extra"]["scores"][0])
    contrib = pd.Series(loc["scores"], index=loc["names"])

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted yield (EBM)", f"{pred:,.0f} kg/ha", f"{pred - roll3:+,.0f} vs 3-yr average")
    m2.metric("Typical error range", f"{pred - EBM_TEST_RMSE:,.0f} – {pred + EBM_TEST_RMSE:,.0f}",
              help="±1 RMSE of the EBM on unseen normal seasons, 2021–2025")
    m3.metric("Estimated production", f"{pred * sown / 1000:,.0f} t",
              help="Predicted yield × sown area. Assumes the whole sown area is harvested.")
    if actual is not None:
        m4.metric("Actual recorded yield", f"{actual:,.0f} kg/ha", f"model error {pred - actual:+,.0f}", delta_color="off")
    else:
        m4.metric("Random Forest (comparison)", f"{float(rf.predict(x)[0]):,.0f} kg/ha")

    st.subheader("Why this prediction?")
    st.caption("EBM is additive: **prediction = average yield + the sum of one contribution per input**. Blue bars pushed "
               "the yield up and orange bars pulled it down. The 10 largest contributions are shown and the rest are grouped.")

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    shown, rest = top.iloc[:10], top.iloc[10:].sum()
    items = list(shown.items()) + [("Other inputs", rest)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    pos = base
    for i, (name, v) in enumerate(items):
        ax.barh(i, v, left=pos, color=BLUE if v >= 0 else ORANGE, height=0.62)
        ax.text(pos + v + (12 if v >= 0 else -12), i, f"{v:+,.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        pos += v
    ax.set_yticks(range(len(items)), [LABEL.get(n, n) for n, _ in items])
    ax.set_ylim(len(items) + 0.2, -1.3)  # inverted, with room for labels above and below
    ax.axvline(base, color=INK2, lw=0.9, ls=":")
    ax.axvline(pred, color=INK, lw=1.4)
    ax.text(base, -0.9, f"average {base:,.0f}", ha="center", fontsize=8, color=INK2)
    ax.text(pred, len(items) - 0.55, f"prediction {pred:,.0f}", ha="center", va="top", fontsize=9,
            color=INK, fontweight="bold")
    if actual is not None:
        ax.axvline(actual, color=RED, lw=1.5, ls="--")
        ax.text(actual, -0.9, f"actual {actual:,.0f}", ha="center", fontsize=8, color=RED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Yield (kg/ha)")
    st.pyplot(fig, width="stretch")

    def fmt_value(name, v):
        if isinstance(v, str):
            return v.title()
        if name == "log_sown_total":
            return f"{np.expm1(v):,.0f} ha"
        if name.startswith("sown_share"):
            return f"{v:.0%}"
        return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"

    with st.expander("All contributions (table)"):
        st.dataframe(pd.DataFrame({"Input": [LABEL.get(n, n) for n in contrib.index],
                                   "Value": [fmt_value(n, x.iloc[0][n]) for n in contrib.index],
                                   "Contribution (kg/ha)": contrib.values.round(0)})
                     .sort_values("Contribution (kg/ha)", key=abs, ascending=False),
                     hide_index=True, width="stretch")

    st.warning("**Limitations.** The model assumes a *normal* season. It cannot anticipate policy or input shocks: during the "
               "2021 chemical fertilizer ban it over-predicted by about 1,350 kg/ha. The weather comes from a ~55 km grid, and "
               "predictions for the northern districts (shorter records) are less reliable. Use the prediction as decision "
               "support, not as a final figure.")

# ---------------------------------------------------------------- model tab
with tab_model:
    st.subheader("Global feature importance")
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([LABEL[n] for n in imp.index], imp.values, color=BLUE, height=0.65)
    ax.set_xlabel("Mean |contribution| (kg/ha)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    st.pyplot(fig, width="stretch")

    st.subheader("How one input changes the yield")
    g = ebm.explain_global()
    numeric_terms = [n for n in ebm.term_names_ if n in NUMERIC]
    feat = st.selectbox("Input", numeric_terms, index=numeric_terms.index("sown_share_rainfed"),
                        format_func=lambda n: LABEL[n])
    d = g.data(ebm.term_names_.index(feat))
    edges, s = np.asarray(d["names"], dtype=float), np.asarray(d["scores"])
    lo, hi = np.asarray(d["lower_bounds"]), np.asarray(d["upper_bounds"])
    xs = np.repeat(edges, 2)[1:-1]
    vals = df[df.year <= 2020][feat].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(xs, np.repeat(lo, 2), np.repeat(hi, 2), color=BLUE, alpha=0.18, lw=0)
    ax.plot(xs, np.repeat(s, 2), color=BLUE, lw=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(vals.quantile(0.005), vals.quantile(0.995))
    ax.set_xlabel(LABEL[feat])
    ax.set_ylabel("Effect on yield (kg/ha)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    st.pyplot(fig, width="stretch")
    st.caption("The line is the EBM shape function, and the shaded band is its uncertainty. Above 0 means the input raises the "
               "predicted yield, and below 0 means it lowers it.")

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(f"""
**Problem.** Paddy is Sri Lanka's staple crop. Seasonal district yields vary with irrigation, weather and past performance.
Earlier yield estimates support rice stock, import and fertilizer planning.

**Data.**
- Department of Census and Statistics paddy statistics: yield, sown area and irrigation split by district and season, 1981–2025.
- NASA POWER monthly weather (rainfall, temperature, humidity, solar radiation, wind), summed or averaged over each season.

**Model.** An **Explainable Boosting Machine** (InterpretML), a glass-box additive model with one learned curve per input.
It was tuned on 2018–2020 and tested on 2021–2025.

| Test 2021–2025 (normal seasons) | RMSE | R² |
|---|---|---|
| EBM (this app) | {EBM_TEST_RMSE} kg/ha | 0.70 |
| Random Forest | 451 kg/ha | 0.75 |
| Naive "same as last year" | 625 kg/ha | 0.52 |

**Ethics.** Only public, district-level government statistics are used. There is no personal or farm-level data.
""")
