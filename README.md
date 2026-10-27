# Sri Lanka Paddy Yield Prediction

Predicting district-level paddy yield (kg/ha) for the Maha and Yala seasons in Sri Lanka using an
**Explainable Boosting Machine (EBM)**, benchmarked against Random Forest and XGBoost, and explained
with SHAP and LIME.

MSc in AI — Applied Machine Learning assignment (Student ID: 269180E).

## Problem
Paddy is Sri Lanka's staple crop. Yield varies widely by district and season with rainfall,
temperature and policy shocks (e.g. the 2021 chemical fertilizer ban). A model that predicts
yield from seasonal weather and cultivation data can support planning for food security,
irrigation and fertilizer distribution.

## Data sources
| Data | Source |
|---|---|
| Sown/harvested extent, average yield, production by district and season | Department of Census and Statistics (DCS), Paddy Statistics |
| Seasonal weather (temperature, humidity, solar radiation, rainfall) | NASA POWER API |
| Rainfall (optional cross-check) | CHIRPS |

All data are public, aggregated at district level and contain no personal information.

## Project structure
```
data_raw/          original downloaded files (unchanged)
data_processed/    cleaned, merged dataset used for modelling
notebooks/         01_eda, 02_modelling, 03_explainability
src/               data collection and preprocessing scripts
models/            saved trained models
reports/figures/   plots used in the report
app/               Streamlit front-end
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app/app.py
```
