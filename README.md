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

To use the notebooks in Jupyter or VS Code, register the venv as a kernel and select **"Python (paddy venv)"**:
```bash
python -m ipykernel install --user --name paddy-venv --display-name "Python (paddy venv)"
```

## Reproduce the pipeline
```bash
python src/fetch_weather.py      # NASA POWER monthly weather -> data_raw/nasa_power/
python src/build_dataset.py      # DCS + weather -> data_processed/paddy_dataset.csv
```
Then run, in this order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_modelling.ipynb`, which saves the models to `models/`
3. `notebooks/03_explainability.ipynb`

## Results (test set: 2021–2025, unseen)
| Model | RMSE, normal seasons (kg/ha) | R² |
|---|---|---|
| Naive "same as last year" | 625 | 0.52 |
| Ridge | 458 | 0.74 |
| Random Forest | 451 | 0.75 |
| XGBoost | 502 | 0.69 |
| **EBM** | **492** | **0.70** |

No model predicts the 2021/22 fertilizer-ban seasons (RMSE ≈ 1,400–1,500), because the shock has no precedent in the training data.

## Run the app
Run this from the project root, so that `.streamlit/config.toml` (the light theme) is picked up:
```bash
streamlit run app/app.py
```

The app works like this:
1. Pick a district, a season and a harvest year.
   - A past season is pre-filled with its real recorded values and shows the actual yield.
   - A future season is pre-filled with recent typical weather and the latest yields.
2. Edit the cultivation, weather and recent-yield inputs.
3. The app shows:
   - the EBM prediction, with its typical error range
   - a **contribution waterfall** that splits the prediction exactly into one contribution per input
4. The **What the model learned** tab shows the global importance and each input's shape function.
