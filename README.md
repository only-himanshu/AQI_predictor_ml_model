# Climate Monitoring: ML-Based Air Quality Prediction System

An end-to-end machine learning system that predicts India's National Air
Quality Index (AQI) from pollutant concentrations and meteorological
readings, with a Streamlit interface for interactive prediction.

---

## Important note on data

This project is built against the schema of a real, publicly available
CPCB dataset:

- **Dataset:** *Air Quality Data in India (2015–2020)*
- **Official source:** Central Pollution Control Board (CPCB), via
  [data.gov.in](https://data.gov.in)
- **Convenient mirror:** Kaggle —
  [`rohanrao/air-quality-data-in-india`](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
  → file `city_day.csv`
- **Schema:** `City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
  Benzene, Toluene, Xylene, AQI, AQI_Bucket`

**To run this project on the real dataset:** download `city_day.csv` from
the link above and place it at `data/raw/city_day.csv`. Nothing else needs
to change — `src/data_loader.py` reads that exact schema.

If that file is absent, `src/make_demo_data.py` generates a synthetic
stand-in with the identical schema (clearly labeled as synthetic in the
code's docstring) purely so the whole pipeline is runnable and demonstrable
without internet access. The demo data's AQI values are computed using the
real, official **CPCB National AQI sub-index / breakpoint formula**, so the
AQI ↔ category relationship is methodologically authentic even though the
underlying pollutant readings in the demo file are simulated. **Cite the
real CPCB/Kaggle source in your internship report**, and swap in the real
CSV before final submission if authenticity of the raw readings matters for
grading.

---

## Project Structure

```
aqi_project/
├── data/
│   ├── raw/city_day.csv           # raw CPCB data (real or demo)
│   └── processed/                 # cleaned + feature-engineered data
├── notebooks/                     # (optional) exploratory notebooks
├── src/
│   ├── data_loader.py             # loads raw CSV
│   ├── preprocessing.py           # missing values, outliers, dedup
│   ├── feature_engineering.py     # derived features, encoding, leakage guard
│   ├── eda.py                     # generates all EDA plots
│   ├── train.py                   # trains + tunes 4–7 model families
│   ├── evaluate.py                # metrics, comparison, best-model selection
│   ├── explain.py                 # SHAP (or permutation importance fallback)
│   ├── aqi_utils.py                # AQI category / health-advice mapping
│   └── make_demo_data.py          # synthetic demo-data generator
├── models/
│   ├── training_artifacts.joblib  # all fitted models + train/test split
│   ├── tuning_results.json        # best hyperparameters per model
│   └── best_model.joblib          # final chosen model bundle
├── images/                        # all report-ready figures (01–12)
├── reports/
│   ├── model_comparison.csv       # metrics table for every model
│   └── report.md                  # full internship report
├── streamlit_app.py                # prediction web app
├── run_pipeline.py                 # runs the entire pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate     
pip install -r requirements.txt
```

## Running the pipeline

```bash
python run_pipeline.py
```

This runs, in order: data loading → cleaning → feature engineering → EDA →
model training & hyperparameter tuning → evaluation & model selection →
explainability. All figures land in `images/`, the trained model in
`models/best_model.joblib`, and metrics in `reports/model_comparison.csv`.

You can also run each stage independently, e.g.:

```bash
python src/data_loader.py
python src/preprocessing.py
python src/feature_engineering.py
python src/eda.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

## Running the Streamlit app

```bash
streamlit run streamlit_app.py
```

Enter pollutant (PM2.5, PM10, NO2, SO2, CO, O3) and meteorological
(temperature, humidity, wind speed, pressure, rainfall) readings, plus city
and month, to get:
- Predicted AQI
- AQI category (Good / Satisfactory / Moderate / Poor / Very Poor / Severe)
- A color indicator matching the official CPCB scale
- A health recommendation

---

## Models trained

| Model              | Notes |
|---------------------|-------|
| Linear Regression    | baseline |
| Decision Tree         | tuned via grid search |
| Random Forest         | tuned via grid search |
| Gradient Boosting     | tuned via grid search |
| XGBoost               | used automatically if `xgboost` is installed |
| LightGBM              | used automatically if `lightgbm` is installed |
| CatBoost              | used automatically if `catboost` is installed |

`src/train.py` detects which of XGBoost/LightGBM/CatBoost are installed and
skips missing ones gracefully, so the script is portable across
environments. All models are compared on Test MAE / MSE / RMSE / R² and
5-fold cross-validated RMSE; the model with the lowest test RMSE is saved
as the production model.

## Results (on the demo run in this sandbox — see reports/model_comparison.csv for exact numbers)

The best-performing model in this run was **Gradient Boosting** with a test
RMSE of ~16 AQI points and R² ≈ 0.97. Re-run `run_pipeline.py` on the real
CPCB CSV to get numbers for your actual submission — with a full
`pip install -r requirements.txt` (adding XGBoost/LightGBM/CatBoost/SHAP),
expect the gradient-boosted variants to remain competitive or improve
further, and the SHAP summary plot (`images/12_shap_summary.png`) to
replace the permutation-importance fallback produced in this sandbox.

---

## Data leakage prevention

- `AQI_Bucket` (a category derived directly from `AQI`) is dropped before
  building the model matrix — using it as a feature would leak the target.
- Train/test split is done before scaling, and the scaler is fit only on
  the training fold.
- Missing-value imputation uses city + calendar-month medians (not the
  specific date), avoiding any use of future information.

## Future Work

- Incorporate satellite-derived aerosol optical depth (AOD) as an
  additional feature for cities/times without ground-station coverage.
- Add LSTM / temporal models to capture day-to-day autocorrelation in
  pollution levels.
- Deploy the Streamlit app with a scheduled job that pulls the latest CPCB
  station readings via the [OpenAQ API](https://openaq.org) for live
  predictions.
- Add per-city model variants, since pollution dynamics differ
  substantially between coastal and inland/landlocked cities.
