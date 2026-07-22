import sys
from pathlib import Path
import os
import gdown

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aqi_utils import categorize_aqi 
from feature_engineering import SEASON_MAP  


MODEL_PATH = Path(__file__).resolve().parent / "best_model.joblib"


DRIVE_URL = 'https://drive.google.com/file/d/1KtWTr0_6jyQlFLFKvjZcjdU39KdnZzKX/view?usp=drive_link'

st.set_page_config(page_title="AQI Prediction System", page_icon="🌫️", layout="centered")


@st.cache_resource
def load_model_bundle():
  
    if not MODEL_PATH.exists():
        with st.spinner("Downloading your trained ML model from Google Drive... Please wait..."):
            try:
                gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Failed to download model from Google Drive. Error: {e}")
                return None
    return joblib.load(MODEL_PATH)


def month_to_season(month: int) -> str:
    return SEASON_MAP[month]


def build_input_row(inputs: dict, feature_names: list) -> pd.DataFrame:
    """Builds a single-row dataframe matching the training feature matrix,
    including one-hot columns, filling anything not provided with 0.
    Only fields actually present in feature_names affect the prediction —
    this keeps the UI honest about what the model actually uses."""
    row = {name: 0 for name in feature_names}

    numeric_map = {
        "PM2.5": inputs["pm25"], "PM10": inputs["pm10"], "NO2": inputs["no2"],
        "SO2": inputs["so2"], "CO": inputs["co"], "O3": inputs["o3"],
        "NH3": inputs.get("nh3", 15.0), "NOx": inputs.get("nox", inputs["no2"] * 1.2),
        "Month": inputs["month"],
    }
    if "Temperature" in row:
        numeric_map["Temperature"] = inputs["temp"]
    if "Humidity" in row:
        numeric_map["Humidity"] = inputs["humidity"]
    if "WindSpeed" in row:
        numeric_map["WindSpeed"] = inputs["wind"]
    if "Pressure" in row:
        numeric_map["Pressure"] = inputs["pressure"]
    if "Rainfall" in row:
        numeric_map["Rainfall"] = inputs["rainfall"]

    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    if "PM_ratio" in row:
        row["PM_ratio"] = inputs["pm25"] / inputs["pm10"] if inputs["pm10"] else 0
    if "Pollutant_Load" in row:
        row["Pollutant_Load"] = (
            0.4 * inputs["pm25"] + 0.25 * inputs["pm10"] + 0.15 * inputs["no2"]
            + 0.1 * inputs["so2"] + 0.1 * inputs["co"] * 10
        )

    city_col = f"City_{inputs['city']}"
    if city_col in row:
        row[city_col] = 1

    season = month_to_season(inputs["month"])
    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    return pd.DataFrame([row], columns=feature_names)


def main():
    st.title("🌫️ Climate Monitoring: ML-Based AQI Prediction System")
    st.caption(
        "Predicts India's National Air Quality Index (AQI) from pollutant "
        "and meteorological readings, using CPCB-schema data. "
        "Model: trained via src/train.py + src/evaluate.py."
    )

    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            "No trained model found or downloaded. "
            "Please ensure your Google Drive link is set to 'Anyone with the link'."
        )
        return

    model = bundle["model"]
    scaler = bundle["scaler"]
    needs_scaling = bundle["needs_scaling"]
    feature_names = bundle["feature_names"]
    model_name = bundle["model_name"]
    metrics = bundle.get("metrics", {})

    with st.sidebar:
        st.header("Model Info")
        st.write(f"**Best model:** {model_name}")
        if metrics:
            st.write(f"Test RMSE: `{metrics.get('RMSE', 0):.2f}`")
            st.write(f"Test R²: `{metrics.get('R2', 0):.3f}`")
        st.markdown("---")
        st.caption(
            "Data source (recommended): CPCB 'Air Quality Data in India' "
            "(data.gov.in / Kaggle: rohanrao/air-quality-data-in-india)."
        )

    cities = sorted([c.replace("City_", "") for c in feature_names if c.startswith("City_")])
    cities = ["Reference City"] + cities  # the drop_first=True baseline city

    has_weather = any(f in feature_names for f in
                       ["Temperature", "Humidity", "WindSpeed", "Pressure", "Rainfall"])

    st.subheader("Enter Readings")
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", cities)
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 999.0, 80.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 999.0, 140.0)
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 35.0)
        so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 12.0)
        co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.2)
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 40.0)

    temp = humidity = wind = pressure = rainfall = None
    with col2:
        month = st.selectbox("Month", list(range(1, 13)), index=0)
        if has_weather:
            temp = st.number_input("Temperature (°C)", -10.0, 55.0, 25.0)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 55.0)
            wind = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 8.0)
            pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1010.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)
        else:
            st.info(
                "This model was trained on data with no weather columns "
                "(the real CPCB/OpenAQ pollutant feed doesn't include "
                "weather), so weather inputs aren't shown here — the "
                "prediction is based on pollutants + city + month only."
            )

    if st.button("Predict AQI", type="primary"):
        inputs = dict(
            city=city, pm25=pm25, pm10=pm10, no2=no2, so2=so2, co=co, o3=o3,
            month=month,
            temp=temp if temp is not None else 25.0,
            humidity=humidity if humidity is not None else 55.0,
            wind=wind if wind is not None else 8.0,
            pressure=pressure if pressure is not None else 1010.0,
            rainfall=rainfall if rainfall is not None else 0.0,
        )
        X_input = build_input_row(inputs, feature_names)
        X_model = (
            pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)
            if needs_scaling else X_input
        )
        pred_aqi = float(model.predict(X_model)[0])
        pred_aqi = max(0.0, pred_aqi)
        category, color, advice = categorize_aqi(pred_aqi)

        st.markdown("---")
        st.markdown(
            f"""
            <div style="padding:24px;border-radius:12px;background-color:{color}22;
                        border:2px solid {color};text-align:center;">
                <h2 style="color:{color};margin:0;">Predicted AQI: {pred_aqi:.0f}</h2>
                <h3 style="color:{color};margin:6px 0;">{category}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Health Recommendation:** {advice}")

    st.markdown("---")
    with st.expander("About this project"):
        st.markdown(
            "Full pipeline: data cleaning → EDA → feature engineering → "
            "multi-model training & tuning → evaluation → explainability "
            "(SHAP/permutation importance) → this Streamlit interface. "
            "See `README.md` and `reports/report.md` for full details."
        )


if __name__ == "__main__":
    main()
