
import numpy as np
import pandas as pd

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]
WEATHER_COLS = ["Temperature", "Humidity", "WindSpeed", "Pressure", "Rainfall"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop rows with no target
    before = len(df)
    df = df.dropna(subset=["AQI"])
    print(f"[preprocessing] dropped {before - len(df)} rows with missing AQI")

    # 2. Drop exact duplicates
    df = df.drop_duplicates(subset=["City", "Date"])

    # 3. Impute pollutant/weather missing values via city+month median
    df["Month"] = df["Date"].dt.month
    all_num_cols = [c for c in POLLUTANT_COLS + WEATHER_COLS if c in df.columns]

    for col in all_num_cols:
        df[col] = df[col].astype(float)
        group_median = df.groupby(["City", "Month"])[col].transform("median")
        df[col] = df[col].fillna(group_median)
        df[col] = df[col].fillna(df[col].median())

    # 4. Outlier clipping (IQR rule), applied per column
    for col in all_num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 3 * iqr, q3 + 3 * iqr
        n_clipped = ((df[col] < low) | (df[col] > high)).sum()
        df[col] = df[col].clip(lower=max(low, 0), upper=high)
        if n_clipped:
            print(f"[preprocessing] clipped {n_clipped} outliers in {col}")

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    from data_loader import load_raw
    raw = load_raw()
    cleaned = clean(raw)
    out_path = "data/processed/city_day_clean.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path}, shape={cleaned.shape}")
