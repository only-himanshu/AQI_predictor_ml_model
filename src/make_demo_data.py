
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai",
    "Hyderabad", "Lucknow", "Patna", "Ahmedabad", "Jaipur",
    "Varanasi", "Pune",
]

# City-level pollution "baseline" multipliers (rough realism: Delhi/Patna/Lucknow
# tend to run higher PM levels than coastal metros like Chennai/Mumbai in CPCB data)
CITY_BASELINE = {
    "Delhi": 1.6, "Lucknow": 1.5, "Patna": 1.55, "Varanasi": 1.4,
    "Kolkata": 1.2, "Ahmedabad": 1.15, "Jaipur": 1.2, "Hyderabad": 0.95,
    "Mumbai": 0.9, "Pune": 0.85, "Bengaluru": 0.8, "Chennai": 0.85,
}

# ---------------------------------------------------------------------------
# Official CPCB National AQI breakpoints (sub-index calculation), 2014 standard
# Each entry: (C_low, C_high, I_low, I_high)
# ---------------------------------------------------------------------------
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10":  [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
              (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)],
    "SO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
              (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO":    [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
              (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "O3":    [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
              (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}


def sub_index(value: float, pollutant: str) -> float:
    """Compute the CPCB sub-index for one pollutant concentration."""
    bps = BREAKPOINTS[pollutant]
    value = max(value, 0)
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= value <= c_hi:
            return i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (value - c_lo)
    # above the top breakpoint -> extrapolate from the last band
    c_lo, c_hi, i_lo, i_hi = bps[-1]
    return i_hi + (value - c_hi) * (i_hi - i_lo) / (c_hi - c_lo)


def aqi_bucket(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def generate(n_days: int = 1460) -> pd.DataFrame:
    """Generate ~4 years of daily records per city (synthetic demo data)."""
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")
    rows = []

    for city in CITIES:
        base = CITY_BASELINE[city]
        for date in dates:
            month = date.month
            # crude seasonal factor: winter (Nov-Feb) higher pollution in India
            season_factor = 1.5 if month in (11, 12, 1, 2) else (
                0.75 if month in (6, 7, 8, 9) else 1.0)

            pm25 = max(5, RNG.normal(70, 25) * base * season_factor)
            pm10 = max(10, pm25 * RNG.uniform(1.3, 1.9))
            no2 = max(2, RNG.normal(35, 15) * base)
            so2 = max(1, RNG.normal(12, 6) * base)
            co = max(0.1, RNG.normal(1.2, 0.6) * base)
            o3 = max(2, RNG.normal(40, 20))
            nox = no2 * RNG.uniform(1.0, 1.4)
            nh3 = max(1, RNG.normal(20, 10))
            benzene = max(0, RNG.normal(2, 1.5))
            toluene = max(0, RNG.normal(8, 5))
            xylene = max(0, RNG.normal(3, 2))

            temp = RNG.normal(
                28 - (8 if month in (12, 1) else 0) + (6 if month in (4, 5, 6) else 0), 4)
            humidity = np.clip(RNG.normal(55 + (20 if month in (7, 8, 9) else 0), 12), 5, 100)
            wind_speed = max(0.2, RNG.normal(8, 3))
            pressure = RNG.normal(1010, 6)
            rainfall = max(0, RNG.normal(6 if month in (6, 7, 8, 9) else 0.3, 4))

            sub_idx = {
                "PM2.5": sub_index(pm25, "PM2.5"),
                "PM10": sub_index(pm10, "PM10"),
                "NO2": sub_index(no2, "NO2"),
                "SO2": sub_index(so2, "SO2"),
                "CO": sub_index(co, "CO"),
                "O3": sub_index(o3, "O3"),
            }
            aqi = max(sub_idx.values())  # CPCB rule: AQI = max sub-index

            rows.append({
                "City": city,
                "Date": date,
                "PM2.5": round(pm25, 2),
                "PM10": round(pm10, 2),
                "NO": round(no2 * RNG.uniform(0.3, 0.6), 2),
                "NO2": round(no2, 2),
                "NOx": round(nox, 2),
                "NH3": round(nh3, 2),
                "CO": round(co, 2),
                "SO2": round(so2, 2),
                "O3": round(o3, 2),
                "Benzene": round(benzene, 2),
                "Toluene": round(toluene, 2),
                "Xylene": round(xylene, 2),
                "Temperature": round(temp, 2),
                "Humidity": round(humidity, 2),
                "WindSpeed": round(wind_speed, 2),
                "Pressure": round(pressure, 2),
                "Rainfall": round(rainfall, 2),
                "AQI": round(aqi, 1),
                "AQI_Bucket": aqi_bucket(aqi),
            })

    df = pd.DataFrame(rows)

    # Inject realistic missingness (CPCB real data has plenty of NaNs too)
    for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "Xylene"]:
        mask = RNG.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    out_path = RAW_DIR / "city_day.csv"
    df.to_csv(out_path, index=False)
    print(f"Demo dataset written to {out_path} ({len(df)} rows, {df['City'].nunique()} cities)")
    print(df.head())
