"""
eda.py
======
Generates the exploratory data analysis visualizations used in the
internship report. Saves all figures to images/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "images"
IMG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


def run_eda(df: pd.DataFrame, df_raw: pd.DataFrame = None):
    # 1. Missing values
    src_df = df_raw if df_raw is not None else df
    plt.figure(figsize=(10, 5))
    missing = src_df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        sns.barplot(x=missing.values, y=missing.index, color="steelblue")
        plt.title("Fraction of Missing Values per Column (raw data)")
        plt.xlabel("Fraction Missing")
    else:
        plt.text(0.5, 0.5, "No missing values", ha="center")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_missing_values.png", dpi=150)
    plt.close()

    # 2. AQI distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["AQI"], bins=50, kde=True, color="darkorange")
    plt.title("Distribution of AQI")
    plt.xlabel("AQI")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_aqi_distribution.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. City-wise AQI (boxplot)
    plt.figure(figsize=(12, 6))
    order = df.groupby("City")["AQI"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="City", y="AQI", order=order, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("City-wise AQI Distribution")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "04_citywise_aqi.png", dpi=150)
    plt.close()

    # 5. Seasonal trend
    if "Season" in df.columns:
        plt.figure(figsize=(8, 5))
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
        sns.boxplot(data=df, x="Season", y="AQI", order=season_order, palette="coolwarm")
        plt.title("Seasonal AQI Trend")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "05_seasonal_trend.png", dpi=150)
        plt.close()

    # 6. Pollutant trends over time (Delhi as example, since often has full data)
    example_city = df["City"].iloc[0]
    city_df = df[df["City"] == example_city].sort_values("Date")
    plt.figure(figsize=(12, 5))
    plt.plot(city_df["Date"], city_df["PM2.5"], label="PM2.5", alpha=0.7)
    plt.plot(city_df["Date"], city_df["PM10"], label="PM10", alpha=0.7)
    plt.plot(city_df["Date"], city_df["AQI"], label="AQI", alpha=0.7, linewidth=1.5)
    plt.legend()
    plt.title(f"Pollutant & AQI Trend Over Time — {example_city}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "06_pollutant_trend_timeseries.png", dpi=150)
    plt.close()

    # 7. Pairplot (sampled for speed) of key pollutants vs AQI
    sample = df.sample(min(1500, len(df)), random_state=42)
    pair_cols = ["PM2.5", "PM10", "NO2", "SO2", "AQI"]
    g = sns.pairplot(sample[pair_cols], diag_kind="kde", plot_kws={"alpha": 0.3, "s": 15})
    g.fig.suptitle("Pairplot of Key Pollutants vs AQI", y=1.02)
    g.savefig(IMG_DIR / "07_pairplot.png", dpi=150)
    plt.close("all")

    print(f"[eda] Saved 7 figures to {IMG_DIR}")


if __name__ == "__main__":
    from data_loader import load_raw
    from preprocessing import clean
    from feature_engineering import engineer

    raw = load_raw()
    df = engineer(clean(raw))
    run_eda(df, df_raw=raw)
