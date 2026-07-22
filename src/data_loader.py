"""
data_loader.py
==============
Loads the CPCB "city_day.csv" air-quality dataset.

Real data source (recommended):
    Kaggle: rohanrao/air-quality-data-in-india -> city_day.csv
    Original: CPCB / data.gov.in

Place the real file at data/raw/city_day.csv and this loader works with
zero changes. If that file is missing, it falls back to the synthetic demo
data generator (src/make_demo_data.py) purely so the pipeline is runnable.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "city_day.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CPCB-schema CSV. Generates demo data if not present."""
    if not path.exists():
        print(f"[data_loader] {path} not found — generating demo dataset instead.")
        from make_demo_data import generate
        df = generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    df = pd.read_csv(path, parse_dates=["Date"])
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
