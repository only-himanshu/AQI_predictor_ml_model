import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_raw
from preprocessing import clean
from feature_engineering import engineer, build_model_matrix
from eda import run_eda
from train import train_all
from evaluate import evaluate_all
from explain import explain_best_model
import joblib

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    print("=" * 70)
    print("STEP 1-2: Load + Clean data")
    print("=" * 70)
    df_raw = load_raw()
    df_clean = clean(df_raw)

    print("\n" + "=" * 70)
    print("STEP 3: Feature engineering")
    print("=" * 70)
    df_feat = engineer(df_clean)
    df_feat.to_csv("data/processed/city_day_features.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 4: EDA")
    print("=" * 70)
    run_eda(df_feat, df_raw=df_raw)

    X, y, feature_names = build_model_matrix(df_feat)

    print("\n" + "=" * 70)
    print("STEP 5: Train + tune models")
    print("=" * 70)
    fitted_models, results, split = train_all(X, y)
    joblib.dump({"models": fitted_models, "split": split, "feature_names": feature_names},
                MODELS_DIR / "training_artifacts.joblib")

    print("\n" + "=" * 70)
    print("STEP 6: Evaluate + select best model")
    print("=" * 70)
    results_df, best_name = evaluate_all(fitted_models, split, feature_names)
    print(results_df)

    print("\n" + "=" * 70)
    print("STEP 7: Explainability")
    print("=" * 70)
    explain_best_model()

    print("\nPipeline complete. Run `streamlit run streamlit_app.py` to launch the app.")


if __name__ == "__main__":
    main()
