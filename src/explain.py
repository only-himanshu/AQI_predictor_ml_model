
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "images"
MODELS_DIR = BASE_DIR / "models"

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def explain_best_model():
    bundle = joblib.load(MODELS_DIR / "best_model.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    needs_scaling = bundle["needs_scaling"]
    feature_names = bundle["feature_names"]
    model_name = bundle["model_name"]

    art = joblib.load(MODELS_DIR / "training_artifacts.joblib")
    X_train, X_test, y_train, y_test, _ = art["split"]

    X_explain = X_test.sample(min(300, len(X_test)), random_state=42)
    if needs_scaling:
        X_explain_input = pd.DataFrame(
            scaler.transform(X_explain), columns=X_explain.columns, index=X_explain.index
        )
    else:
        X_explain_input = X_explain

    if HAS_SHAP:
        print("[explain] Using SHAP TreeExplainer/Explainer")
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model, X_explain_input)
        shap_values = explainer(X_explain_input)

        plt.figure()
        shap.summary_plot(shap_values, X_explain_input, show=False)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "12_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[explain] Saved SHAP summary plot to images/12_shap_summary.png")
    else:
        print("[explain] `shap` not installed in this environment — using "
              "sklearn permutation_importance instead (install shap for full "
              "SHAP summary/waterfall plots).")
        result = permutation_importance(
            model, X_explain_input, y_test.loc[X_explain.index],
            n_repeats=10, random_state=42, n_jobs=-1,
        )
        importances = pd.Series(result.importances_mean, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)

        plt.figure(figsize=(9, 6))
        importances[::-1].plot(kind="barh", color="purple")
        plt.title(f"Permutation Importance (SHAP fallback) — {model_name}")
        plt.xlabel("Mean decrease in model score when feature is shuffled")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "12_permutation_importance.png", dpi=150)
        plt.close()
        print("[explain] Saved images/12_permutation_importance.png")


if __name__ == "__main__":
    explain_best_model()
