
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve, KFold

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "images"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
for d in (IMG_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)


def evaluate_all(fitted_models, split, feature_names):
    X_train, X_test, y_train, y_test, scaler = split
    rows = []
    predictions = {}

    for name, (model, needs_scaling) in fitted_models.items():
        Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns,
                            index=X_test.index) if needs_scaling else X_test
        Xtr = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns,
                            index=X_train.index) if needs_scaling else X_train

        y_pred = model.predict(Xte)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv,
                                     scoring="neg_root_mean_squared_error")
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        rows.append({
            "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "CV_RMSE_mean": cv_rmse_mean, "CV_RMSE_std": cv_rmse_std,
        })
        print(f"[evaluate] {name}: RMSE={rmse:.2f}  R2={r2:.4f}  "
              f"CV_RMSE={cv_rmse_mean:.2f}±{cv_rmse_std:.2f}")

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # --- comparison bar chart ---
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["RMSE"], color="teal")
    plt.ylabel("Test RMSE (lower is better)")
    plt.title("Model Comparison — Test RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "08_model_comparison.png", dpi=150)
    plt.close()

    best_name = results_df.iloc[0]["Model"]
    best_model, best_needs_scaling = fitted_models[best_name]
    print(f"[evaluate] BEST MODEL: {best_name}")

    Xte_best = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns,
                             index=X_test.index) if best_needs_scaling else X_test
    y_pred_best = best_model.predict(Xte_best)

    # --- residual plot ---
    residuals = y_test.values - y_pred_best
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred_best, residuals, alpha=0.3, s=12)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted AQI")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title(f"Residual Plot — {best_name}")

    axes[1].hist(residuals, bins=40, color="salmon")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "09_residuals_best_model.png", dpi=150)
    plt.close()

    # --- learning curve ---
    Xtr_best = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns,
                             index=X_train.index) if best_needs_scaling else X_train
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, Xtr_best, y_train, cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1,
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, -train_scores.mean(axis=1), "o-", label="Training RMSE")
    plt.plot(train_sizes, -val_scores.mean(axis=1), "o-", label="Validation RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "10_learning_curve_best_model.png", dpi=150)
    plt.close()

    # --- feature importance (best model, if it exposes one) ---
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        importances[::-1].plot(kind="barh", color="darkgreen")
        plt.title(f"Top 15 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "11_feature_importance.png", dpi=150)
        plt.close()

    # --- save best model bundle ---
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "needs_scaling": best_needs_scaling,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": results_df.iloc[0].to_dict(),
    }
    joblib.dump(bundle, MODELS_DIR / "best_model.joblib")
    print(f"[evaluate] Saved best model bundle to models/best_model.joblib")

    return results_df, best_name


if __name__ == "__main__":
    import joblib as jl
    art = jl.load(MODELS_DIR / "training_artifacts.joblib")
    evaluate_all(art["models"], art["split"], art["feature_names"])
