"""
train.py
========
Trains and tunes multiple regression models to predict AQI, then saves the
best one to models/best_model.joblib.

Models attempted:
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting (sklearn)
    - XGBoost        (optional — used automatically if installed)
    - LightGBM       (optional — used automatically if installed)
    - CatBoost       (optional — used automatically if installed)

If a library isn't installed in the current environment, it is skipped with
a printed notice rather than crashing — this keeps the script portable
between the restricted demo sandbox and a full local/production environment
(`pip install -r requirements.txt` enables all of them).
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_model_grid():
    """Returns dict of {name: (estimator, param_grid, needs_scaling)}."""
    grids = {
        "LinearRegression": (
            LinearRegression(), {}, True,
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [6, 10, 16, None], "min_samples_leaf": [1, 5, 10]},
            False,
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200], "max_depth": [16, None]},
            False,
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [150], "learning_rate": [0.05, 0.1], "max_depth": [3]},
            False,
        ),
    }

    if HAS_XGB:
        grids["XGBoost"] = (
            xgb.XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror"),
            {"n_estimators": [300], "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
            False,
        )
    else:
        print("[train] xgboost not installed in this environment — skipping "
              "(will run automatically once you `pip install xgboost`).")

    if HAS_LGBM:
        grids["LightGBM"] = (
            lgb.LGBMRegressor(random_state=42),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
            False,
        )

    if HAS_CATBOOST:
        grids["CatBoost"] = (
            CatBoostRegressor(random_state=42, verbose=0),
            {"iterations": [300], "learning_rate": [0.05, 0.1]},
            False,
        )

    return grids


def train_all(X: pd.DataFrame, y: pd.Series, cv_folds: int = 3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    grids = get_model_grid()
    results = {}
    fitted_models = {}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, (estimator, param_grid, needs_scaling) in grids.items():
        t0 = time.time()
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        if param_grid:
            search = GridSearchCV(
                estimator, param_grid, cv=cv,
                scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            search.fit(Xtr, y_train)
            best_est = search.best_estimator_
            best_params = search.best_params_
        else:
            best_est = estimator.fit(Xtr, y_train)
            best_params = {}

        elapsed = time.time() - t0
        fitted_models[name] = (best_est, needs_scaling)
        results[name] = {"best_params": best_params, "train_time_sec": round(elapsed, 2)}
        print(f"[train] {name} tuned in {elapsed:.1f}s — best params: {best_params}")

    return fitted_models, results, (X_train, X_test, y_train, y_test, scaler)


if __name__ == "__main__":
    from data_loader import load_raw
    from preprocessing import clean
    from feature_engineering import engineer, build_model_matrix

    df = engineer(clean(load_raw()))
    X, y, feature_names = build_model_matrix(df)

    fitted_models, results, split = train_all(X, y)

    joblib.dump({"models": fitted_models, "split": split, "feature_names": feature_names},
                MODELS_DIR / "training_artifacts.joblib")

    with open(MODELS_DIR / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved training artifacts to models/training_artifacts.joblib")
