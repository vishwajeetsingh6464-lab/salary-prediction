"""
train_model.py — Train a Random Forest salary prediction model
and persist it along with the encoders to models/
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "salary_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

CATEGORICAL_COLS = ["Education", "Location", "Job_Title", "Gender"]
FEATURE_COLS = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
TARGET_COL = "Salary"


def train():
    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # ── Encode categoricals ──────────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  {col}: {list(le.classes_)}")

    # ── Split ────────────────────────────────────────────────────────────────
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train ────────────────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print(f"\nTest MAE  : ${mae:,.2f}")
    print(f"Test R²   : {r2:.4f}")
    print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Store feature metadata for the frontend/API
    meta = {
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "classes": {col: list(encoders[col].classes_) for col in CATEGORICAL_COLS},
        "metrics": {"mae": round(mae, 2), "r2": round(r2, 4)},
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved: models/model.pkl, models/encoders.pkl, models/meta.json")
    print("Training complete!")


if __name__ == "__main__":
    train()
