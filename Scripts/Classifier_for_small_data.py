#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train baseline classifiers on the Small dataset (AirQualityUCI).

Input:
  Data/Small/Preprocessed_dataset/prepared_for_models.csv

Output (created if missing):
  Data/Small/Models_Baseline/
    - metrics.json
    - predictions_<MODEL>.csv
    - model_<MODEL>.pkl
    - class_distribution.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

# Optional: XGBoost (if installed)
HAS_XGB = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- CONFIG -----------------
INPUT_PATH = Path("Data\Small\Preprocessed_dataset\Standerized_preprocessed_data.csv")
OUT_DIR = Path("Data\Small")
RANDOM_STATE = 42
SPLIT_FRAC = 0.8   # chronological: first 80% train, last 20% test
SPLIT_MODE = "chronological"  # or "stratified"

# ----------------- UTILS -----------------
def load_features(input_path: Path):
    df = pd.read_csv(input_path)
    # Try to sort chronologically if a datetime column exists
    dt_col = None
    for c in df.columns:
        if c.lower() == "datetime":
            dt_col = c
            break
    if dt_col is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        df = df.sort_values(dt_col)
    # Ensure label exists
    if "label" not in df.columns:
        raise ValueError("Column 'label' was not found in prepared_for_models.csv.")
    # Separate X/y; drop datetime if present
    drop_cols = ["label"]
    if dt_col is not None:
        drop_cols.append(dt_col)
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)
    return df, X, y, dt_col

def chrono_split(X: pd.DataFrame, y: pd.Series, frac: float):
    n = len(X)
    idx = int(np.floor(n * frac))
    return X.iloc[:idx].values, X.iloc[idx:].values, y.iloc[:idx].values, y.iloc[idx:].values

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    return train_test_split(
        X.values, y.values,
        test_size=(1 - test_size),
        random_state=random_state,
        stratify=y.values
    )

def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# ----------------- MAIN -----------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_PATH}")
    df_all, X, y, dt_col = load_features(INPUT_PATH)

    # Class distribution
    class_counts = y.value_counts().sort_index().to_dict()
    with open(OUT_DIR / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump({"counts": class_counts}, f, indent=2)
    print("Class distribution:", class_counts)

    # Split
    if SPLIT_MODE == "chronological":
        X_train, X_test, y_train, y_test = chrono_split(X, y, SPLIT_FRAC)
    else:
        X_train, X_test, y_train, y_test = stratified_split(X, y, SPLIT_FRAC, RANDOM_STATE)

    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

    # Models
    models = {}

    # Logistic Regression (multinomial)
    models["LogisticRegression"] = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=500,
        n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
        random_state=RANDOM_STATE
    )

    # Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Gradient Boosting (fallback if XGBoost unavailable)
    models["GradientBoosting"] = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    # MLP
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=RANDOM_STATE
    )

    # XGBoost (optional)
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        print("XGBoost detected and will be trained.")
    else:
        print("XGBoost not installed; skipping. (pip install xgboost)")

    # Train, evaluate, save
    all_metrics = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = eval_metrics(y_test, y_pred)
        all_metrics[name] = metrics

        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })
        pred_df.to_csv(OUT_DIR / f"predictions_{name}.csv", index=False)

        # Save model
        joblib.dump(model, OUT_DIR / f"model_{name}.pkl")

        print(f"{name} | Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

    # Save metrics
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ Baseline training complete.")
    print(f"Results saved to: {OUT_DIR}")
