#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large_dataset_preprocessing.py
Robust preprocessing for weatherHistory.csv (Large dataset)

Outputs (to Data/Large/Preprocessed_dataset/):
  - clean_weather.csv                      (ALL columns preserved, cleaned)
  - prepared_for_models.csv                (numeric standardized, low-card cats one-hot)
  - prepared_for_models_noleak.csv         (same as above, but EXCLUDES label-source column)
  - prep_report.json                       (summary of processing)

Key features:
  - Parses 'Formatted Date' with timezone -> UTC -> naive
  - (Optional) Enforces hourly continuity (fills missing timestamps)
  - Time-aware interpolation for numeric columns; median fallback
  - Forward/backward fill for non-numeric columns
  - 3-class label from 'Apparent Temperature (C)' (fallback to 'Temperature (C)')
  - Drops high-cardinality free-text columns from MODEL files (kept in CLEAN)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
INPUT_PATH = Path("Data\Large\weatherHistory.csv")
OUTPUT_DIR = Path("Data\Large\Preprocessed_dataset")

ENFORCE_HOURLY = True               # reindex to hourly continuity
MAX_CARD_FOR_OHE = 50               # one-hot only low-card categoricals
DROP_LABEL_SOURCE_IN_NOLEAK = True  # create prepared_for_models_noleak.csv

# Preferred columns to derive the label from (in order)
LABEL_COLUMN_PREFS = ["Apparent Temperature (C)", "Temperature (C)"]


# -------------- IO / READ --------------
def read_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # trim headers and drop unnamed junk
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


# -------------- DATETIME --------------
def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Try to find a date column (prefer 'Formatted Date')
    candidates = [c for c in df.columns if c.lower().replace("_", " ") in ("formatted date", "date", "datetime")]
    if not candidates:
        raise ValueError("Could not find a 'Formatted Date' / 'Date' / 'Datetime' column.")
    dt_col = candidates[0]

    # Parse with timezone to UTC, then drop tz to get naive index
    dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    df = df.assign(datetime=dt)
    df = df.sort_values("datetime").dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.set_index("datetime").drop(columns=[dt_col])

    if ENFORCE_HOURLY:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="H")
        df = df.reindex(full_idx)

    return df


# -------------- MISSING VALUES --------------
def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Interpolate numeric, then median-fill leftovers
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        try:
            df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        except Exception:
            df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    # Categorical: forward/back fill
    other_cols = df.columns.difference(num_cols)
    for c in other_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()

    return df


# -------------- LABELING --------------
def choose_label_source(df: pd.DataFrame) -> str:
    # exact match first
    for name in LABEL_COLUMN_PREFS:
        if name in df.columns:
            return name
    # then permissive match
    lc_cols = {c.lower(): c for c in df.columns}
    for name in LABEL_COLUMN_PREFS:
        key = name.lower()
        for lc, orig in lc_cols.items():
            if key in lc:
                return orig
    raise ValueError(f"None of label source columns found: {LABEL_COLUMN_PREFS}")

def make_label(df: pd.DataFrame):
    source_col = choose_label_source(df)
    s = pd.to_numeric(df[source_col], errors="coerce")

    # Robust qcut on rank to break ties; prefer 3 bins, fallback to 2 or 1
    try:
        labels, bins = pd.qcut(s.rank(method="first"), q=3, labels=False, retbins=True, duplicates="drop")
        if pd.Series(labels).nunique() < 3:
            raise ValueError("qcut produced <3 bins; falling back to 2 bins.")
        bins_used = bins.tolist()
    except Exception:
        try:
            labels, bins = pd.qcut(s.rank(method="first"), q=2, labels=False, retbins=True, duplicates="drop")
            bins_used = bins.tolist()
        except Exception:
            labels = pd.Series(0, index=s.index)
            bins_used = [-np.inf, np.inf]
    labels = labels.astype(int)
    return source_col, labels, bins_used


# -------------- MODEL FILE BUILD --------------
def build_model_table(df: pd.DataFrame, label_col="label", drop_cols=None, max_card=50):
    """
    Create model-ready table:
      - one-hot encode low-card categoricals
      - drop high-card text columns
      - standardize all numeric columns except the label
      - optionally drop leakage columns (drop_cols)
    """
    out = df.copy()
    # Optionally drop leakage columns (e.g., the label-source feature)
    if drop_cols:
        for c in drop_cols:
            if c in out.columns:
                out = out.drop(columns=[c])

    # One-hot encode low-card categorical/text columns; drop high-card free text
    obj_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in obj_cols if out[c].nunique(dropna=True) <= max_card]
    high_card = [c for c in obj_cols if c not in low_card]

    if low_card:
        out = pd.get_dummies(out, columns=low_card, drop_first=False)
    if high_card:
        out = out.drop(columns=high_card)

    # Standardize numeric columns (except label)
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)
    if num_cols:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])

    return out, {"one_hot_columns": low_card, "dropped_text_columns": high_card, "dropped_leakage_columns": drop_cols or []}


# -------------- MAIN --------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read & datetime
    df_raw = read_weather(INPUT_PATH)
    df = build_datetime(df_raw)

    # Missing stats (before/after)
    before_missing = df.isna().sum().to_dict()
    df = fix_missing(df)
    after_missing = df.isna().sum().to_dict()

    # Label
    label_source_col, labels, bins_used = make_label(df)
    df["label"] = labels  # 0/1/2 if possible; 0/1 if fallback; 0 if degenerate

    # -------- Save CLEAN (everything preserved) --------
    clean_path = OUTPUT_DIR / "clean_weather.csv"
    df.to_csv(clean_path, index=True)

    # -------- Save MODEL (all features, numeric std, low-card OHE) --------
    model_all, enc_info_all = build_model_table(
        df, label_col="label", drop_cols=None, max_card=MAX_CARD_FOR_OHE
    )
    model_all_path = OUTPUT_DIR / "prepared_for_models.csv"
    model_all.to_csv(model_all_path, index=True)

    # -------- Save MODEL (NO LEAK: drop label source feature) --------
    leak_drop_cols = [label_source_col] if DROP_LABEL_SOURCE_IN_NOLEAK else None
    model_noleak, enc_info_noleak = build_model_table(
        df, label_col="label", drop_cols=leak_drop_cols, max_card=MAX_CARD_FOR_OHE
    )
    model_noleak_path = OUTPUT_DIR / "prepared_for_models_noleak.csv"
    model_noleak.to_csv(model_noleak_path, index=True)

    # -------- Report --------
    report = {
        "input": str(INPUT_PATH),
        "rows": int(df.shape[0]),
        "cols_clean": int(df.shape[1]),
        "label_source_column": label_source_col,
        "label_bins_used": bins_used,
        "enforce_hourly": ENFORCE_HOURLY,
        "missing_before": before_missing,
        "missing_after": after_missing,
        "encoding_info": {
            "prepared_for_models": enc_info_all,
            "prepared_for_models_noleak": enc_info_noleak
        },
        "outputs": {
            "clean_csv": str(clean_path),
            "prepared_for_models_csv": str(model_all_path),
            "prepared_for_models_noleak_csv": str(model_noleak_path),
        }
    }
    with open(OUTPUT_DIR / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Preprocessing (Large) complete.")
    print(f"• Cleaned file:                  {clean_path}")
    print(f"• Model file (with source):      {model_all_path}")
    print(f"• Model file (NO leak):          {model_noleak_path}")
    print(f"• Report:                        {OUTPUT_DIR / 'prep_report.json'}")
