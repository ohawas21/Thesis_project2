#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large_dataset_preprocessing.py
Final version for weatherHistory.csv (Large dataset)

What it does
------------
- Reads weatherHistory.csv (comma-separated)
- Preserves ALL columns (drops only empty 'Unnamed' ones)
- Builds datetime index from 'Formatted Date' (handles timezone)
- Interpolates numeric columns (time-aware), median fallback
- Ffills/bfills non-numeric columns
- Creates 'label' (0/1/2) via robust quantile binning on 'Apparent Temperature (C)'
  (fallback to 'Temperature (C)' if needed; fallback to 2/1 classes if ties)
- Saves:
    Data/Large/Preprocessed_dataset/clean_weather.csv
    Data/Large/Preprocessed_dataset/prepared_for_models.csv
        - numeric standardized
        - categoricals one-hot (only low-cardinality to avoid huge expansion)
    Data/Large/Preprocessed_dataset/prep_report.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==== CONFIG (hard-coded paths) ====
INPUT_PATH = Path("Data/Large/weatherHistory.csv")
OUTPUT_DIR = Path("Data/Large/Preprocessed_dataset")

# Columns commonly present:
# 'Formatted Date','Summary','Precip Type','Temperature (C)','Apparent Temperature (C)',
# 'Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)',
# 'Loud Cover','Pressure (millibars)','Daily Summary'

# ---------- IO ----------
def read_weather(path: Path) -> pd.DataFrame:
    """Read CSV, drop unnamed junk, strip header whitespace."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df

# ---------- Datetime ----------
def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create datetime index from 'Formatted Date'.
    Example format: '2006-04-01 00:00:00.000 +0200'
    """
    # find column (case/space tolerant)
    cand = [c for c in df.columns if c.lower().replace("_", " ") in ("formatted date", "date", "datetime")]
    if not cand:
        raise ValueError("Could not find 'Formatted Date' / 'Date' column.")
    dt_col = cand[0]

    dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    df = df.assign(datetime=dt)
    df = df.sort_values("datetime").dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    df = df.set_index("datetime")
    # Optionally convert to local tz; comment out if not needed:
    # df.index = df.index.tz_convert("UTC")  # or your timezone
    return df.drop(columns=[dt_col])

# ---------- Missing values ----------
def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate numeric columns (time-aware); median fill leftovers.
    Ffill/bfill for non-numeric columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    # time-aware interpolation if index is datetime
    if len(num_cols) > 0:
        try:
            df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        except Exception:
            df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
        # median fallback
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    # non-numeric: ffill/bfill
    other_cols = df.columns.difference(num_cols)
    for c in other_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()
    return df

# ---------- Labeling ----------
def make_temperature_label(df: pd.DataFrame):
    """
    Create label (0/1/2) from 'Apparent Temperature (C)' (preferred) or 'Temperature (C)' (fallback)
    using robust quantile binning with tie handling and fallbacks to 2/1 classes.
    """
    # detect columns
    def find_col(candidates):
        for cand in candidates:
            hits = [c for c in df.columns if c.strip().lower() == cand]
            if hits:
                return hits[0]
        # looser match
        hits = [c for c in df.columns if any(cand in c.strip().lower() for cand in candidates)]
        return hits[0] if hits else None

    apparent = find_col(["apparent temperature (c)", "apparent temperature"])
    temp = find_col(["temperature (c)", "temperature"])

    base_col = apparent or temp
    if base_col is None:
        raise ValueError("Neither 'Apparent Temperature (C)' nor 'Temperature (C)' found.")
    series = pd.to_numeric(df[base_col], errors="coerce")

    # Robust qcut on ranks to break ties; prefer 3 bins, fallback to 2/1
    try:
        labels, bins = pd.qcut(series.rank(method="first"), q=3, labels=False, retbins=True, duplicates="drop")
        if pd.Series(labels).nunique() < 3:
            raise ValueError("qcut produced <3 bins; fallback.")
        bins_used = bins.tolist()
    except Exception:
        try:
            labels, bins = pd.qcut(series.rank(method="first"), q=2, labels=False, retbins=True, duplicates="drop")
            bins_used = bins.tolist()
        except Exception:
            labels = pd.Series(0, index=series.index)
            bins_used = [-np.inf, np.inf]

    labels = labels.astype(int)
    return base_col, labels, bins_used

# ---------- Model-ready transform ----------
def standardize_and_encode(df: pd.DataFrame, label_col="label", max_cardinality=50) -> pd.DataFrame:
    """
    Standardize numeric columns, one-hot encode categorical columns with
    reasonable cardinality to avoid massive expansion (keeps clean file intact).
    """
    out = df.copy()

    # Identify categorical candidates (object/category) and their cardinalities
    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card_cats = [c for c in cat_cols if out[c].nunique(dropna=True) <= max_cardinality]

    # One-hot only low-cardinality categoricals (e.g., 'Precip Type'); skip free-text ('Summary', 'Daily Summary')
    if low_card_cats:
        out = pd.get_dummies(out, columns=low_card_cats, drop_first=False)

    # Standardize numeric columns except label
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)
    if num_cols:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])

    return out, {"one_hot_columns": low_card_cats, "skipped_text_columns": [c for c in cat_cols if c not in low_card_cats]}

# ---------- MAIN ----------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & datetime
    df_raw = read_weather(INPUT_PATH)
    df = build_datetime(df_raw)

    # Missing stats (before/after)
    before_missing = df.isna().sum().to_dict()
    df = fix_missing(df)
    after_missing = df.isna().sum().to_dict()

    # Labeling
    base_temp_col, labels, bins_used = make_temperature_label(df)
    df["label"] = labels  # 0/1/2 if possible; 0/1 if fallback; 0 if degenerate

    # Save CLEAN (all columns preserved)
    clean_path = OUTPUT_DIR / "clean_weather.csv"
    df.to_csv(clean_path, index=True)

    # Prepare MODEL dataset (numeric standardized, low-card categoricals one-hot)
    model_df, enc_info = standardize_and_encode(df, label_col="label", max_cardinality=50)
    model_path = OUTPUT_DIR / "prepared_for_models.csv"
    model_df.to_csv(model_path, index=True)

    # Report
    report = {
        "input": str(INPUT_PATH),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "label_source_column": base_temp_col,
        "label_bins_used": bins_used,
        "missing_before": before_missing,
        "missing_after": after_missing,
        "encoding_info": enc_info,
        "outputs": {
            "clean_csv": str(clean_path),
            "prepared_for_models_csv": str(model_path),
        },
        "notes": {
            "numeric_standardized": True,
            "categorical_one_hot_low_card": True,
            "free_text_kept_in_clean_only": True,
            "datetime_index": True
        }
    }
    with open(OUTPUT_DIR / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Preprocessing (Large) done.")
    print(f"• Cleaned file:       {clean_path}")
    print(f"• Standardized file:  {model_path}")
    print(f"• Report:             {OUTPUT_DIR / 'prep_report.json'}")
