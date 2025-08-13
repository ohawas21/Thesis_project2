#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medium_dataset_preprocessing.py
Final version for Beijing_Weather_data_2010.1.1-2014.12.31.csv (Medium dataset)

What it does
------------
- Reads CSV with robust NA handling (e.g., 'NA' in pm2.5)
- Preserves ALL columns (drops only empty 'Unnamed' ones)
- Builds hourly datetime index from year/month/day/hour
- Replaces obvious sentinels with NaN (numeric cols), interpolates (time-aware), median fallback
- Ffills/bfills non-numeric NaNs
- Creates 'label' from pm2.5 via robust quantile binning (3 classes if possible; fallback to 2/1)
- Saves:
    Data/Medium/Preprocessed_dataset/clean_beijing.csv
    Data/Medium/Preprocessed_dataset/prepared_for_models.csv (standardized numerics + one-hot categoricals, label preserved)
    Data/Medium/Preprocessed_dataset/prep_report.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==== CONFIG (hard-coded paths as requested) ====
INPUT_PATH = Path("Data/Medium/Beijing_Weather_data_2010.1.1-2014.12.31.csv")
OUTPUT_DIR = Path("Data/Medium/Preprocessed_dataset")

# Columns typical in this dataset:
# ['No','year','month','day','hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']

def read_beijing(path: Path) -> pd.DataFrame:
    """Robust reader: handle NA strings and drop empty unnamed columns."""
    df = pd.read_csv(
        path,
        na_values=["NA", "NaN", "nan", ""],
        keep_default_na=True
    )
    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df

def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime index from year/month/day/hour (assumed integers)."""
    required = {"year", "month", "day", "hour"}
    lower_map = {c.lower(): c for c in df.columns}
    if not required.issubset(lower_map.keys()):
        raise ValueError(f"Expected columns {sorted(list(required))} in any case; got {list(df.columns)}.")
    y = df[lower_map["year"]].astype(int, errors="ignore")
    m = df[lower_map["month"]].astype(int, errors="ignore")
    d = df[lower_map["day"]].astype(int, errors="ignore")
    h = df[lower_map["hour"]].astype(int, errors="ignore")

    dt = pd.to_datetime(
        dict(year=y, month=m, day=d, hour=h),
        errors="coerce"
    )
    df = df.assign(datetime=dt)
    df = df.sort_values("datetime").dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    df = df.set_index("datetime")
    return df

def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate numeric, fill categoricals."""
    # Treat '-200' (AirQuality style) if present
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace(-200, np.nan)

    # Time-aware interpolation for numeric columns
    if len(num_cols) > 0:
        try:
            df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        except Exception:
            df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
        # Median-fill numeric leftovers
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    # For non-numeric (e.g., cbwd), forward/back fill
    other_cols = df.columns.difference(num_cols)
    for c in other_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()
    return df

def make_label_from_pm25(df: pd.DataFrame):
    """Create 'label' from PM2.5 via robust quantile binning (qcut with ties fallback)."""
    # Detect pm2.5 column (case-insensitive)
    pm_candidates = [c for c in df.columns if c.lower() in ("pm2.5", "pm2_5", "pm25", "pm 2.5")]
    if not pm_candidates:
        # Also try substring match
        pm_candidates = [c for c in df.columns if "pm2.5" in c.lower() or "pm25" in c.lower()]
    if not pm_candidates:
        raise ValueError("PM2.5 column not found (expected something like 'pm2.5').")
    pm_col = pm_candidates[0]
    pm = df[pm_col].astype(float)

    # Robust qcut on ranks to break ties; prefer 3 bins, fallback to 2 or 1
    try:
        labels, bins = pd.qcut(pm.rank(method="first"), q=3, labels=False, retbins=True, duplicates="drop")
        if pd.Series(labels).nunique() < 3:
            raise ValueError("qcut produced <3 bins; falling back to 2 bins.")
        bins_used = bins.tolist()
    except Exception:
        try:
            labels, bins = pd.qcut(pm.rank(method="first"), q=2, labels=False, retbins=True, duplicates="drop")
            bins_used = bins.tolist()
        except Exception:
            labels = pd.Series(0, index=pm.index)
            bins_used = [-np.inf, np.inf]
    labels = labels.astype(int)
    return pm_col, labels, bins_used

def standardize_and_encode(df: pd.DataFrame, label_col="label") -> pd.DataFrame:
    """Standardize all numeric cols; one-hot encode categoricals; keep label intact."""
    out = df.copy()
    # Identify categorical columns (object or category)
    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    # One-hot encode categoricals
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, drop_first=False)

    # Standardize numeric columns except label
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)

    if num_cols:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])

    return out

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & parse
    df_raw = read_beijing(INPUT_PATH)
    df = build_datetime(df_raw)

    # Missing stats (before/after)
    before_missing = df.isna().sum().to_dict()
    df = fix_missing(df)
    after_missing = df.isna().sum().to_dict()

    # Label from PM2.5
    pm_col, labels, bins_used = make_label_from_pm25(df)
    df["label"] = labels  # 0/1/2 if possible; 0/1 if fallback; 0 if degenerate

    # Save cleaned dataset (ALL columns preserved)
    clean_path = OUTPUT_DIR / "clean_beijing.csv"
    df.to_csv(clean_path, index=True)

    # Save standardized + one-hot version for models
    model_df = standardize_and_encode(df, label_col="label")
    model_path = OUTPUT_DIR / "prepared_for_models.csv"
    model_df.to_csv(model_path, index=True)

    # Save report
    report = {
        "input": str(INPUT_PATH),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "pm25_column": pm_col,
        "label_bins_used": bins_used,
        "missing_before": before_missing,
        "missing_after": after_missing,
        "outputs": {
            "clean_csv": str(clean_path),
            "prepared_for_models_csv": str(model_path),
        },
        "notes": {
            "categorical_one_hot": True,
            "numeric_standardized": True,
            "datetime_index": True
        }
    }
    with open(OUTPUT_DIR / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Preprocessing (Medium) done.")
    print(f"• Cleaned file:       {clean_path}")
    print(f"• Standardized file:  {model_path}")
    print(f"• Report:             {OUTPUT_DIR / 'prep_report.json'}")
