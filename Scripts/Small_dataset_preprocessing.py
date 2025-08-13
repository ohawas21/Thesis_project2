#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small_dataset_preprocessing.py
Final version for AirQualityUCI (Small dataset)

What it does
------------
- Reads AirQualityUCI.csv (semicolon separator, decimal commas)
- Preserves ALL columns (drops only empty 'Unnamed' ones)
- Builds a proper datetime index from Date + Time ('HH.MM.SS' -> 'HH:MM:SS')
- Replaces -200 with NaN (numeric cols), interpolates (time-aware), median fallback
- Ffills/bfills non-numeric NaNs
- Creates 'pollution_index' (sum of CO(GT), C6H6(GT), NMHC(GT))
- Creates 'label' using robust quantile binning (3 classes if possible, fallback to 2/1)
- Saves:
    Data/Small/Preprocessed_dataset/clean_airquality.csv
    Data/Small/Preprocessed_dataset/prepared_for_models.csv (standardized)
    Data/Small/Preprocessed_dataset/prep_report.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==== CONFIG (hard-coded paths as requested) ====
INPUT_PATH = Path("Data/Small/AirQualityUCI.csv")
OUTPUT_DIR = Path("Data/Small/Preprocessed_dataset")


# ==== HELPERS ====
def read_airquality(path: Path) -> pd.DataFrame:
    """Read with semicolon separator; convert decimal commas to dots where applicable."""
    df = pd.read_csv(path, sep=";")
    # Drop unnamed junk columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    # Convert decimal commas in object-like columns if they look numeric
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()
            conv = pd.to_numeric(s, errors="coerce")
            # Keep conversion if meaningful fraction converted
            if conv.notna().mean() >= 0.2:
                df[c] = conv
    return df


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime index from 'Date' + 'Time' (Time like '18.00.00')."""
    if {"Date", "Time"}.issubset(df.columns):
        time_norm = (
            df["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
        )
        dt = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + time_norm,
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df = df.assign(datetime=dt).drop(columns=["Date", "Time"])
    else:
        raise ValueError("Expected 'Date' and 'Time' columns in AirQualityUCI.")
    df = df.sort_values("datetime").dropna(subset=["datetime"])
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime")
    return df


def fix_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace -200 -> NaN (numeric), interpolate, median-fill; ffill/bfill non-numeric."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(-200, np.nan)
    # Time-aware interpolation
    try:
        df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
    except Exception:
        df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    # Median fallback for remaining numeric NaNs
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    # Ffill/bfill for non-numeric
    other_cols = df.columns.difference(num_cols)
    for c in other_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()
    return df


def make_composite_and_label(df: pd.DataFrame):
    """
    Create composite pollution index + label.
    - Auto-detect the expected pollutant columns (CO(GT), C6H6(GT), NMHC(GT))
    - Prefer 3-class quantile binning; fallback to 2/1 class if ties prevent strict bins.
    """
    expected = ["CO(GT)", "C6H6(GT)", "NMHC(GT)"]
    cols = [c for c in df.columns if c in expected]
    if len(cols) < 3:
        # tolerate slight name variants by substring search
        cols = [c for c in df.columns if any(k in str(c) for k in expected)]
    if not cols:
        raise ValueError("Pollutant columns for composite index not found.")

    composite = df[cols].sum(axis=1)

    # Robust binning with qcut on ranked values to break ties
    labels = None
    bins_used = None
    try:
        labels, bins = pd.qcut(
            composite.rank(method="first"),
            q=3,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        # Ensure we actually got 3 bins/classes
        if pd.Series(labels).nunique() < 3:
            raise ValueError("qcut produced <3 bins; falling back to 2 bins.")
        bins_used = bins
    except Exception:
        try:
            labels, bins = pd.qcut(
                composite.rank(method="first"),
                q=2,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            bins_used = bins
        except Exception:
            # Degenerate case: no variability -> single class
            labels = pd.Series(0, index=composite.index)
            bins_used = np.array([-np.inf, np.inf])

    labels = labels.astype(int)
    return composite, (bins_used.tolist() if hasattr(bins_used, "tolist") else list(bins_used)), labels, cols


def standardize_numeric(df: pd.DataFrame, drop_cols=("label",)) -> pd.DataFrame:
    """Standardize all numeric columns except those listed in drop_cols."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.difference(drop_cols)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


# ==== MAIN ====
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & parse
    df_raw = read_airquality(INPUT_PATH)
    df = build_datetime(df_raw)

    # Missing stats (before/after)
    before_missing = df.isna().sum().to_dict()
    df = fix_missing(df)
    after_missing = df.isna().sum().to_dict()

    # Composite index + label
    composite, bins_used, labels, used_cols = make_composite_and_label(df)
    df["pollution_index"] = composite
    df["label"] = labels  # 0/1/2 if possible; 0/1 if fallback; 0 if degenerate

    # Save cleaned dataset (ALL columns preserved)
    clean_path = OUTPUT_DIR / "clean_airquality.csv"
    df.to_csv(clean_path, index=True)

    # Save standardized version for models (ALL numeric cols standardized, label intact)
    std_df = standardize_numeric(df, drop_cols=("label",))
    std_path = OUTPUT_DIR / "prepared_for_models.csv"
    std_df.to_csv(std_path, index=True)

    # Save report
    report = {
        "input": str(INPUT_PATH),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "pollutant_columns_used": used_cols,
        "label_bins_used": bins_used,
        "missing_before": before_missing,
        "missing_after": after_missing,
        "outputs": {
            "clean_csv": str(clean_path),
            "prepared_for_models_csv": str(std_path),
        },
    }
    with open(OUTPUT_DIR / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Preprocessing done.")
    print(f"• Cleaned file:       {clean_path}")
    print(f"• Standardized file:  {std_path}")
    print(f"• Report:             {OUTPUT_DIR / 'prep_report.json'}")
