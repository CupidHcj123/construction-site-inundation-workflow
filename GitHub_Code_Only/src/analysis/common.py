#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def _norm_col_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("²", "2").replace("³", "3")
    text = re.sub(r"[\s_]+", "", text)
    return text


_COLUMN_ALIASES = {
    "runid": "Run_ID",
    "pitid": "Pit_ID",
    "id": "Pit_ID",
    "rainfall(mm)": "Rainfall_mm",
    "rainfall(mm)": "Rainfall_mm",
    "rainfallmm": "Rainfall_mm",
    "rainfall": "Rainfall_mm",
    "rainfallpattern": "Rainfall_Pattern",
    "pattern": "Rainfall_Pattern",
    "scenariotype": "Scenario_Type",
    "scenario": "Scenario_Type",
    "designregion": "Design_Region",
    "returnperiod(yr)": "Return_Period_yr",
    "returnperiodyr": "Return_Period_yr",
    "returnperiod": "Return_Period_yr",
    "chicagopeakratio": "Chicago_Peak_Ratio",
    "peakratio": "Chicago_Peak_Ratio",
    "rainfiletotalmm": "Rain_File_Total_mm",
    "rainfilepeakintensitymm/h": "Rain_File_Peak_Intensity_mm_h",
    "rainfilepeakintensitymmh": "Rain_File_Peak_Intensity_mm_h",
    "rainfileintervals": "Rain_File_Interval_s",
    "duration(s)": "Rainfall_Duration_s",
    "duration": "Rainfall_Duration_s",
    "rainfalldurations": "Rainfall_Duration_s",
    "pitarea(m2)": "Pit_Area_m2",
    "pitarea": "Pit_Area_m2",
    "area(m2)": "Pit_Area_m2",
    "pitmaxdepth(m)": "Pit_Max_Depth_m",
    "maxdepth(m)": "Pit_Max_Depth_m",
    "pitavgdepth(m)": "Pit_Avg_Depth_m",
    "averagedepth(m)": "Pit_Avg_Depth_m",
    "pitvolume(m3)": "Pit_Volume_m3",
    "volume(m3)": "Pit_Volume_m3",
    "finalinundationdepth(m)": "Final_Inundation_Depth_m",
    "maxinundationdepth(m)": "Final_Inundation_Depth_m",
    "hmaxm": "Hmax_m",
    "shape_ratio": "Shape_Ratio",
    "shaperatio": "Shape_Ratio",
    "elongation": "Elongation",
    "meanslope(deg)": "Mean_Slope_deg",
    "stddevslope(deg)": "Std_Dev_Slope_deg",
    "micro_roughness": "Micro_Roughness",
}


REQUIRED_COLUMNS = [
    "Pit_ID",
    "Rainfall_mm",
    "Rainfall_Pattern",
    "Pit_Area_m2",
    "Pit_Max_Depth_m",
    "Pit_Avg_Depth_m",
    "Final_Inundation_Depth_m",
]

NUMERIC_COLUMNS = [
    "Pit_ID",
    "Rainfall_mm",
    "Rainfall_Duration_s",
    "Return_Period_yr",
    "Chicago_Peak_Ratio",
    "Rain_File_Total_mm",
    "Rain_File_Peak_Intensity_mm_h",
    "Rain_File_Interval_s",
    "Pit_Area_m2",
    "Pit_Max_Depth_m",
    "Pit_Avg_Depth_m",
    "Pit_Volume_m3",
    "Final_Inundation_Depth_m",
    "Hmax_m",
    "Shape_Ratio",
    "Elongation",
    "Mean_Slope_deg",
    "Std_Dev_Slope_deg",
    "Micro_Roughness",
]


@dataclass
class PreparedData:
    df: pd.DataFrame
    report: Dict[str, object]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        norm = _norm_col_name(col)
        if norm in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[norm]
    out = df.rename(columns=rename_map).copy()
    if "Rainfall_Pattern" not in out.columns:
        out["Rainfall_Pattern"] = "UNIFORM"
    out["Rainfall_Pattern"] = out["Rainfall_Pattern"].astype(str).str.upper().str.strip()
    if "Scenario_Type" not in out.columns:
        out["Scenario_Type"] = np.where(out["Rainfall_Pattern"].eq("CHICAGO"), "DESIGN_CHICAGO", "IDEALIZED")
    out["Scenario_Type"] = out["Scenario_Type"].astype(str).str.upper().str.strip()
    if "Design_Region" in out.columns:
        out["Design_Region"] = out["Design_Region"].astype(str).str.strip()
    return out


def _parse_duration_from_run_id(run_id: object) -> float:
    if pd.isna(run_id):
        return np.nan
    text = str(run_id)
    m = re.search(r"(?<!\d)(\d{3,5})s(?!\w)", text, flags=re.IGNORECASE)
    if not m:
        return np.nan
    return float(m.group(1))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Rainfall_Duration_s" not in out.columns:
        if "Run_ID" in out.columns:
            out["Rainfall_Duration_s"] = out["Run_ID"].map(_parse_duration_from_run_id)
        else:
            out["Rainfall_Duration_s"] = np.nan

    out["RF_over_MaxDepth"] = out["Rainfall_mm"] / out["Pit_Max_Depth_m"].replace(0, np.nan)
    out["RF_over_AvgDepth"] = out["Rainfall_mm"] / out["Pit_Avg_Depth_m"].replace(0, np.nan)
    out["RainIntensity_over_AvgDepth"] = np.nan
    out["RainIntensity_over_MaxDepth"] = np.nan

    out["y_ratio_raw"] = out["Final_Inundation_Depth_m"] / out["Pit_Max_Depth_m"].replace(0, np.nan)
    out["y_ratio_clip"] = out["y_ratio_raw"].clip(EPS, 1 - EPS)
    out["y_logit"] = np.log(out["y_ratio_clip"] / (1 - out["y_ratio_clip"]))

    if "Rainfall_Duration_s" in out.columns:
        if out["Rainfall_Duration_s"].notna().any():
            out["Rainfall_Intensity_mm_per_h"] = (
                out["Rainfall_mm"] / (out["Rainfall_Duration_s"] / 3600.0)
            )
        else:
            out["Rainfall_Intensity_mm_per_h"] = np.nan
    else:
        out["Rainfall_Intensity_mm_per_h"] = np.nan
    out["RainIntensity_over_AvgDepth"] = out["Rainfall_Intensity_mm_per_h"] / out["Pit_Avg_Depth_m"].replace(0, np.nan)
    out["RainIntensity_over_MaxDepth"] = out["Rainfall_Intensity_mm_per_h"] / out["Pit_Max_Depth_m"].replace(0, np.nan)
    return out


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _build_report(df: pd.DataFrame) -> Dict[str, object]:
    report: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_unique_pits": int(df["Pit_ID"].nunique()) if "Pit_ID" in df.columns else 0,
        "n_unique_rainfall": int(df["Rainfall_mm"].nunique()) if "Rainfall_mm" in df.columns else 0,
        "rainfall_patterns": sorted(df["Rainfall_Pattern"].dropna().unique().tolist())
        if "Rainfall_Pattern" in df.columns
        else [],
        "scenario_types": sorted(df["Scenario_Type"].dropna().unique().tolist())
        if "Scenario_Type" in df.columns
        else [],
        "n_duration_levels": int(df["Rainfall_Duration_s"].dropna().nunique())
        if "Rainfall_Duration_s" in df.columns
        else 0,
        "n_return_period_levels": int(df["Return_Period_yr"].dropna().nunique())
        if "Return_Period_yr" in df.columns
        else 0,
        "ratio_above_1_count": int((df["y_ratio_raw"] > 1).sum())
        if "y_ratio_raw" in df.columns
        else 0,
    }
    miss = {}
    for c in REQUIRED_COLUMNS:
        miss[c] = int(df[c].isna().sum()) if c in df.columns else -1
    report["missing_required"] = miss
    return report


def load_events_table(path: str, sheet_name: int | str = 0, dropna_required: bool = True) -> PreparedData:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    if src.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(src, sheet_name=sheet_name)
    else:
        df = pd.read_csv(src)

    df = standardize_columns(df)
    _coerce_numeric(df, NUMERIC_COLUMNS)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns after normalization: {missing_required}")

    if dropna_required:
        df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    df = add_engineered_features(df)
    report = _build_report(df)
    return PreparedData(df=df, report=report)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def grouped_kfold(df: pd.DataFrame, group_col: str = "Pit_ID", n_splits: int = 5) -> GroupKFold:
    n_groups = int(df[group_col].nunique())
    if n_groups < 2:
        raise ValueError(f"At least 2 groups required, got {n_groups}.")
    n_splits = max(2, min(n_splits, n_groups))
    return GroupKFold(n_splits=n_splits)


def metric_ci95(values: Iterable[float]) -> Tuple[float, float, float, int]:
    arr = pd.Series(values, dtype="float64").dropna().values
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, np.nan, np.nan, 1
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half, n
