#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

# Avoid matplotlib cache permission issues in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "step23_mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from .common import ensure_dir, grouped_kfold, load_events_table, metric_ci95, write_json
except ImportError:
    from common import ensure_dir, grouped_kfold, load_events_table, metric_ci95, write_json

warnings.filterwarnings("ignore", category=UserWarning)


def build_labels(
    df: pd.DataFrame,
    label_mode: str = "relative",
    rel_thresh: Tuple[float, float] = (0.3, 0.6),
    abs_thresh: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if label_mode == "absolute":
        if abs_thresh is None:
            raise ValueError("abs_thresh must be provided when label_mode='absolute'")
        t1, t2 = abs_thresh
        y = np.digitize(df["Final_Inundation_Depth_m"].values, bins=[t1, t2]).astype(int)
        scheme = {
            "mode": "absolute",
            "t1": float(t1),
            "t2": float(t2),
            "low": f"D<{t1}",
            "mid": f"{t1}<=D<{t2}",
            "high": f"D>={t2}",
        }
        return y, scheme

    t1, t2 = rel_thresh
    y = np.digitize(df["y_ratio_raw"].values, bins=[t1, t2]).astype(int)
    scheme = {
        "mode": "relative",
        "t1": float(t1),
        "t2": float(t2),
        "low": f"ratio<{t1}",
        "mid": f"{t1}<=ratio<{t2}",
        "high": f"ratio>={t2}",
    }
    return y, scheme


def choose_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_candidates = [
        "Rainfall_mm",
        "Rainfall_Duration_s",
        "Rainfall_Intensity_mm_per_h",
        "RainIntensity_over_AvgDepth",
        "RainIntensity_over_MaxDepth",
        "Return_Period_yr",
        "Chicago_Peak_Ratio",
        "Rain_File_Total_mm",
        "Rain_File_Peak_Intensity_mm_h",
        "Rain_File_Interval_s",
        "Pit_Area_m2",
        "Pit_Max_Depth_m",
        "Pit_Avg_Depth_m",
        "RF_over_AvgDepth",
        "RF_over_MaxDepth",
        "Shape_Ratio",
        "Elongation",
        "Mean_Slope_deg",
        "Std_Dev_Slope_deg",
        "Micro_Roughness",
    ]
    num_cols = [c for c in num_candidates if c in df.columns and df[c].dropna().nunique() > 1]

    cat_cols = []
    for c in ["Rainfall_Pattern", "Scenario_Type", "Design_Region"]:
        if c in df.columns and df[c].dropna().nunique() > 1:
            cat_cols.append(c)
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="NONE")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(kind: str, n_jobs: int = 1):
    if kind == "rf":
        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        )
        grid = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [None, 8, 12],
            "clf__min_samples_leaf": [1, 3, 5],
        }
        return base, grid

    if kind == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available in the current environment.")
        base = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=n_jobs,
            random_state=42,
        )
        grid = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.10],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.9],
        }
        return base, grid

    raise ValueError(f"Unknown model kind: {kind}")


def _safe_ap(y_bin: np.ndarray, p: np.ndarray) -> float:
    if len(y_bin) == 0:
        return np.nan
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(average_precision_score(y_bin, p))


def _safe_brier(y_bin: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(brier_score_loss(y_bin, p))


def _binary_event_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # POD
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_pod": recall,
        "far": far,
        "csi": csi,
    }


def _fit_with_calibration(
    pipe: Pipeline,
    param_grid: Dict[str, List[object]],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    inner_splits: int,
    n_jobs: int,
):
    inner_n = max(2, min(inner_splits, len(np.unique(groups_tr))))
    inner_cv = grouped_kfold(pd.DataFrame({"g": groups_tr}), group_col="g", n_splits=inner_n)
    search = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, n_jobs=n_jobs, scoring="f1_macro", refit=True)
    search.fit(X_tr, y_tr, groups=groups_tr)

    best_est = search.best_estimator_
    cal_cv = min(3, max(2, int(np.unique(y_tr).size)))
    try:
        calibrated = CalibratedClassifierCV(best_est, method="isotonic", cv=cal_cv)
        calibrated.fit(X_tr, y_tr)
        return calibrated, search.best_params_
    except Exception:
        try:
            calibrated = CalibratedClassifierCV(best_est, method="sigmoid", cv=cal_cv)
            calibrated.fit(X_tr, y_tr)
            return calibrated, search.best_params_
        except Exception:
            best_est.fit(X_tr, y_tr)
            return best_est, search.best_params_


def evaluate_one_model(
    df: pd.DataFrame,
    y: np.ndarray,
    kind: str,
    outdir: Path,
    n_splits: int = 5,
    inner_splits: int = 3,
    perm_repeats: int = 20,
    high_label: int = 2,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_cols, cat_cols = choose_feature_columns(df)
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    groups = df["Pit_ID"].values

    pre = build_preprocessor(num_cols, cat_cols)
    base, grid = build_model(kind, n_jobs=n_jobs)
    pipe = Pipeline([("pre", pre), ("clf", base)])

    outer_cv = grouped_kfold(df, group_col="Pit_ID", n_splits=n_splits)

    metric_rows = []
    pred_rows = []
    perm_rows = []
    param_rows = []

    for fold, (tr, te) in enumerate(outer_cv.split(df, y, groups), 1):
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = y[tr]
        y_te = y[te]
        g_tr = groups[tr]

        fitted, best_params = _fit_with_calibration(
            pipe,
            grid,
            X_tr,
            y_tr,
            g_tr,
            inner_splits=inner_splits,
            n_jobs=n_jobs,
        )
        param_rows.append({"fold": fold, "model": kind, **best_params})

        prob = fitted.predict_proba(X_te)
        y_pred = np.argmax(prob, axis=1)
        y_true_high = (y_te == high_label).astype(int)
        y_pred_high = (y_pred == high_label).astype(int)
        p_high = prob[:, high_label]

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        evt = _binary_event_metrics(y_true_high, y_pred_high)
        metric_rows.append(
            {
                "fold": fold,
                "model": kind,
                "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
                "f1_weighted": float(f1_score(y_te, y_pred, average="weighted")),
                "pr_auc_high": _safe_ap(y_true_high, p_high),
                "brier_high": _safe_brier(y_true_high, p_high),
                "n_test": int(len(y_te)),
                **evt,
                **{f"cm_{i}{j}": int(cm[i, j]) for i in range(3) for j in range(3)},
            }
        )

        fold_pred = pd.DataFrame(
            {
                "fold": fold,
                "idx": X_te.index.values,
                "Run_ID": df.iloc[te]["Run_ID"].values if "Run_ID" in df.columns else np.nan,
                "Pit_ID": df.iloc[te]["Pit_ID"].values,
                "Rainfall_mm": df.iloc[te]["Rainfall_mm"].values,
                "Rainfall_Pattern": df.iloc[te]["Rainfall_Pattern"].values,
                "Rainfall_Duration_s": df.iloc[te]["Rainfall_Duration_s"].values if "Rainfall_Duration_s" in df.columns else np.nan,
                "Scenario_Type": df.iloc[te]["Scenario_Type"].values if "Scenario_Type" in df.columns else np.nan,
                "Design_Region": df.iloc[te]["Design_Region"].values if "Design_Region" in df.columns else np.nan,
                "Return_Period_yr": df.iloc[te]["Return_Period_yr"].values if "Return_Period_yr" in df.columns else np.nan,
                "Chicago_Peak_Ratio": df.iloc[te]["Chicago_Peak_Ratio"].values if "Chicago_Peak_Ratio" in df.columns else np.nan,
                "Rain_File_Peak_Intensity_mm_h": df.iloc[te]["Rain_File_Peak_Intensity_mm_h"].values if "Rain_File_Peak_Intensity_mm_h" in df.columns else np.nan,
                "true": y_te,
                "pred": y_pred,
                "p_low": prob[:, 0],
                "p_mid": prob[:, 1],
                "p_high": prob[:, 2],
            }
        )
        pred_rows.append(fold_pred)

        # Permutation importance on test fold (feature-level)
        try:
            pi = permutation_importance(
                fitted,
                X_te,
                y_te,
                n_repeats=perm_repeats,
                random_state=42,
                scoring="f1_macro",
            )
            for j, name in enumerate(feature_cols):
                perm_rows.append(
                    {
                        "fold": fold,
                        "model": kind,
                        "feature": name,
                        "importance_mean": float(pi.importances_mean[j]),
                        "importance_std": float(pi.importances_std[j]),
                    }
                )
        except Exception:
            pass

        joblib.dump(fitted, outdir / f"model_{kind}_fold{fold}.joblib")

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    perm_df = pd.DataFrame(perm_rows)
    params_df = pd.DataFrame(param_rows)

    metrics_df.to_csv(outdir / f"cv_metrics_{kind}.csv", index=False)
    preds_df.to_csv(outdir / f"cv_predictions_{kind}.csv", index=False)
    params_df.to_csv(outdir / f"best_params_{kind}.csv", index=False)

    if not perm_df.empty:
        perm_df.to_csv(outdir / f"perm_importance_{kind}_byfold.csv", index=False)
        agg = (
            perm_df.groupby("feature", as_index=False)["importance_mean"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "importance_mean", "std": "importance_std"})
            .sort_values("importance_mean", ascending=False)
        )
        agg.to_csv(outdir / f"perm_importance_{kind}_agg.csv", index=False)
    return metrics_df, preds_df, perm_df


def scan_thresholds(
    pred_df: pd.DataFrame,
    outdir: Path,
    model_name: str,
    high_label: int = 2,
    cost_fn: float = 3.0,
    cost_fp: float = 1.0,
    recall_floor: float = 0.90,
) -> None:
    y_true = (pred_df["true"].values == high_label).astype(int)
    p_high = pred_df["p_high"].values

    grid = np.unique(np.concatenate([np.linspace(0, 1, 501), p_high]))
    rows = []
    for t in grid:
        y_hat = (p_high >= t).astype(int)
        evt = _binary_event_metrics(y_true, y_hat)
        cost = cost_fn * evt["fn"] + cost_fp * evt["fp"]
        cost_norm = cost / len(y_true)
        f1 = f1_score(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan
        rows.append({"threshold": float(t), "f1_high": f1, "cost_norm": float(cost_norm), **evt})

    scan_df = pd.DataFrame(rows)
    scan_df.to_csv(outdir / f"threshold_scan_{model_name}.csv", index=False)

    best_f1 = scan_df.iloc[scan_df["f1_high"].idxmax()] if scan_df["f1_high"].notna().any() else scan_df.iloc[0]
    best_cost = scan_df.iloc[scan_df["cost_norm"].idxmin()]

    cand = scan_df[scan_df["recall_pod"] >= recall_floor]
    if cand.empty:
        best_recall = scan_df.iloc[scan_df["recall_pod"].idxmax()]
    else:
        best_recall = cand.sort_values(["precision", "f1_high"], ascending=[False, False]).iloc[0]

    rec_df = pd.DataFrame(
        [
            {"criterion": "max_f1_high", **best_f1.to_dict()},
            {"criterion": "min_cost_norm", **best_cost.to_dict()},
            {"criterion": f"recall_floor_{recall_floor}", **best_recall.to_dict()},
        ]
    )
    rec_df.to_csv(outdir / f"threshold_recommend_{model_name}.csv", index=False)

    # Plot threshold curves
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(scan_df["threshold"], scan_df["precision"], label="Precision")
    ax.plot(scan_df["threshold"], scan_df["recall_pod"], label="Recall(POD)")
    ax.plot(scan_df["threshold"], scan_df["f1_high"], label="F1(high)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"threshold_curves_{model_name}.png", dpi=220)
    plt.close(fig)

    # PR curve with recommended operating points
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    if len(np.unique(y_true)) >= 2:
        prec, rec, thr = precision_recall_curve(y_true, p_high)
        ap = average_precision_score(y_true, p_high)
        ax.plot(rec, prec, label=f"PR curve (AP={ap:.3f})")
        for _, row in rec_df.iterrows():
            t = row["threshold"]
            y_hat = (p_high >= t).astype(int)
            evt = _binary_event_metrics(y_true, y_hat)
            ax.scatter(evt["recall_pod"], evt["precision"], s=25)
            ax.text(evt["recall_pod"], evt["precision"], f" t={t:.3f}", fontsize=8)
    ax.set_xlabel("Recall(POD)")
    ax.set_ylabel("Precision(1-FAR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"pr_with_operating_points_{model_name}.png", dpi=220)
    plt.close(fig)


def summarize_cv_metrics(metrics_df: pd.DataFrame, outpath: Path, model_name: str) -> pd.DataFrame:
    rows = []
    for col in ["f1_macro", "f1_weighted", "pr_auc_high", "brier_high", "recall_pod", "far", "csi"]:
        m, lo, hi, n = metric_ci95(metrics_df[col].values)
        rows.append({"model": model_name, "metric": col, "mean": m, "ci95_lo": lo, "ci95_hi": hi, "n_folds": n})
    out = pd.DataFrame(rows)
    out.to_csv(outpath, index=False)
    return out


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray, high_label: int = 2) -> Dict[str, float]:
    y_true_high = (y_true == high_label).astype(int)
    y_pred_high = (y_pred == high_label).astype(int)
    evt = _binary_event_metrics(y_true_high, y_pred_high)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else np.nan,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else np.nan,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")) if len(y_true) else np.nan,
        "pr_auc_high": _safe_ap(y_true_high, prob[:, high_label]) if prob.size else np.nan,
        "brier_high": _safe_brier(y_true_high, prob[:, high_label]) if prob.size else np.nan,
        **evt,
    }


def _fit_full_model_for_external(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    kind: str,
    inner_splits: int,
    n_jobs: int,
):
    num_cols, cat_cols = choose_feature_columns(df_train)
    feature_cols = num_cols + cat_cols
    X_train = df_train[feature_cols].copy()
    groups = df_train["Pit_ID"].values
    pre = build_preprocessor(num_cols, cat_cols)
    base, grid = build_model(kind, n_jobs=n_jobs)
    pipe = Pipeline([("pre", pre), ("clf", base)])
    fitted, best_params = _fit_with_calibration(
        pipe,
        grid,
        X_train,
        y_train,
        groups,
        inner_splits=inner_splits,
        n_jobs=n_jobs,
    )
    return fitted, best_params, feature_cols


def run_external_design_validation(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_external: pd.DataFrame,
    y_external: np.ndarray,
    outdir: Path,
    model_name: str,
    inner_splits: int,
    n_jobs: int,
    high_label: int,
) -> None:
    if df_external.empty:
        return
    fitted, best_params, feature_cols = _fit_full_model_for_external(
        df_train,
        y_train,
        kind=model_name,
        inner_splits=inner_splits,
        n_jobs=n_jobs,
    )
    X_ext = df_external[feature_cols].copy()
    prob = fitted.predict_proba(X_ext)
    y_pred = np.argmax(prob, axis=1)
    metrics = _multiclass_metrics(y_external, y_pred, prob, high_label=high_label)
    pd.DataFrame([{ "model": model_name, "n_external": int(len(y_external)), **metrics, **best_params }]).to_csv(
        outdir / f"external_design_metrics_{model_name}.csv",
        index=False,
    )

    pred = pd.DataFrame(
        {
            "idx": df_external.index.values,
            "Run_ID": df_external["Run_ID"].values if "Run_ID" in df_external.columns else np.nan,
            "Pit_ID": df_external["Pit_ID"].values,
            "Rainfall_mm": df_external["Rainfall_mm"].values,
            "Rainfall_Pattern": df_external["Rainfall_Pattern"].values,
            "Rainfall_Duration_s": df_external["Rainfall_Duration_s"].values if "Rainfall_Duration_s" in df_external.columns else np.nan,
            "Scenario_Type": df_external["Scenario_Type"].values if "Scenario_Type" in df_external.columns else np.nan,
            "Return_Period_yr": df_external["Return_Period_yr"].values if "Return_Period_yr" in df_external.columns else np.nan,
            "true": y_external,
            "pred": y_pred,
            "p_low": prob[:, 0],
            "p_mid": prob[:, 1],
            "p_high": prob[:, 2],
            "y_ratio_raw": df_external["y_ratio_raw"].values,
            "Final_Inundation_Depth_m": df_external["Final_Inundation_Depth_m"].values,
        }
    )
    pred.to_csv(outdir / f"external_design_predictions_{model_name}.csv", index=False)

    rec_path = outdir / f"threshold_recommend_{model_name}.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        rows = []
        y_true_high = (y_external == high_label).astype(int)
        for _, row in rec.iterrows():
            t = float(row["threshold"])
            y_hat_high = (prob[:, high_label] >= t).astype(int)
            evt = _binary_event_metrics(y_true_high, y_hat_high)
            rows.append({"model": model_name, "criterion": row["criterion"], "threshold": t, **evt})
        pd.DataFrame(rows).to_csv(outdir / f"external_design_threshold_metrics_{model_name}.csv", index=False)

    _write_external_design_diagnostics(pred, outdir, model_name, high_label=high_label)
    joblib.dump(fitted, outdir / f"external_full_model_{model_name}.joblib")


def _write_external_design_diagnostics(pred: pd.DataFrame, outdir: Path, model_name: str, high_label: int = 2) -> None:
    pit = pred.groupby("Pit_ID").agg(
        true_mean_ratio=("y_ratio_raw", "mean"),
        pred_mean_high_prob=("p_high", "mean"),
        true_high_rate=("true", lambda s: float((s == high_label).mean())),
        pred_high_rate=("pred", lambda s: float((s == high_label).mean())),
    ).reset_index()
    rho_prob, p_prob = spearmanr(pit["true_mean_ratio"], pit["pred_mean_high_prob"])
    rho_rate, p_rate = spearmanr(pit["true_high_rate"], pit["pred_high_rate"])
    pd.DataFrame(
        [
            {
                "model": model_name,
                "pit_rank_spearman_prob": float(rho_prob),
                "pit_rank_p_prob": float(p_prob),
                "pit_rank_spearman_high_rate": float(rho_rate),
                "pit_rank_p_high_rate": float(p_rate),
            }
        ]
    ).to_csv(outdir / f"external_design_pit_rank_{model_name}.csv", index=False)

    rp = pred.groupby("Return_Period_yr", dropna=False).agg(
        true_mean_ratio=("y_ratio_raw", "mean"),
        pred_mean_high_prob=("p_high", "mean"),
        true_high_rate=("true", lambda s: float((s == high_label).mean())),
        pred_high_rate=("pred", lambda s: float((s == high_label).mean())),
    ).reset_index().sort_values("Return_Period_yr")
    rp["model"] = model_name
    rp["true_ratio_monotonic"] = bool(rp["true_mean_ratio"].is_monotonic_increasing)
    rp["pred_prob_monotonic"] = bool(rp["pred_mean_high_prob"].is_monotonic_increasing)
    rp["true_high_rate_monotonic"] = bool(rp["true_high_rate"].is_monotonic_increasing)
    rp["pred_high_rate_monotonic"] = bool(rp["pred_high_rate"].is_monotonic_increasing)
    rp.to_csv(outdir / f"external_design_return_period_monotonicity_{model_name}.csv", index=False)


def slice_metrics(pred_df: pd.DataFrame, full_df: pd.DataFrame, outdir: Path, model_name: str, high_label: int = 2) -> None:
    if pred_df.empty:
        return
    merged = pred_df.copy()
    needed = [
        "Rainfall_mm",
        "Rainfall_Pattern",
        "Rainfall_Duration_s",
        "Scenario_Type",
        "Design_Region",
        "Return_Period_yr",
        "Chicago_Peak_Ratio",
        "Rain_File_Peak_Intensity_mm_h",
    ]
    missing_needed = [c for c in needed if c not in merged.columns]
    if missing_needed:
        add_cols = ["_row_idx"] + [c for c in missing_needed if c in full_df.columns]
        lookup = full_df.reset_index(names="_row_idx")[add_cols]
        merged = merged.merge(
            lookup,
            left_on="idx",
            right_on="_row_idx",
            how="left",
        ).drop(columns=["_row_idx"], errors="ignore")
    merged["rain_bin"] = pd.cut(
        merged["Rainfall_mm"],
        bins=[-np.inf, 30, 60, 100, np.inf],
        labels=["<=30", "30-60", "60-100", ">100"],
    )

    def _slice_one(group_col: str, fname: str) -> None:
        rows = []
        for g, sub in merged.groupby(group_col, dropna=False, observed=True):
            if sub.empty:
                continue
            y = sub["true"].values
            yp = sub["pred"].values
            y_bin = (y == high_label).astype(int)
            yp_bin = (yp == high_label).astype(int)
            p = sub["p_high"].values
            evt = _binary_event_metrics(y_bin, yp_bin)
            rows.append(
                {
                    "group": g,
                    "n": int(len(sub)),
                    "f1_macro": float(f1_score(y, yp, average="macro")),
                    "pr_auc_high": _safe_ap(y_bin, p),
                    **evt,
                }
            )
        pd.DataFrame(rows).to_csv(outdir / fname, index=False)

    _slice_one("Rainfall_Pattern", f"slice_metrics_pattern_{model_name}.csv")
    _slice_one("rain_bin", f"slice_metrics_rainbin_{model_name}.csv")
    if "Scenario_Type" in merged.columns and merged["Scenario_Type"].notna().sum() > 0 and merged["Scenario_Type"].dropna().nunique() > 1:
        _slice_one("Scenario_Type", f"slice_metrics_scenario_type_{model_name}.csv")
    if "Return_Period_yr" in merged.columns and merged["Return_Period_yr"].notna().sum() > 0 and merged["Return_Period_yr"].dropna().nunique() > 1:
        _slice_one("Return_Period_yr", f"slice_metrics_return_period_{model_name}.csv")
    if "Chicago_Peak_Ratio" in merged.columns and merged["Chicago_Peak_Ratio"].notna().sum() > 0 and merged["Chicago_Peak_Ratio"].dropna().nunique() > 1:
        _slice_one("Chicago_Peak_Ratio", f"slice_metrics_chicago_peak_ratio_{model_name}.csv")
    if merged["Rainfall_Duration_s"].notna().sum() > 0 and merged["Rainfall_Duration_s"].dropna().nunique() > 1:
        _slice_one("Rainfall_Duration_s", f"slice_metrics_duration_{model_name}.csv")


def run_step3(args: argparse.Namespace) -> None:
    outdir = ensure_dir(args.outdir)
    prepared = load_events_table(args.input, sheet_name=args.sheet_name)
    df = prepared.df.copy()

    # Step3 use rows with complete ML features/targets
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "Pit_ID",
            "Rainfall_mm",
            "Rainfall_Pattern",
            "Pit_Area_m2",
            "Pit_Max_Depth_m",
            "Pit_Avg_Depth_m",
            "Final_Inundation_Depth_m",
            "RF_over_AvgDepth",
            "RF_over_MaxDepth",
            "y_ratio_raw",
        ]
    ).copy()

    if "Scenario_Type" in df.columns:
        df_train = df[df["Scenario_Type"].eq(args.train_scenario_type)].copy()
        df_external = df[df["Scenario_Type"].eq(args.external_scenario_type)].copy()
    else:
        df_train = df.copy()
        df_external = df.iloc[0:0].copy()

    abs_thresh = tuple(args.abs_thresh) if args.abs_thresh is not None else None
    y, scheme = build_labels(
        df_train,
        label_mode=args.label_mode,
        rel_thresh=tuple(args.rel_thresh),
        abs_thresh=abs_thresh,
    )
    df_train["RiskLabel"] = y
    if not df_external.empty:
        y_external, _ = build_labels(
            df_external,
            label_mode=args.label_mode,
            rel_thresh=tuple(args.rel_thresh),
            abs_thresh=abs_thresh,
        )
        df_external["RiskLabel"] = y_external
    else:
        y_external = np.array([], dtype=int)

    report = dict(prepared.report)
    report.update(
        {
            "analysis_design": "A=idealized_only_training; B=design_chicago_external_validation",
            "train_scenario_type": args.train_scenario_type,
            "external_scenario_type": args.external_scenario_type,
            "n_train_rows": int(len(df_train)),
            "n_external_rows": int(len(df_external)),
            "train_label_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
            "external_label_counts": {str(k): int(v) for k, v in pd.Series(y_external).value_counts().sort_index().items()},
        }
    )
    write_json(outdir / "step3_data_quality.json", report)
    write_json(outdir / "step3_label_scheme.json", scheme)
    num_cols, cat_cols = choose_feature_columns(df_train)
    write_json(
        outdir / "step3_feature_columns.json",
        {
            "numeric": num_cols,
            "categorical": cat_cols,
            "all": num_cols + cat_cols,
        },
    )
    write_json(
        outdir / "step3_run_config.json",
        {
            "models": args.models,
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "perm_repeats": args.perm_repeats,
            "n_jobs": args.n_jobs,
            "cost_fn": args.cost_fn,
            "cost_fp": args.cost_fp,
            "recall_floor": args.recall_floor,
            "train_scenario_type": args.train_scenario_type,
            "external_scenario_type": args.external_scenario_type,
        },
    )

    summary_rows = []
    for kind in args.models:
        if kind == "xgb" and XGBClassifier is None:
            print("[Step3] Skip xgb: xgboost is not available.")
            continue

        mdf, pdf, _ = evaluate_one_model(
            df=df_train,
            y=y,
            kind=kind,
            outdir=outdir,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            perm_repeats=args.perm_repeats,
            high_label=args.high_label,
            n_jobs=args.n_jobs,
        )
        scan_thresholds(
            pred_df=pdf,
            outdir=outdir,
            model_name=kind,
            high_label=args.high_label,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            recall_floor=args.recall_floor,
        )
        slice_metrics(pdf, df_train, outdir, kind, high_label=args.high_label)
        summary = summarize_cv_metrics(mdf, outdir / f"summary_{kind}.csv", kind)
        summary_rows.append(summary)
        run_external_design_validation(
            df_train=df_train,
            y_train=y,
            df_external=df_external,
            y_external=y_external,
            outdir=outdir,
            model_name=kind,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
            high_label=args.high_label,
        )

        # Combined confusion matrix
        cm = confusion_matrix(pdf["true"], pdf["pred"], labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_yticklabels(["low", "mid", "high"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(outdir / f"confusion_matrix_{kind}_combined.png", dpi=220)
        plt.close(fig)

    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(outdir / "summary_compare_models.csv", index=False)
    print(f"[Step3] Done. Outputs saved to: {outdir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refactored Step3 machine-learning pipeline.")
    p.add_argument("--input", required=True, help="Input event table (.csv/.xlsx)")
    p.add_argument("--sheet-name", default=0, help="Sheet name/index for Excel input")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    p.add_argument("--label-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--rel-thresh", nargs=2, type=float, default=[0.30, 0.60])
    p.add_argument("--abs-thresh", nargs=2, type=float, default=None)
    p.add_argument("--high-label", type=int, default=2)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--perm-repeats", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearch and models")
    p.add_argument("--cost-fn", type=float, default=3.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--recall-floor", type=float, default=0.90)
    p.add_argument("--train-scenario-type", default="IDEALIZED", help="Scenario_Type used for main training A")
    p.add_argument("--external-scenario-type", default="DESIGN_CHICAGO", help="Scenario_Type used for external validation B")
    return p


if __name__ == "__main__":
    run_step3(build_arg_parser().parse_args())
