#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrices, dmatrix
from scipy.special import expit
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from .common import (
        ensure_dir,
        grouped_kfold,
        load_events_table,
        metric_ci95,
        write_json,
    )
except ImportError:
    from common import ensure_dir, grouped_kfold, load_events_table, metric_ci95, write_json

warnings.filterwarnings("ignore", category=UserWarning)

LME_FORMULA_DEFAULT = (
    "y_logit ~ Rainfall_mm + C(Rainfall_Pattern)"
    " + Pit_Max_Depth_m + RF_over_AvgDepth + Rainfall_Duration_s"
    " + Rainfall_mm:Pit_Max_Depth_m + RF_over_AvgDepth:Pit_Max_Depth_m"
)


def _nunique_nonnull(df: pd.DataFrame, col: str) -> int:
    return int(df[col].dropna().nunique()) if col in df.columns else 0


def build_auto_lme_formula(df: pd.DataFrame) -> str:
    terms = [
        "Rainfall_mm",
        "C(Rainfall_Pattern)",
        "Pit_Max_Depth_m",
        "RF_over_AvgDepth",
    ]
    # Keep the mixed model conservative. Many optional design-rain fields are
    # deterministic functions of pattern/duration/rainfall and can make the
    # fixed-effect design singular.
    optional_numeric = ["Rainfall_Duration_s"]
    for col in optional_numeric:
        if _nunique_nonnull(df, col) > 1:
            terms.append(col)
    # Include estimable interaction terms described in Methods. The candidate
    # RF_over_MaxDepth:Pit_Max_Depth_m is not included here because it is
    # algebraically equivalent to Rainfall_mm.
    if _nunique_nonnull(df, "Rainfall_mm") > 1 and _nunique_nonnull(df, "Pit_Max_Depth_m") > 1:
        terms.append("Rainfall_mm:Pit_Max_Depth_m")
    if _nunique_nonnull(df, "RF_over_AvgDepth") > 1 and _nunique_nonnull(df, "Pit_Max_Depth_m") > 1:
        terms.append("RF_over_AvgDepth:Pit_Max_Depth_m")
    return "y_logit ~ " + " + ".join(terms)


def build_lme_formula_notes() -> Dict[str, object]:
    return {
        "interaction_terms_included": [
            "Rainfall_mm:Pit_Max_Depth_m",
            "RF_over_AvgDepth:Pit_Max_Depth_m",
        ],
        "interaction_terms_considered_but_excluded": [
            {
                "term": "RF_over_MaxDepth:Pit_Max_Depth_m",
                "reason": (
                    "Exact collinearity with Rainfall_mm because "
                    "RF_over_MaxDepth = Rainfall_mm / Pit_Max_Depth_m."
                ),
            }
        ],
    }


def run_descriptive_summaries(
    df: pd.DataFrame,
    outdir: Path,
    target_col: str = "Final_Inundation_Depth_m",
) -> None:
    metric_cols = [
        target_col,
        "Hmax_m",
        "y_ratio_raw",
        "Peak_to_Potential_Ratio",
        "Rainfall_Intensity_mm_per_h",
        "Rain_File_Peak_Intensity_mm_h",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    group_sets = [
        ["Pit_ID"],
        ["Rainfall_mm"],
        ["Scenario_Type"],
        ["Rainfall_Pattern"],
        ["Scenario_Type", "Rainfall_Pattern"],
        ["Rainfall_Duration_s"],
        ["Rainfall_mm", "Rainfall_Duration_s", "Rainfall_Pattern"],
        ["Rainfall_mm", "Rainfall_Duration_s", "Rainfall_Pattern", "Pit_ID"],
        ["Return_Period_yr"],
        ["Chicago_Peak_Ratio"],
        ["Scenario_Type", "Rainfall_Duration_s", "Return_Period_yr", "Chicago_Peak_Ratio"],
    ]
    for group_cols in group_sets:
        cols = [c for c in group_cols if c in df.columns and _nunique_nonnull(df, c) > 0]
        if not cols:
            continue
        agg = df.groupby(cols, dropna=False)[metric_cols].agg(["count", "mean", "std", "median", "min", "max"])
        agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns.to_flat_index()]
        name = "_".join(cols).lower()
        agg.reset_index().to_csv(outdir / f"step2_descriptive_by_{name}.csv", index=False)

    scenario_cols = [
        "Source_Run_Dir",
        "Rainfall_mm",
        "Rainfall_Pattern",
        "Rainfall_Duration_s",
        "Scenario_Type",
        "Design_Region",
        "Return_Period_yr",
        "Chicago_Peak_Ratio",
        "Rain_File_Total_mm",
        "Rain_File_Peak_Intensity_mm_h",
    ]
    scenario_cols = [c for c in scenario_cols if c in df.columns]
    if "Source_Run_Dir" in df.columns:
        per_scenario = (
            df.groupby(scenario_cols, dropna=False)
            .agg(
                n_pits=("Pit_ID", "nunique"),
                final_mean_m=(target_col, "mean"),
                final_max_m=(target_col, "max"),
                hmax_mean_m=("Hmax_m", "mean") if "Hmax_m" in df.columns else (target_col, "mean"),
                high_ratio_count=("y_ratio_raw", lambda s: int((s >= 0.6).sum())) if "y_ratio_raw" in df.columns else (target_col, "count"),
            )
            .reset_index()
        )
        per_scenario.to_csv(outdir / "step2_scenario_summary.csv", index=False)


def fill_optional_model_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "Return_Period_yr",
        "Chicago_Peak_Ratio",
        "Rain_File_Total_mm",
        "Rain_File_Peak_Intensity_mm_h",
        "Rain_File_Interval_s",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    for col in ["Scenario_Type", "Design_Region"]:
        if col in out.columns:
            out[col] = out[col].fillna("NONE").astype(str).replace({"nan": "NONE", "": "NONE"})
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    # Convert through ndarray to avoid pandas index alignment between original
    # row indices and 0-based prediction arrays.
    y_t = pd.Series(np.asarray(list(y_true), dtype="float64"))
    y_p = pd.Series(np.asarray(list(y_pred), dtype="float64"))
    mask = y_t.notna() & y_p.notna() & np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n_eval": 0}
    yt = y_t[mask].to_numpy()
    yp = y_p[mask].to_numpy()
    r2 = np.nan if len(yt) < 2 else r2_score(yt, yp)
    return {
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": _rmse(yt, yp),
        "R2": r2,
        "n_eval": int(len(yt)),
    }


def _fit_mixedlm_robust(formula: str, data: pd.DataFrame, groups: pd.Series):
    md = smf.mixedlm(formula, data=data, groups=groups)
    errors = []
    for reml in (True, False):
        for method in ("lbfgs", "powell", "nm", "cg"):
            try:
                fit = md.fit(method=method, maxiter=2000, disp=False, reml=reml)
                if not getattr(fit, "converged", False):
                    raise RuntimeError("optimizer returned non-converged fit")
                if not np.isfinite(float(fit.llf)):
                    raise RuntimeError(f"optimizer returned non-finite log-likelihood: {fit.llf}")
                if not np.isfinite(fit.fe_params.to_numpy(dtype="float64")).all():
                    raise RuntimeError("optimizer returned non-finite fixed-effect coefficients")
                return fit, {"method": method, "reml": reml}
            except Exception as exc:
                errors.append(f"{method}/reml={reml}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All MixedLM fits failed; " + " | ".join(errors))


def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    for i, idx in enumerate(order):
        adj[idx] = min((m - i) * pvals[idx], 1.0)
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    return adj.tolist()


def run_nonparametric_pattern_tests(
    df: pd.DataFrame,
    outdir: Path,
    target_col: str = "Final_Inundation_Depth_m",
) -> None:
    strata_cols = ["Rainfall_mm"]
    if "Rainfall_Duration_s" in df.columns and df["Rainfall_Duration_s"].dropna().nunique() > 1:
        strata_cols.append("Rainfall_Duration_s")

    friedman_rows = []
    pair_rows = []

    for key, sub in df.groupby(strata_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = {strata_cols[i]: key_tuple[i] for i in range(len(strata_cols))}

        wide = sub.pivot_table(
            index="Pit_ID",
            columns="Rainfall_Pattern",
            values=target_col,
            aggfunc="mean",
        )
        if wide.empty:
            continue

        ordered = [p for p in ["FRONT", "UNIFORM", "BACK"] if p in wide.columns]
        ordered += [p for p in sorted(wide.columns) if p not in ordered]
        wide = wide[ordered].dropna(axis=0, how="any")
        n_pits, k_patterns = wide.shape

        if n_pits < 3 or k_patterns < 2:
            continue

        if k_patterns >= 3:
            arrays = [wide[c].values for c in ordered]
            stat, pval = friedmanchisquare(*arrays)
            kendall_w = stat / (n_pits * (k_patterns - 1))
        else:
            stat, pval, kendall_w = np.nan, np.nan, np.nan

        friedman_rows.append(
            {
                **stratum,
                "n_pits": int(n_pits),
                "k_patterns": int(k_patterns),
                "friedman_chi2": float(stat) if np.isfinite(stat) else np.nan,
                "friedman_p": float(pval) if np.isfinite(pval) else np.nan,
                "kendall_w": float(kendall_w) if np.isfinite(kendall_w) else np.nan,
            }
        )

        pairs = list(itertools.combinations(ordered, 2))
        raw_pvals = []
        tmp_rows = []
        for a, b in pairs:
            x = wide[a].values
            y = wide[b].values
            try:
                w_stat, p_raw = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w_stat, p_raw = 0.0, 1.0
            dir_mean = f"{a}>{b}" if np.nanmean(x) > np.nanmean(y) else (f"{a}<{b}" if np.nanmean(x) < np.nanmean(y) else f"{a}≈{b}")
            tmp_rows.append({**stratum, "pair": f"{a} vs {b}", "wilcoxon_w": float(w_stat), "p_raw": float(p_raw), "direction": dir_mean})
            raw_pvals.append(float(p_raw))

        p_holm = _holm_adjust(raw_pvals)
        for row, p_adj in zip(tmp_rows, p_holm):
            row["p_holm"] = float(p_adj)
            row["significant_0_05"] = bool(p_adj < 0.05)
            pair_rows.append(row)

    fried_df = pd.DataFrame(friedman_rows).sort_values(strata_cols)
    pair_df = pd.DataFrame(pair_rows).sort_values(strata_cols + ["pair"])

    fried_df.to_csv(outdir / "step2_friedman_summary.csv", index=False)
    pair_df.to_csv(outdir / "step2_pairwise_wilcoxon_holm.csv", index=False)


def run_spearman_vif(df: pd.DataFrame, outdir: Path) -> None:
    numeric_candidates = [
        "Rainfall_mm",
        "Rainfall_Duration_s",
        "Rainfall_Intensity_mm_per_h",
        "Pit_Area_m2",
        "Pit_Max_Depth_m",
        "Pit_Avg_Depth_m",
        "Pit_Volume_m3",
        "RF_over_AvgDepth",
        "RF_over_MaxDepth",
        "Final_Inundation_Depth_m",
        "y_ratio_raw",
    ]
    cols = [c for c in numeric_candidates if c in df.columns]

    sp_rows = []
    for a, b in itertools.combinations(cols, 2):
        sub = df[[a, b]].dropna()
        if len(sub) < 3:
            continue
        rho, pval = spearmanr(sub[a], sub[b])
        sp_rows.append({"var1": a, "var2": b, "spearman_rho": float(rho), "p_value": float(pval), "n": int(len(sub))})
    pd.DataFrame(sp_rows).sort_values("spearman_rho", ascending=False).to_csv(outdir / "step2_spearman_pairs.csv", index=False)

    vif_base = ["Rainfall_mm", "Pit_Area_m2", "Pit_Max_Depth_m", "Pit_Avg_Depth_m", "RF_over_AvgDepth", "RF_over_MaxDepth"]
    if "Rainfall_Duration_s" in df.columns and df["Rainfall_Duration_s"].dropna().nunique() > 1:
        vif_base.append("Rainfall_Duration_s")
    vif_cols = [c for c in vif_base if c in df.columns]

    vdf = df[vif_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vdf) >= len(vif_cols) + 2 and len(vif_cols) >= 2:
        X = sm.add_constant(vdf, has_constant="add")
        rows = []
        for i, c in enumerate(X.columns):
            if c == "const":
                continue
            rows.append({"feature": c, "vif": float(variance_inflation_factor(X.values, i))})
        pd.DataFrame(rows).sort_values("vif", ascending=False).to_csv(outdir / "step2_vif.csv", index=False)


def evaluate_baseline(df: pd.DataFrame, n_splits: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gkf = grouped_kfold(df, group_col="Pit_ID", n_splits=n_splits)

    strata = ["Rainfall_mm", "Rainfall_Pattern"]
    if "Rainfall_Duration_s" in df.columns and df["Rainfall_Duration_s"].dropna().nunique() > 1:
        strata.append("Rainfall_Duration_s")

    metric_rows = []
    pred_rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=df["Pit_ID"]), 1):
        train = df.iloc[tr].copy()
        test = df.iloc[te].copy()

        baseline_map = train.groupby(strata, dropna=False)["y_ratio_raw"].mean().reset_index(name="pred_ratio")
        global_mean = float(train["y_ratio_raw"].mean())

        key_df = test[strata].copy()
        key_df["_row"] = np.arange(len(key_df))
        merged = key_df.merge(baseline_map, on=strata, how="left").sort_values("_row")
        pred_ratio = merged["pred_ratio"].fillna(global_mean).to_numpy()
        pred_abs = pred_ratio * test["Pit_Max_Depth_m"].to_numpy()

        ratio_m = _safe_metrics(test["y_ratio_raw"], pred_ratio)
        abs_m = _safe_metrics(test["Final_Inundation_Depth_m"], pred_abs)

        metric_rows.append(
            {
                "fold": fold,
                "model": "baseline",
                "ratio_MAE": ratio_m["MAE"],
                "ratio_RMSE": ratio_m["RMSE"],
                "ratio_R2": ratio_m["R2"],
                "abs_MAE_m": abs_m["MAE"],
                "abs_RMSE_m": abs_m["RMSE"],
                "abs_R2": abs_m["R2"],
                "n_eval": abs_m["n_eval"],
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "baseline",
                    "Run_ID": test["Run_ID"] if "Run_ID" in test.columns else np.nan,
                    "Pit_ID": test["Pit_ID"].values,
                    "Rainfall_mm": test["Rainfall_mm"].values,
                    "Rainfall_Pattern": test["Rainfall_Pattern"].values,
                    "Rainfall_Duration_s": test["Rainfall_Duration_s"].values if "Rainfall_Duration_s" in test.columns else np.nan,
                    "y_true_ratio": test["y_ratio_raw"].values,
                    "y_pred_ratio": pred_ratio,
                    "y_true_abs_m": test["Final_Inundation_Depth_m"].values,
                    "y_pred_abs_m": pred_abs,
                }
            )
        )
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), pred_df


def evaluate_lme(
    df: pd.DataFrame,
    outdir: Path,
    n_splits: int = 5,
    formula: str = LME_FORMULA_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gkf = grouped_kfold(df, group_col="Pit_ID", n_splits=n_splits)
    rhs = formula.split("~", 1)[1]

    metric_rows = []
    pred_rows = []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=df["Pit_ID"]), 1):
        train = df.iloc[tr].copy()
        test = df.iloc[te].copy()
        try:
            fit, fit_info = _fit_mixedlm_robust(formula, data=train, groups=train["Pit_ID"])
        except Exception as exc:
            metric_rows.append({"fold": fold, "model": "lme", "fit_error": str(exc)})
            continue

        X_te = dmatrix(rhs, test, return_type="dataframe")
        fe = fit.fe_params
        X_te = X_te.reindex(columns=fe.index, fill_value=0.0)
        pred_logit = np.dot(X_te.values, fe.values)
        pred_ratio = expit(pred_logit)
        pred_abs = pred_ratio * test["Pit_Max_Depth_m"].values

        ratio_m = _safe_metrics(test["y_ratio_clip"], pred_ratio)
        abs_m = _safe_metrics(test["Final_Inundation_Depth_m"], pred_abs)
        metric_rows.append(
            {
                "fold": fold,
                "model": "lme",
                "ratio_MAE": ratio_m["MAE"],
                "ratio_RMSE": ratio_m["RMSE"],
                "ratio_R2": ratio_m["R2"],
                "abs_MAE_m": abs_m["MAE"],
                "abs_RMSE_m": abs_m["RMSE"],
                "abs_R2": abs_m["R2"],
                "n_eval": abs_m["n_eval"],
                "fit_method": fit_info["method"],
                "fit_reml": fit_info["reml"],
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "lme",
                    "Run_ID": test["Run_ID"] if "Run_ID" in test.columns else np.nan,
                    "Pit_ID": test["Pit_ID"].values,
                    "Rainfall_mm": test["Rainfall_mm"].values,
                    "Rainfall_Pattern": test["Rainfall_Pattern"].values,
                    "Rainfall_Duration_s": test["Rainfall_Duration_s"].values if "Rainfall_Duration_s" in test.columns else np.nan,
                    "y_true_ratio": test["y_ratio_clip"].values,
                    "y_pred_ratio": pred_ratio,
                    "y_true_abs_m": test["Final_Inundation_Depth_m"].values,
                    "y_pred_abs_m": pred_abs,
                }
            )
        )

    # fit full model for coefficients/summary
    try:
        fit_full, fit_info = _fit_mixedlm_robust(formula, data=df, groups=df["Pit_ID"])
        with open(outdir / "step2_lme_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Fit method: {fit_info['method']}; REML: {fit_info['reml']}\n\n")
            f.write(str(fit_full.summary()))
        fe = fit_full.fe_params
        pd.DataFrame(
            {
                "term": fe.index,
                "coef": fe.values,
                "p_value": fit_full.pvalues.reindex(fe.index).values,
                "std_err": fit_full.bse.reindex(fe.index).values,
            }
        ).to_csv(outdir / "step2_lme_fixed_effects.csv", index=False)
    except Exception as exc:
        with open(outdir / "step2_lme_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"LME full-model fit failed: {exc}\nFormula: {formula}\n")
        pd.DataFrame([{"fit_error": str(exc), "formula": formula}]).to_csv(outdir / "step2_lme_fixed_effects.csv", index=False)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), pred_df


def evaluate_spline_ols(
    df: pd.DataFrame,
    outdir: Path,
    n_splits: int = 5,
    df_rf: int = 5,
    df_rainfall: int = 5,
    bounds_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bdf = bounds_df if bounds_df is not None else df
    rf_upper = max(float(bdf["RF_over_AvgDepth"].replace([np.inf, -np.inf], np.nan).max()) * 1.001, 1.0)
    rain_upper = max(float(bdf["Rainfall_mm"].replace([np.inf, -np.inf], np.nan).max()) * 1.001, 1.0)
    terms = [
        f"bs(RF_over_AvgDepth, df={df_rf}, lower_bound=0, upper_bound={rf_upper:.12g})",
        f"bs(Rainfall_mm, df={df_rainfall}, lower_bound=0, upper_bound={rain_upper:.12g})",
        "Pit_Max_Depth_m",
        "C(Rainfall_Pattern)",
    ]
    optional_numeric = ["Rainfall_Duration_s"]
    for col in optional_numeric:
        if _nunique_nonnull(df, col) > 1:
            terms.append(col)
    formula = "y_logit ~ " + " + ".join(terms)
    gkf = grouped_kfold(df, group_col="Pit_ID", n_splits=n_splits)
    metric_rows = []
    pred_rows = []

    for fold, (tr, te) in enumerate(gkf.split(df, groups=df["Pit_ID"]), 1):
        train = df.iloc[tr].copy()
        test = df.iloc[te].copy()
        y_tr, X_tr = dmatrices(formula, train, return_type="dataframe")
        X_te = build_design_matrices([X_tr.design_info], test, return_type="dataframe")[0]
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

        fit = sm.OLS(y_tr, X_tr).fit()
        pred_logit = fit.predict(X_te)
        pred_ratio = expit(pred_logit)
        pred_abs = pred_ratio * test["Pit_Max_Depth_m"].values

        ratio_m = _safe_metrics(test["y_ratio_clip"], pred_ratio)
        abs_m = _safe_metrics(test["Final_Inundation_Depth_m"], pred_abs)
        metric_rows.append(
            {
                "fold": fold,
                "model": "spline_ols",
                "ratio_MAE": ratio_m["MAE"],
                "ratio_RMSE": ratio_m["RMSE"],
                "ratio_R2": ratio_m["R2"],
                "abs_MAE_m": abs_m["MAE"],
                "abs_RMSE_m": abs_m["RMSE"],
                "abs_R2": abs_m["R2"],
                "n_eval": abs_m["n_eval"],
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "spline_ols",
                    "Run_ID": test["Run_ID"] if "Run_ID" in test.columns else np.nan,
                    "Pit_ID": test["Pit_ID"].values,
                    "Rainfall_mm": test["Rainfall_mm"].values,
                    "Rainfall_Pattern": test["Rainfall_Pattern"].values,
                    "Rainfall_Duration_s": test["Rainfall_Duration_s"].values if "Rainfall_Duration_s" in test.columns else np.nan,
                    "y_true_ratio": test["y_ratio_clip"].values,
                    "y_pred_ratio": pred_ratio,
                    "y_true_abs_m": test["Final_Inundation_Depth_m"].values,
                    "y_pred_abs_m": pred_abs,
                }
            )
        )

    y_full, X_full = dmatrices(formula, df, return_type="dataframe")
    fit_full = sm.OLS(y_full, X_full).fit()
    with open(outdir / "step2_spline_ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(fit_full.summary()))
    pd.DataFrame(
        {
            "term": fit_full.params.index,
            "coef": fit_full.params.values,
            "p_value": fit_full.pvalues.values,
            "std_err": fit_full.bse.values,
        }
    ).to_csv(outdir / "step2_spline_ols_coefficients.csv", index=False)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def summarize_cv(metrics_df: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    rows = []
    metric_cols = ["ratio_MAE", "ratio_RMSE", "ratio_R2", "abs_MAE_m", "abs_RMSE_m", "abs_R2"]
    for model, g in metrics_df.groupby("model"):
        for col in metric_cols:
            if col not in g.columns:
                continue
            m, lo, hi, n = metric_ci95(g[col].values)
            rows.append(
                {
                    "model": model,
                    "metric": col,
                    "mean": m,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_folds": n,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outpath, index=False)
    return out


def _risk_class_from_ratio(values: Iterable[float], thresholds: Tuple[float, float] = (0.3, 0.6)) -> np.ndarray:
    t1, t2 = thresholds
    return np.digitize(np.asarray(list(values), dtype=float), bins=[t1, t2]).astype(int)


def _classification_summary(y_true: np.ndarray, y_pred: np.ndarray, high_label: int = 2) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else np.nan
    high_mask = y_true == high_label
    high_recall = float(np.mean(y_pred[high_mask] == high_label)) if high_mask.sum() else np.nan
    return {
        "risk_accuracy": acc,
        "high_recall": high_recall,
        "n_high": int(high_mask.sum()),
    }


def _coerce_unseen_patterns_for_prediction(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if "Rainfall_Pattern" not in train_df.columns or "Rainfall_Pattern" not in pred_df.columns:
        return pred_df
    known = set(train_df["Rainfall_Pattern"].dropna().astype(str).unique())
    if not known:
        return pred_df
    reference = "BACK" if "BACK" in known else sorted(known)[0]
    out = pred_df.copy()
    out["Rainfall_Pattern"] = out["Rainfall_Pattern"].astype(str)
    out.loc[~out["Rainfall_Pattern"].isin(known), "Rainfall_Pattern"] = reference
    return out


def _external_regression_metrics(
    external: pd.DataFrame,
    pred_ratio: np.ndarray,
    model_name: str,
    thresholds: Tuple[float, float] = (0.3, 0.6),
) -> Dict[str, float | str | int]:
    pred_abs = pred_ratio * external["Pit_Max_Depth_m"].to_numpy()
    ratio_m = _safe_metrics(external["y_ratio_clip"], pred_ratio)
    abs_m = _safe_metrics(external["Final_Inundation_Depth_m"], pred_abs)
    y_true_cls = _risk_class_from_ratio(external["y_ratio_raw"], thresholds)
    y_pred_cls = _risk_class_from_ratio(pred_ratio, thresholds)
    cls_m = _classification_summary(y_true_cls, y_pred_cls)
    return {
        "model": model_name,
        "n_eval": abs_m["n_eval"],
        "ratio_MAE": ratio_m["MAE"],
        "ratio_RMSE": ratio_m["RMSE"],
        "ratio_R2": ratio_m["R2"],
        "abs_MAE_m": abs_m["MAE"],
        "abs_RMSE_m": abs_m["RMSE"],
        "abs_R2": abs_m["R2"],
        **cls_m,
    }


def run_design_chicago_external_validation(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    outdir: Path,
    lme_formula: str,
    spline_df_rf: int,
    spline_df_rainfall: int,
    bounds_df: pd.DataFrame,
) -> None:
    if external_df.empty:
        pd.DataFrame().to_csv(outdir / "design_chicago_external_metrics.csv", index=False)
        return

    rows = []
    pred_tables = []

    # LME trained only on idealized scenarios. Unseen CHICAGO category is
    # treated as the reference pattern, so the validation tests hydrologic
    # extrapolation rather than learning a Chicago-specific offset.
    try:
        fit, fit_info = _fit_mixedlm_robust(lme_formula, data=train_df, groups=train_df["Pit_ID"])
        rhs = lme_formula.split("~", 1)[1]
        pred_design_df = _coerce_unseen_patterns_for_prediction(train_df, external_df)
        X_ext = dmatrix(rhs, pred_design_df, return_type="dataframe")
        X_ext = X_ext.reindex(columns=fit.fe_params.index, fill_value=0.0)
        pred_ratio = expit(np.dot(X_ext.values, fit.fe_params.values))
        row = _external_regression_metrics(external_df, pred_ratio, "lme")
        row.update({"fit_method": fit_info["method"], "fit_reml": fit_info["reml"]})
        rows.append(row)
        pred_tables.append(_external_prediction_table(external_df, pred_ratio, "lme"))
    except Exception as exc:
        rows.append({"model": "lme", "fit_error": str(exc)})

    # Spline-OLS trained on idealized scenarios with fixed spline bases.
    try:
        bdf = bounds_df
        rf_upper = max(float(bdf["RF_over_AvgDepth"].replace([np.inf, -np.inf], np.nan).max()) * 1.001, 1.0)
        rain_upper = max(float(bdf["Rainfall_mm"].replace([np.inf, -np.inf], np.nan).max()) * 1.001, 1.0)
        formula = (
            f"y_logit ~ bs(RF_over_AvgDepth, df={spline_df_rf}, lower_bound=0, upper_bound={rf_upper:.12g})"
            f" + bs(Rainfall_mm, df={spline_df_rainfall}, lower_bound=0, upper_bound={rain_upper:.12g})"
            " + Pit_Max_Depth_m + C(Rainfall_Pattern)"
        )
        if _nunique_nonnull(train_df, "Rainfall_Duration_s") > 1:
            formula += " + Rainfall_Duration_s"
        y_tr, X_tr = dmatrices(formula, train_df, return_type="dataframe")
        fit = sm.OLS(y_tr, X_tr).fit()
        pred_design_df = _coerce_unseen_patterns_for_prediction(train_df, external_df)
        X_ext = build_design_matrices([X_tr.design_info], pred_design_df, return_type="dataframe")[0]
        X_ext = X_ext.reindex(columns=X_tr.columns, fill_value=0.0)
        pred_ratio = expit(fit.predict(X_ext))
        rows.append(_external_regression_metrics(external_df, pred_ratio, "spline_ols"))
        pred_tables.append(_external_prediction_table(external_df, pred_ratio, "spline_ols"))
    except Exception as exc:
        rows.append({"model": "spline_ols", "fit_error": str(exc)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "design_chicago_external_metrics.csv", index=False)
    if pred_tables:
        pred_df = pd.concat(pred_tables, ignore_index=True)
        pred_df.to_csv(outdir / "design_chicago_external_predictions.csv", index=False)
        _write_design_validation_diagnostics(pred_df, outdir)


def _external_prediction_table(external_df: pd.DataFrame, pred_ratio: np.ndarray, model_name: str) -> pd.DataFrame:
    out = external_df[
        [
            "Run_ID",
            "Source_Run_Dir",
            "Pit_ID",
            "Rainfall_mm",
            "Rainfall_Duration_s",
            "Rainfall_Pattern",
            "Scenario_Type",
            "Return_Period_yr",
            "Chicago_Peak_Ratio",
            "Final_Inundation_Depth_m",
            "Pit_Max_Depth_m",
            "y_ratio_raw",
        ]
    ].copy()
    out["model"] = model_name
    out["y_pred_ratio"] = pred_ratio
    out["y_pred_abs_m"] = pred_ratio * out["Pit_Max_Depth_m"]
    out["true_risk"] = _risk_class_from_ratio(out["y_ratio_raw"])
    out["pred_risk"] = _risk_class_from_ratio(out["y_pred_ratio"])
    return out


def _write_design_validation_diagnostics(pred_df: pd.DataFrame, outdir: Path) -> None:
    rank_rows = []
    monotonic_rows = []
    for model, sub in pred_df.groupby("model"):
        pit = sub.groupby("Pit_ID").agg(
            true_mean_ratio=("y_ratio_raw", "mean"),
            pred_mean_ratio=("y_pred_ratio", "mean"),
            true_mean_abs_m=("Final_Inundation_Depth_m", "mean"),
            pred_mean_abs_m=("y_pred_abs_m", "mean"),
        ).reset_index()
        rho_ratio, p_ratio = spearmanr(pit["true_mean_ratio"], pit["pred_mean_ratio"])
        rho_abs, p_abs = spearmanr(pit["true_mean_abs_m"], pit["pred_mean_abs_m"])
        rank_rows.append(
            {
                "model": model,
                "pit_rank_spearman_ratio": float(rho_ratio),
                "pit_rank_p_ratio": float(p_ratio),
                "pit_rank_spearman_abs": float(rho_abs),
                "pit_rank_p_abs": float(p_abs),
            }
        )

        rp = sub.groupby("Return_Period_yr", dropna=False).agg(
            true_mean_ratio=("y_ratio_raw", "mean"),
            pred_mean_ratio=("y_pred_ratio", "mean"),
            true_high_rate=("true_risk", lambda s: float((s == 2).mean())),
            pred_high_rate=("pred_risk", lambda s: float((s == 2).mean())),
        ).reset_index().sort_values("Return_Period_yr")
        rp["model"] = model
        rp["true_monotonic_ratio"] = bool(rp["true_mean_ratio"].is_monotonic_increasing)
        rp["pred_monotonic_ratio"] = bool(rp["pred_mean_ratio"].is_monotonic_increasing)
        rp["true_monotonic_high_rate"] = bool(rp["true_high_rate"].is_monotonic_increasing)
        rp["pred_monotonic_high_rate"] = bool(rp["pred_high_rate"].is_monotonic_increasing)
        monotonic_rows.append(rp)

    pd.DataFrame(rank_rows).to_csv(outdir / "design_chicago_pit_rank_validation.csv", index=False)
    pd.concat(monotonic_rows, ignore_index=True).to_csv(outdir / "design_chicago_return_period_monotonicity.csv", index=False)


def run_step2(args: argparse.Namespace) -> None:
    outdir = ensure_dir(args.outdir)
    prepared = load_events_table(args.input, sheet_name=args.sheet_name)
    full_df = prepared.df.copy()
    if "Scenario_Type" in full_df.columns:
        df = full_df[full_df["Scenario_Type"].eq(args.train_scenario_type)].copy()
        external_df = full_df[full_df["Scenario_Type"].eq(args.external_scenario_type)].copy()
    else:
        df = full_df.copy()
        external_df = full_df.iloc[0:0].copy()

    split_report = dict(prepared.report)
    split_report.update(
        {
            "analysis_design": "A=idealized_only_main_analysis; B=design_chicago_external_validation",
            "train_scenario_type": args.train_scenario_type,
            "external_scenario_type": args.external_scenario_type,
            "n_train_rows": int(len(df)),
            "n_external_rows": int(len(external_df)),
            "train_patterns": sorted(df["Rainfall_Pattern"].dropna().unique().tolist()) if "Rainfall_Pattern" in df.columns else [],
            "external_patterns": sorted(external_df["Rainfall_Pattern"].dropna().unique().tolist()) if "Rainfall_Pattern" in external_df.columns else [],
        }
    )
    write_json(outdir / "step2_data_quality.json", split_report)

    # Step2-A: nonparametric and diagnostics
    run_descriptive_summaries(df, outdir, target_col=args.target_col)
    run_nonparametric_pattern_tests(df, outdir, target_col=args.target_col)
    run_spearman_vif(df, outdir)

    # Step2-B/C: grouped regression models
    # LME/Spline需要logit目标，因此需有限值
    required_model_cols = [
        "Pit_ID",
        "Rainfall_mm",
        "Rainfall_Pattern",
        "Pit_Area_m2",
        "Pit_Max_Depth_m",
        "Pit_Avg_Depth_m",
        "Final_Inundation_Depth_m",
        "RF_over_AvgDepth",
        "RF_over_MaxDepth",
        "y_logit",
        "y_ratio_clip",
        "y_ratio_raw",
    ]
    model_df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            *required_model_cols,
        ]
    ).copy()
    external_model_df = external_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_model_cols).copy()
    model_df = fill_optional_model_values(model_df)
    external_model_df = fill_optional_model_values(external_model_df)
    lme_formula = build_auto_lme_formula(model_df) if args.lme_formula.upper() == "AUTO" else args.lme_formula
    formula_record = {
        "lme_formula": lme_formula,
        "spline_formula": "auto",
        **build_lme_formula_notes(),
    }
    write_json(outdir / "step2_model_formulas.json", formula_record)

    baseline_metrics, baseline_pred = evaluate_baseline(model_df, n_splits=args.n_splits)
    lme_metrics, lme_pred = evaluate_lme(model_df, outdir, n_splits=args.n_splits, formula=lme_formula)
    spline_metrics, spline_pred = evaluate_spline_ols(
        model_df,
        outdir,
        n_splits=args.n_splits,
        df_rf=args.spline_df_rf,
        df_rainfall=args.spline_df_rainfall,
        bounds_df=pd.concat([model_df, external_model_df], ignore_index=True) if not external_model_df.empty else model_df,
    )

    all_metrics = pd.concat([baseline_metrics, lme_metrics, spline_metrics], ignore_index=True)
    all_pred = pd.concat([baseline_pred, lme_pred, spline_pred], ignore_index=True)
    all_metrics.to_csv(outdir / "step2_cv_metrics_all_models.csv", index=False)
    all_pred.to_csv(outdir / "step2_cv_predictions_all_models.csv", index=False)
    summarize_cv(all_metrics, outdir / "step2_cv_summary_ci95.csv")

    run_design_chicago_external_validation(
        train_df=model_df,
        external_df=external_model_df,
        outdir=outdir,
        lme_formula=lme_formula,
        spline_df_rf=args.spline_df_rf,
        spline_df_rainfall=args.spline_df_rainfall,
        bounds_df=pd.concat([model_df, external_model_df], ignore_index=True) if not external_model_df.empty else model_df,
    )

    print(f"[Step2] Done. Outputs saved to: {outdir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refactored Step2 statistical analysis pipeline.")
    p.add_argument("--input", required=True, help="Input event table (.csv/.xlsx)")
    p.add_argument("--sheet-name", default=0, help="Sheet name/index for Excel input")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--target-col", default="Final_Inundation_Depth_m", help="Response column for nonparametric tests")
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits by Pit_ID")
    p.add_argument("--spline-df-rf", type=int, default=5, help="Spline df for RF_over_AvgDepth")
    p.add_argument("--spline-df-rainfall", type=int, default=5, help="Spline df for Rainfall_mm")
    p.add_argument("--lme-formula", default="AUTO", help="Statsmodels mixedlm formula, or AUTO")
    p.add_argument("--train-scenario-type", default="IDEALIZED", help="Scenario_Type used for main analysis A")
    p.add_argument("--external-scenario-type", default="DESIGN_CHICAGO", help="Scenario_Type used for external validation B")
    return p


if __name__ == "__main__":
    run_step2(build_arg_parser().parse_args())
