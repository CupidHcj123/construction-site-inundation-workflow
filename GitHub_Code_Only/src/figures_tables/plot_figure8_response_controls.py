#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 8: statistical controls of pit-scale inundation.

Panel A: Spearman correlations for selected predictors against absolute final
         inundation depth and relative inundation ratio.
Panel B: interaction LME fixed effects with 95% CI. Continuous predictors are
         scaled by one standard deviation; categorical terms are contrasts
         relative to BACK.
Panel C: Spline-OLS partial response curves for nonlinear effects of rainfall-
         to-storage ratio and total rainfall on predicted relative inundation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from scipy.special import expit


VARIABLE_LABELS = {
    "Rainfall_mm": "Total rainfall",
    "Rainfall_Duration_s": "Duration",
    "Rainfall_Intensity_mm_per_h": "Mean intensity",
    "RF_over_AvgDepth": "Rainfall / avg. pit depth",
    "RF_over_MaxDepth": "Rainfall / max. pit depth",
    "Pit_Max_Depth_m": "Max. pit depth",
}

SPEARMAN_VARIABLES = [
    "Rainfall_mm",
    "Rainfall_Intensity_mm_per_h",
    "RF_over_AvgDepth",
    "RF_over_MaxDepth",
    "Pit_Max_Depth_m",
    "Rainfall_Duration_s",
]

LME_TERM_LABELS = {
    "Rainfall_mm": r"Total rainfall, $P$",
    "RF_over_AvgDepth": r"Rainfall / avg. pit depth, $\rho_{\mathrm{avg}}$",
    "Pit_Max_Depth_m": r"Max. pit depth, $h_{\max}$",
    "Rainfall_Duration_s": "Duration",
    "C(Rainfall_Pattern)[T.FRONT]": "FRONT vs BACK",
    "C(Rainfall_Pattern)[T.UNIFORM]": "UNIFORM vs BACK",
    "Rainfall_mm:Pit_Max_Depth_m": r"$P \times h_{\max}$",
    "RF_over_AvgDepth:Pit_Max_Depth_m": r"$\rho_{\mathrm{avg}} \times h_{\max}$",
}

LME_TERM_ORDER = [
    "Rainfall_mm",
    "Rainfall_mm:Pit_Max_Depth_m",
    "RF_over_AvgDepth",
    "RF_over_AvgDepth:Pit_Max_Depth_m",
    "Pit_Max_Depth_m",
    "Rainfall_Duration_s",
    "C(Rainfall_Pattern)[T.FRONT]",
    "C(Rainfall_Pattern)[T.UNIFORM]",
]


COL_REL = "#2F6F8F"
COL_ABS = "#D08A2E"
COL_POS = "#2F6F8F"
COL_NEG = "#B84A3A"
COL_NS = "#A9A9A9"


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_publication_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def p_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def get_pairwise_rho(spearman: pd.DataFrame, var: str, target: str) -> Tuple[float, float, int]:
    hit = spearman[
        ((spearman["var1"].eq(var)) & (spearman["var2"].eq(target)))
        | ((spearman["var2"].eq(var)) & (spearman["var1"].eq(target)))
    ]
    if hit.empty:
        return np.nan, np.nan, 0
    row = hit.iloc[0]
    return float(row["spearman_rho"]), float(row["p_value"]), int(row["n"])


def build_spearman_table(step2_outdir: Path) -> pd.DataFrame:
    spearman = pd.read_csv(step2_outdir / "step2_spearman_pairs.csv")
    rows = []
    for var in SPEARMAN_VARIABLES:
        rho_rel, p_rel, n_rel = get_pairwise_rho(spearman, var, "y_ratio_raw")
        rho_abs, p_abs, n_abs = get_pairwise_rho(spearman, var, "Final_Inundation_Depth_m")
        rows.append(
            {
                "variable": var,
                "label": VARIABLE_LABELS.get(var, var),
                "rho_relative": rho_rel,
                "p_relative": p_rel,
                "n_relative": n_rel,
                "rho_absolute": rho_abs,
                "p_absolute": p_abs,
                "n_absolute": n_abs,
            }
        )
    return pd.DataFrame(rows)


def load_idealized_events(events_path: Path) -> pd.DataFrame:
    df = pd.read_csv(events_path)
    if "Scenario_Type" in df.columns:
        df = df[df["Scenario_Type"].astype(str).str.upper().eq("IDEALIZED")].copy()

    numeric = [
        "Rainfall_mm",
        "Pit_Avg_Depth_m",
        "Pit_Max_Depth_m",
        "Rainfall_Duration_s",
        "Final_Inundation_Depth_m",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RF_over_AvgDepth"] = df["Rainfall_mm"] / df["Pit_Avg_Depth_m"].replace(0, np.nan)
    df["RF_over_MaxDepth"] = df["Rainfall_mm"] / df["Pit_Max_Depth_m"].replace(0, np.nan)
    df["Rainfall_Intensity_mm_per_h"] = df["Rainfall_mm"] / (df["Rainfall_Duration_s"] / 3600.0)
    df["Rainfall_mm:Pit_Max_Depth_m"] = df["Rainfall_mm"] * df["Pit_Max_Depth_m"]
    df["RF_over_AvgDepth:Pit_Max_Depth_m"] = df["RF_over_AvgDepth"] * df["Pit_Max_Depth_m"]
    df["y_ratio_raw"] = df["Final_Inundation_Depth_m"] / df["Pit_Max_Depth_m"].replace(0, np.nan)
    df["y_ratio_clip"] = df["y_ratio_raw"].clip(1e-6, 1 - 1e-6)
    df["y_logit"] = np.log(df["y_ratio_clip"] / (1 - df["y_ratio_clip"]))
    df["Rainfall_Pattern"] = df["Rainfall_Pattern"].astype(str).str.upper().str.strip()
    return df.replace([np.inf, -np.inf], np.nan)


def build_lme_table(step2_outdir: Path, events_df: pd.DataFrame) -> pd.DataFrame:
    lme = pd.read_csv(step2_outdir / "step2_lme_fixed_effects.csv")
    rows = []
    for _, row in lme.iterrows():
        term = row["term"]
        if term == "Intercept" or term not in LME_TERM_LABELS:
            continue
        coef = float(row["coef"])
        se = float(row["std_err"])
        p = float(row["p_value"])
        if term in events_df.columns:
            scale = float(events_df[term].std(ddof=0))
            effect_type = "standardized continuous"
        else:
            scale = 1.0
            effect_type = "categorical contrast"
        est = coef * scale
        lo = (coef - 1.96 * se) * scale
        hi = (coef + 1.96 * se) * scale
        rows.append(
            {
                "term": term,
                "label": LME_TERM_LABELS[term],
                "raw_coef": coef,
                "raw_std_err": se,
                "scale_factor": scale,
                "effect": est,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_value": p,
                "p_star": p_star(p),
                "significant_p05": bool(p < 0.05),
                "marginal_p10": bool((p >= 0.05) and (p < 0.10)),
                "effect_type": effect_type,
                "plot_order": LME_TERM_ORDER.index(term),
            }
        )
    return pd.DataFrame(rows).sort_values("plot_order", ascending=False)


def fit_spline_ols(events_df: pd.DataFrame, df_rf: int = 5, df_rainfall: int = 5):
    model_df = events_df.dropna(
        subset=[
            "y_logit",
            "RF_over_AvgDepth",
            "Rainfall_mm",
            "Pit_Max_Depth_m",
            "Rainfall_Pattern",
            "Rainfall_Duration_s",
        ]
    ).copy()
    rf_upper = max(float(model_df["RF_over_AvgDepth"].max()) * 1.001, 1.0)
    rain_upper = max(float(model_df["Rainfall_mm"].max()) * 1.001, 1.0)
    formula = (
        f"y_logit ~ bs(RF_over_AvgDepth, df={df_rf}, lower_bound=0, upper_bound={rf_upper:.12g})"
        f" + bs(Rainfall_mm, df={df_rainfall}, lower_bound=0, upper_bound={rain_upper:.12g})"
        " + Pit_Max_Depth_m + C(Rainfall_Pattern) + Rainfall_Duration_s"
    )
    y, x = dmatrices(formula, model_df, return_type="dataframe")
    fit = sm.OLS(y, x).fit()
    return fit, x.design_info, model_df, formula


def build_partial_response_table(events_df: pd.DataFrame) -> pd.DataFrame:
    fit, design_info, model_df, formula = fit_spline_ols(events_df)
    base = {
        "Rainfall_mm": float(model_df["Rainfall_mm"].median()),
        "RF_over_AvgDepth": float(model_df["RF_over_AvgDepth"].median()),
        "Pit_Max_Depth_m": float(model_df["Pit_Max_Depth_m"].median()),
        "Rainfall_Duration_s": float(model_df["Rainfall_Duration_s"].median()),
        "Rainfall_Pattern": "BACK",
    }
    rows = []
    specs = [
        ("RF_over_AvgDepth", "Rainfall / avg. depth", np.linspace(model_df["RF_over_AvgDepth"].quantile(0.05), model_df["RF_over_AvgDepth"].quantile(0.95), 160)),
        ("Rainfall_mm", "Total rainfall", np.linspace(model_df["Rainfall_mm"].min(), model_df["Rainfall_mm"].max(), 160)),
    ]
    for var, label, values in specs:
        pred_df = pd.DataFrame([base.copy() for _ in values])
        pred_df[var] = values
        x_pred = build_design_matrices([design_info], pred_df, return_type="dataframe")[0]
        pred_logit = fit.predict(x_pred)
        pred_ratio = expit(pred_logit)
        for value, logit, ratio in zip(values, pred_logit, pred_ratio):
            rows.append(
                {
                    "variable": var,
                    "label": label,
                    "x": float(value),
                    "pred_logit": float(logit),
                    "pred_ratio": float(ratio),
                    "held_Rainfall_mm": base["Rainfall_mm"],
                    "held_RF_over_AvgDepth": base["RF_over_AvgDepth"],
                    "held_Pit_Max_Depth_m": base["Pit_Max_Depth_m"],
                    "held_Rainfall_Duration_s": base["Rainfall_Duration_s"],
                    "held_Rainfall_Pattern": base["Rainfall_Pattern"],
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def plot_spearman_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    x = np.arange(len(table))
    width = 0.36
    ax.bar(x - width / 2, table["rho_absolute"], width=width, color=COL_ABS, alpha=0.90, label="Absolute final depth")
    ax.bar(x + width / 2, table["rho_relative"], width=width, color=COL_REL, alpha=0.92, label="Relative inundation ratio")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"], rotation=45, ha="right", fontsize=7.7)
    ax.set_ylabel("Spearman's ρ", fontsize=9)
    ax.set_ylim(-0.65, 0.92)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.tick_params(labelsize=7.7)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.015),
        borderaxespad=0.0,
        fontsize=7.2,
    )
    ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    for xi, (_, row) in enumerate(table.iterrows()):
        for dx, rho, p in [(-width / 2, row["rho_absolute"], row["p_absolute"]), (width / 2, row["rho_relative"], row["p_relative"] )]:
            if pd.isna(rho):
                continue
            va = "bottom" if rho >= 0 else "top"
            y = rho + (0.025 if rho >= 0 else -0.025)
            ax.text(xi + dx, y, p_star(p), ha="center", va=va, fontsize=6.8, color="#333333")


def plot_lme_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    y = np.arange(len(table))
    for yi, (_, row) in enumerate(table.iterrows()):
        sign_color = COL_POS if row["effect"] >= 0 else COL_NEG
        sig = bool(row["significant_p05"])
        marginal = bool(row["marginal_p10"])
        ecolor = sign_color if sig else ("#777777" if marginal else COL_NS)
        face = sign_color if sig else "white"
        alpha = 1.0 if sig else 0.78
        ax.errorbar(
            row["effect"],
            yi,
            xerr=np.array([[row["effect"] - row["ci95_low"]], [row["ci95_high"] - row["effect"]]]),
            fmt="none",
            ecolor=ecolor,
            elinewidth=0.95 if sig else 0.8,
            capsize=2.5,
            capthick=0.8,
            zorder=1,
            alpha=alpha,
        )
        ax.scatter(
            row["effect"],
            yi,
            s=32,
            facecolor=face,
            edgecolor=sign_color if sig else ecolor,
            linewidth=1.0,
            zorder=2,
            alpha=alpha,
        )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"], fontsize=8.2)
    ax.set_xlabel("Standardized LME effect on logit(relative inundation ratio)", fontsize=8.5)
    ax.tick_params(labelsize=7.8)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax.text(-0.12, 1.04, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

    ci_min = float(table["ci95_low"].min())
    ci_max = float(table["ci95_high"].max())
    ci_span = ci_max - ci_min
    ax.set_xlim(ci_min - 0.18 * ci_span, ci_max + 0.20 * ci_span)
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for yi, (_, row) in enumerate(table.iterrows()):
        if row["effect"] >= 0:
            x = row["ci95_high"] + 0.018 * span
            ha = "left"
        else:
            x = row["ci95_low"] - 0.018 * span
            ha = "right"
        ax.text(x, yi, row["p_star"], ha=ha, va="center", fontsize=7.2, color="#333333")

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_POS, markeredgecolor=COL_POS, label="Significant positive (p < 0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_NEG, markeredgecolor=COL_NEG, label="Significant negative (p < 0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COL_NS, label="Not significant (p ≥ 0.05)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=6.8, handletextpad=0.55, labelspacing=0.35)


def plot_spline_panel(ax1: plt.Axes, ax2: plt.Axes, table: pd.DataFrame) -> None:
    specs = [
        (ax1, "RF_over_AvgDepth", r"Rainfall / avg. pit depth, $\rho_{\mathrm{avg}}$"),
        (ax2, "Rainfall_mm", r"Total rainfall, $P$ (mm)"),
    ]
    for ax, var, xlabel in specs:
        sub = table[table["variable"].eq(var)].sort_values("x")
        ax.plot(sub["x"], sub["pred_ratio"], color=COL_REL, linewidth=1.8)
        ax.fill_between(sub["x"].to_numpy(), 0, sub["pred_ratio"].to_numpy(), color=COL_REL, alpha=0.08)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("Predicted relative ratio", fontsize=8.5, labelpad=1.5)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.tick_params(labelsize=7.8)
    ax1.text(-0.22, 1.10, "(c)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="bottom")


def write_metadata(outdir: Path, args: argparse.Namespace, spearman: pd.DataFrame, lme: pd.DataFrame, spline: pd.DataFrame) -> None:
    meta = {
        "figure": "Figure 7",
        "claim": "Rainfall forcing and rainfall-to-storage ratios dominate pit-scale inundation; LME interactions are weak while Spline-OLS captures nonlinear responses.",
        "scenario": "IDEALIZED only",
        "panel_a": "Spearman correlations between selected predictors and absolute final inundation depth or relative inundation ratio.",
        "panel_b": "Standardized interaction-LME fixed effects with 95% confidence intervals; blue and red mark significant positive and negative effects, respectively.",
        "panel_c": "Spline-OLS partial response curves, with all non-focal covariates held at their reference values; low-ratio boundary behaviour should be interpreted cautiously.",
        "significance": {"stars": {"***": "p < 0.001", "**": "p < 0.01", "*": "p < 0.05", "ns": "p >= 0.05"}},
        "input_step2_outdir": str(args.step2_outdir),
        "input_events": str(args.events),
        "n_spearman_rows": int(len(spearman)),
        "n_lme_terms": int(len(lme)),
        "n_spline_rows": int(len(spline)),
    }
    with open(outdir / "figure8_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure 8 response controls from Step2 outputs.")
    p.add_argument(
        "--step2-outdir",
        default="outputs/step2",
        help="Directory containing Step2 statistical outputs.",
    )
    p.add_argument(
        "--events",
        default="data/events_extracted.csv",
        help="Event-level table used to fit Spline-OLS partial responses and compute LME scaling.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/figure8",
        help="Output directory.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_matplotlib()
    outdir = ensure_dir(args.outdir)
    step2_outdir = Path(args.step2_outdir)
    events_df = load_idealized_events(Path(args.events))

    spearman_table = build_spearman_table(step2_outdir)
    lme_table = build_lme_table(step2_outdir, events_df)
    spline_table = build_partial_response_table(events_df)

    spearman_table.to_csv(outdir / "figure8_spearman_source_data.csv", index=False)
    lme_table.to_csv(outdir / "figure8_lme_effect_source_data.csv", index=False)
    spline_table.to_csv(outdir / "figure8_spline_partial_response_source_data.csv", index=False)
    write_metadata(outdir, args, spearman_table, lme_table, spline_table)

    # Nested quantitative-grid layout: keep a protective gap between A and the
    # long LME term labels, while preserving a tight B-C evidence chain.
    fig = plt.figure(figsize=(11.6, 4.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.18, 2.45], wspace=0.26)
    right = outer[0, 1].subgridspec(1, 2, width_ratios=[1.46, 1.04], wspace=0.18)
    cgrid = right[0, 1].subgridspec(2, 1, hspace=0.42)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(right[0, 0])
    ax_c1 = fig.add_subplot(cgrid[0, 0])
    ax_c2 = fig.add_subplot(cgrid[1, 0])

    plot_spearman_panel(ax_a, spearman_table)
    plot_lme_panel(ax_b, lme_table)
    plot_spline_panel(ax_c1, ax_c2, spline_table)
    fig.subplots_adjust(left=0.072, right=0.990, top=0.95, bottom=0.20)
    save_publication_figure(fig, outdir / "figure8_statistical_controls")
    plt.close(fig)
    print(f"[Figure8] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
