#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table 4-2: regression attribution and grouped prediction performance.

Panel A reports standardized continuous LME effects and categorical contrasts
from the interaction LME. Panel B reports grouped cross-validation metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TERM_META = {
    "Rainfall_mm": (
        "Total rainfall, P",
        "dominant positive forcing",
    ),
    "C(Rainfall_Pattern)[T.FRONT]": (
        "FRONT vs BACK",
        "lower conditional response than BACK",
    ),
    "C(Rainfall_Pattern)[T.UNIFORM]": (
        "UNIFORM vs BACK",
        "lower conditional response than BACK",
    ),
    "Pit_Max_Depth_m": (
        "Max. pit depth, hmax",
        "conditional main effect",
    ),
    "RF_over_AvgDepth": (
        "Rainfall ratio, ρavg",
        "conditional main effect",
    ),
    "Rainfall_Duration_s": (
        "Duration",
        "weak independent duration effect",
    ),
    "Rainfall_mm:Pit_Max_Depth_m": (
        "P × hmax",
        "weak interaction",
    ),
    "RF_over_AvgDepth:Pit_Max_Depth_m": (
        "ρavg × hmax",
        "weak interaction",
    ),
}

TERM_ORDER = [
    "Rainfall_mm",
    "C(Rainfall_Pattern)[T.FRONT]",
    "C(Rainfall_Pattern)[T.UNIFORM]",
    "Pit_Max_Depth_m",
    "RF_over_AvgDepth",
    "Rainfall_Duration_s",
    "Rainfall_mm:Pit_Max_Depth_m",
    "RF_over_AvgDepth:Pit_Max_Depth_m",
]

MODEL_META = {
    "baseline": ("Baseline", "reference"),
    "lme": ("LME with interactions", "attribution"),
    "spline_ols": ("Spline-OLS", "continuous prediction"),
}

METRIC_ORDER = ["ratio_MAE", "ratio_RMSE", "abs_MAE_m", "abs_RMSE_m"]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def p_text(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sig_text(effect: float, lo: float, hi: float, p: float) -> str:
    if p < 0.05 and lo > 0:
        return "positive"
    if p < 0.05 and hi < 0:
        return "negative"
    if p < 0.10:
        return "marginal/ns"
    return "ns"


def ci_relation(lo: float, hi: float) -> str:
    if lo > 0:
        return "CI > 0"
    if hi < 0:
        return "CI < 0"
    return "crosses 0"


def fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_signed(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.{digits}f}"


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def load_idealized_events(events_path: Path) -> pd.DataFrame:
    df = pd.read_csv(events_path)
    if "Scenario_Type" in df.columns:
        df = df[df["Scenario_Type"].astype(str).str.upper().eq("IDEALIZED")].copy()
    for col in [
        "Rainfall_mm",
        "Pit_Avg_Depth_m",
        "Pit_Max_Depth_m",
        "Rainfall_Duration_s",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["RF_over_AvgDepth"] = df["Rainfall_mm"] / df["Pit_Avg_Depth_m"].replace(0, np.nan)
    df["Rainfall_mm:Pit_Max_Depth_m"] = df["Rainfall_mm"] * df["Pit_Max_Depth_m"]
    df["RF_over_AvgDepth:Pit_Max_Depth_m"] = df["RF_over_AvgDepth"] * df["Pit_Max_Depth_m"]
    return df.replace([np.inf, -np.inf], np.nan)


def build_panel_a(step2_outdir: Path, events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lme = pd.read_csv(step2_outdir / "step2_lme_fixed_effects.csv")
    events = load_idealized_events(events_path)
    rows = []
    for term in TERM_ORDER:
        hit = lme[lme["term"].eq(term)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        coef = float(row["coef"])
        se = float(row["std_err"])
        p = float(row["p_value"])
        if term in events.columns:
            scale = float(events[term].std(ddof=0))
            effect_type = "standardized continuous effect"
        else:
            scale = 1.0
            effect_type = "categorical contrast"
        est = coef * scale
        lo = (coef - 1.96 * se) * scale
        hi = (coef + 1.96 * se) * scale
        label, interpretation = TERM_META[term]
        rows.append(
            {
                "Term": label,
                "Estimate": est,
                "95% CI low": lo,
                "95% CI high": hi,
                "p-value": p,
                "Direction": sig_text(est, lo, hi, p),
                "CI relation": ci_relation(lo, hi),
                "Interpretation": interpretation,
                "Effect type": effect_type,
                "Raw coefficient": coef,
                "Raw SE": se,
                "Scale factor": scale,
            }
        )
    detail = pd.DataFrame(rows)
    summary = detail[
        ["Term", "Estimate", "95% CI low", "95% CI high", "p-value", "Direction", "CI relation", "Interpretation"]
    ].copy()
    summary["Estimate"] = summary["Estimate"].map(fmt_signed)
    summary["95% CI"] = [fmt_ci(lo, hi) for lo, hi in zip(detail["95% CI low"], detail["95% CI high"])]
    summary["p-value"] = detail["p-value"].map(p_text)
    summary = summary[["Term", "Estimate", "95% CI", "p-value", "Direction", "CI relation", "Interpretation"]]
    return summary, detail


def metric_cell(row: pd.Series) -> str:
    return f"{fmt_num(row['mean'])} [{fmt_num(row['ci95_lo'])}, {fmt_num(row['ci95_hi'])}]"


def build_panel_b(step2_outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = pd.read_csv(step2_outdir / "step2_cv_summary_ci95.csv")
    cv = cv[cv["metric"].isin(METRIC_ORDER)].copy()
    detail = cv.copy()
    rows = []
    for model_key, (model_label, role) in MODEL_META.items():
        sub = cv[cv["model"].eq(model_key)].set_index("metric")
        out = {"Model": model_label}
        for metric in METRIC_ORDER:
            out[metric] = metric_cell(sub.loc[metric]) if metric in sub.index else ""
        out["Main role"] = role
        rows.append(out)
    summary = pd.DataFrame(rows).rename(
        columns={
            "ratio_MAE": "ratio MAE",
            "ratio_RMSE": "ratio RMSE",
            "abs_MAE_m": "abs MAE (m)",
            "abs_RMSE_m": "abs RMSE (m)",
        }
    )
    return summary, detail


def to_markdown(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> str:
    lines = [
        "# Table 4-2. Regression attribution and grouped prediction performance",
        "",
        "Panel A. Interaction LME fixed effects",
        "",
        panel_a.to_markdown(index=False, disable_numparse=True),
        "",
        "Panel B. Grouped cross-validation performance",
        "",
        panel_b.to_markdown(index=False, disable_numparse=True),
        "",
        "Note. Panel A reports standardized continuous effects and categorical contrasts from the interaction LME. "
        "Panel B reports fold-mean metrics with 95% confidence intervals in brackets.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(outdir: Path, panel_a: pd.DataFrame, panel_a_detail: pd.DataFrame, panel_b: pd.DataFrame, panel_b_detail: pd.DataFrame, args: argparse.Namespace) -> None:
    panel_a.to_csv(outdir / "table4_2_panel_a_lme_summary.csv", index=False)
    panel_a_detail.to_csv(outdir / "table4_2_panel_a_lme_detail.csv", index=False)
    panel_b.to_csv(outdir / "table4_2_panel_b_cv_summary.csv", index=False)
    panel_b_detail.to_csv(outdir / "table4_2_panel_b_cv_detail.csv", index=False)
    md = to_markdown(panel_a, panel_b)
    (outdir / "table4_2_regression_cv.md").write_text(md, encoding="utf-8")

    try:
        with pd.ExcelWriter(outdir / "table4_2_regression_cv.xlsx") as writer:
            panel_a.to_excel(writer, sheet_name="Panel A summary", index=False)
            panel_a_detail.to_excel(writer, sheet_name="Panel A detail", index=False)
            panel_b.to_excel(writer, sheet_name="Panel B summary", index=False)
            panel_b_detail.to_excel(writer, sheet_name="Panel B detail", index=False)
    except Exception as exc:
        (outdir / "table4_2_excel_write_warning.txt").write_text(str(exc), encoding="utf-8")

    meta = {
        "table": "Table 4-2",
        "panel_a_source": str(Path(args.step2_outdir) / "step2_lme_fixed_effects.csv"),
        "panel_b_source": str(Path(args.step2_outdir) / "step2_cv_summary_ci95.csv"),
        "events_source": str(args.events),
        "panel_a_note": "Continuous terms are standardized by the idealized-event SD; categorical terms are contrasts.",
        "panel_b_note": "Grouped cross-validation fold means with 95% confidence intervals.",
    }
    (outdir / "table4_2_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Table 4-2 from Step2 outputs.")
    p.add_argument(
        "--step2-outdir",
        default="outputs/step2",
        help="Directory containing Step2 statistical outputs.",
    )
    p.add_argument(
        "--events",
        default="data/events_extracted.csv",
        help="Event-level table used to standardize continuous LME effects.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/tables/table4_2",
        help="Output directory.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    panel_a, panel_a_detail = build_panel_a(Path(args.step2_outdir), Path(args.events))
    panel_b, panel_b_detail = build_panel_b(Path(args.step2_outdir))
    write_outputs(outdir, panel_a, panel_a_detail, panel_b, panel_b_detail, args)
    print(f"[Table4-2] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
