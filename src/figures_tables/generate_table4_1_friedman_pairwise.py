#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table 4-1: Friedman + Kendall's W + Holm-Wilcoxon pairwise summary.

Inputs are the Step2 outputs:
  - step2_friedman_summary.csv
  - step2_pairwise_wilcoxon_holm.csv

Outputs:
  - table4_1_friedman_pairwise_summary.csv
  - table4_1_friedman_pairwise_summary.md
  - table4_1_friedman_pairwise_full.csv
  - table4_1_friedman_pairwise.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PAIR_ORDER = ["FRONT vs BACK", "FRONT vs UNIFORM", "UNIFORM vs BACK"]


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_float(x: float, ndigits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{ndigits}f}"


def fmt_iqr(series: pd.Series, ndigits: int = 3) -> str:
    q1 = series.quantile(0.25)
    med = series.median()
    q3 = series.quantile(0.75)
    return f"{med:.{ndigits}f} [{q1:.{ndigits}f}-{q3:.{ndigits}f}]"


def sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pairwise_summary_text(sub: pd.DataFrame, pair: str) -> str:
    psub = sub[sub["pair"].eq(pair)].copy()
    if psub.empty:
        return ""
    if "friedman_p" in psub.columns:
        psub = psub[psub["friedman_p"] < 0.05].copy()
    n_total = len(psub)
    if n_total == 0:
        return "0/0 interpretable post-hoc tests"
    sig = psub[psub["significant_0_05"].astype(bool)].copy()
    if sig.empty:
        return f"0/{n_total} significant"

    direction_counts = sig["direction"].value_counts()
    dominant_direction = direction_counts.index[0]
    dominant_n = int(direction_counts.iloc[0])
    sig_n = int(len(sig))
    median_p = sig["p_holm"].median()
    return f"{sig_n}/{n_total} significant; dominant: {dominant_direction} ({dominant_n}/{sig_n}); median Holm p={fmt_p(median_p)}"


def build_summary_table(fried: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pair_with_omnibus = pair.merge(
        fried[["Rainfall_mm", "Rainfall_Duration_s", "friedman_p"]],
        on=["Rainfall_mm", "Rainfall_Duration_s"],
        how="left",
    )
    durations = sorted(fried["Rainfall_Duration_s"].dropna().unique())
    for duration in durations:
        fsub = fried[fried["Rainfall_Duration_s"].eq(duration)].copy()
        psub = pair_with_omnibus[pair_with_omnibus["Rainfall_Duration_s"].eq(duration)].copy()
        n_tests = len(fsub)
        n_sig = int((fsub["friedman_p"] < 0.05).sum())
        rows.append(
            {
                "Rainfall duration (s)": int(duration),
                "Rainfall levels tested": n_tests,
                "Friedman p < 0.05, n (%)": f"{n_sig}/{n_tests} ({100*n_sig/n_tests:.1f}%)",
                "Friedman chi-square, median [IQR]": fmt_iqr(fsub["friedman_chi2"]),
                "Kendall's W, median [IQR]": fmt_iqr(fsub["kendall_w"]),
                "Kendall's W range": f"{fsub['kendall_w'].min():.3f}-{fsub['kendall_w'].max():.3f}",
                "FRONT vs BACK, Holm-adjusted p < 0.05": pairwise_summary_text(psub, "FRONT vs BACK"),
                "FRONT vs UNIFORM, Holm-adjusted p < 0.05": pairwise_summary_text(psub, "FRONT vs UNIFORM"),
                "UNIFORM vs BACK, Holm-adjusted p < 0.05": pairwise_summary_text(psub, "UNIFORM vs BACK"),
            }
        )

    n_tests = len(fried)
    n_sig = int((fried["friedman_p"] < 0.05).sum())
    rows.append(
        {
            "Rainfall duration (s)": "Overall",
            "Rainfall levels tested": n_tests,
            "Friedman p < 0.05, n (%)": f"{n_sig}/{n_tests} ({100*n_sig/n_tests:.1f}%)",
            "Friedman chi-square, median [IQR]": fmt_iqr(fried["friedman_chi2"]),
            "Kendall's W, median [IQR]": fmt_iqr(fried["kendall_w"]),
            "Kendall's W range": f"{fried['kendall_w'].min():.3f}-{fried['kendall_w'].max():.3f}",
            "FRONT vs BACK, Holm-adjusted p < 0.05": pairwise_summary_text(pair_with_omnibus, "FRONT vs BACK"),
            "FRONT vs UNIFORM, Holm-adjusted p < 0.05": pairwise_summary_text(pair_with_omnibus, "FRONT vs UNIFORM"),
            "UNIFORM vs BACK, Holm-adjusted p < 0.05": pairwise_summary_text(pair_with_omnibus, "UNIFORM vs BACK"),
        }
    )
    return pd.DataFrame(rows)


def build_full_table(fried: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, frow in fried.sort_values(["Rainfall_Duration_s", "Rainfall_mm"]).iterrows():
        rain = frow["Rainfall_mm"]
        duration = frow["Rainfall_Duration_s"]
        out = {
            "Rainfall_mm": rain,
            "Rainfall_Duration_s": int(duration),
            "n_pits": int(frow["n_pits"]),
            "friedman_chi2": frow["friedman_chi2"],
            "friedman_p": frow["friedman_p"],
            "friedman_sig": sig_star(frow["friedman_p"]),
            "posthoc_interpretable": bool(frow["friedman_p"] < 0.05),
            "kendall_w": frow["kendall_w"],
        }
        psub = pair[pair["Rainfall_mm"].eq(rain) & pair["Rainfall_Duration_s"].eq(duration)].copy()
        for pair_name in PAIR_ORDER:
            prow = psub[psub["pair"].eq(pair_name)]
            safe = pair_name.lower().replace(" ", "_").replace(">", "gt").replace("<", "lt")
            if prow.empty:
                out[f"{safe}_direction"] = ""
                out[f"{safe}_p_holm"] = np.nan
                out[f"{safe}_sig"] = ""
            else:
                prow = prow.iloc[0]
                out[f"{safe}_direction"] = prow["direction"]
                out[f"{safe}_p_holm"] = prow["p_holm"]
                out[f"{safe}_sig"] = bool(prow["significant_0_05"])
        rows.append(out)
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Table 4-1. Rainfall-pattern effects on pit-scale inundation depth\n\n")
        f.write(
            "Friedman tests compare FRONT, BACK, and UNIFORM within each rainfall amount-duration combination "
            "using Pit_ID as the repeated unit (n = 9 pits). Pairwise comparisons use Wilcoxon signed-rank tests "
            "with Holm adjustment and are summarized only for combinations with significant Friedman omnibus tests.\n\n"
        )
        f.write(summary.to_markdown(index=False))
        f.write(
            "\n\nNotes: Kendall's W is the Friedman effect size. "
            "Statistical significance is defined as p < 0.05. "
            "For pairwise post-hoc comparisons, significance refers to Holm-adjusted p < 0.05. "
            "The 0.05 threshold corresponds to a 95% confidence level and is not a 95th percentile of the data distribution. "
            "Only idealized scenarios are included; DESIGN_CHICAGO cases are excluded from this table.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Table 4-1 from Step2 statistical outputs.")
    parser.add_argument(
        "--step2-outdir",
        default="outputs/step2",
        help="Directory containing Step2 CSV outputs.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/tables/table4_1",
        help="Directory for Table 4-1 outputs.",
    )
    args = parser.parse_args()

    step2 = Path(args.step2_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fried = pd.read_csv(step2 / "step2_friedman_summary.csv")
    pair = pd.read_csv(step2 / "step2_pairwise_wilcoxon_holm.csv")
    pair["significant_0_05"] = pair["significant_0_05"].astype(bool)

    summary = build_summary_table(fried, pair)
    full = build_full_table(fried, pair)

    summary.to_csv(outdir / "table4_1_friedman_pairwise_summary.csv", index=False)
    full.to_csv(outdir / "table4_1_friedman_pairwise_full.csv", index=False)
    write_markdown(summary, outdir / "table4_1_friedman_pairwise_summary.md")

    try:
        with pd.ExcelWriter(outdir / "table4_1_friedman_pairwise.xlsx", engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Table4-1_summary", index=False)
            full.to_excel(writer, sheet_name="Full_60_tests", index=False)
    except Exception as exc:
        print(f"[Table4-1] Excel export skipped: {exc}")

    print(f"[Table4-1] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
